# SAM Image Processing — Drone Imagery Semantic Segmentation

End-to-end pipeline for **semantic segmentation** of drone imagery, combining Meta's Segment Anything Model (SAM 2) for automatic mask generation with supervised training of **DeepLabv3+** / **PSPNet** models on CVAT-annotated datasets.

> **Data source.** The dataset is distributed as a single verified archive (`segpack_v1.zip`) in Google Drive and fetched by `segpipe.data_io` (see [Dataset](#dataset)). Azure Blob Storage is no longer used.

---

## Repository Structure

```
sam-image-processing/
├── README.md                              # This file
├── INSTRUCTIONS.md                        # Deep context for future work
├── requirements.txt                       # Python dependencies
└── notebooks/
    ├── automatic_mask_generator_example.ipynb   # SAM 2 auto-mask + COCO export
    └── semantic_segmentation_training.ipynb     # DeepLabv3+/PSPNet training
```

## Notebooks Overview

### 1. `automatic_mask_generator_example.ipynb` — SAM 2 Automatic Mask Generation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datasantana/sam-image-processing/blob/main/notebooks/automatic_mask_generator_example.ipynb)

Generates zero-shot segmentation masks using Meta's **SAM 2** (`SAM2AutomaticMaskGenerator`). Key steps:

| Step | Description |
|------|-------------|
| **Environment setup** | Installs SAM 2 and pycocotools (uses Colab's preinstalled PyTorch) |
| **Image source** | Fetches the segpack with `segpipe.data_io.ensure_dataset` to the VM's local disk |
| **Grid-based prompting** | Samples single-point prompts across the image |
| **Mask filtering** | Quality scoring + non-maximal suppression to deduplicate |
| **COCO export** | Converts masks to COCO-format JSON for ingestion by CVAT |
| **Results** | Written to the local disk; optionally copied to a mounted Google Drive folder |

**Use case:** Pre-annotate drone images before manual refinement in CVAT.

---

### 2. `semantic_segmentation_training.ipynb` — Supervised Segmentation Training

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datasantana/sam-image-processing/blob/main/notebooks/semantic_segmentation_training.ipynb)

Trains pixel-level classification models on CVAT-exported masks. The notebook is divided into four major sections:

#### Section 1 — Data download
- *Historical:* this notebook originally downloaded images, masks and `labelmap.txt` from an Azure Blob container that no longer exists. Data now comes from the segpack archive (see [Dataset](#dataset)).

#### Section 2 — Dataset Preparation
- Pairs images with masks by matching base filenames.
- Configurable train/val split: `TRAIN_RATIO` (default 0.8) with `RANDOM_SEED` (default 42) using `sklearn.model_selection.train_test_split`.
- Reproducible across runs with a fixed random seed.

#### Section 3 — Model Training Setup
| Sub-section | What it does |
|-------------|--------------|
| **3.1 Deep Learning Environment** | Installs PyTorch (CUDA 11.8), `segmentation-models-pytorch`, `albumentations`, `timm`, TensorBoard |
| **3.2 Model Configuration** | Selects architecture (`DeepLabv3+` or `PSPNet`), backbone (`resnet18`–`resnet152`), and hyperparameters (batch size, LR, epochs, image size) |
| **3.3 Class Labels & Label Map** | Parses `labelmap.txt` (supports `ClassName:R,G,B` format), builds RGB→class-index mapping, auto-detects whether masks are RGB or grayscale |
| **3.4 Data Transforms & Dataset** | Custom `RGBSegmentationDataset` with vectorized 256³ LUT for RGB→index conversion (~10-50× faster than per-pixel loops); Albumentations augmentations (flips, rotations, brightness, noise); separate train/val transforms |
| **3.5 Model Initialization** | Instantiates the model with ImageNet-pretrained encoder, DiceLoss, Adam optimizer, ReduceLROnPlateau scheduler |
| **3.6 Forward Pass Test** | Validates the full pipeline (input → model → output) with `model.eval()` to avoid BatchNorm issues on single-sample batches |

#### Section 4 — Training Loop & Visualization
- Epoch-level training with `tqdm` progress bars.
- Tracks **DiceLoss** and **per-class mean IoU (mIoU)** per epoch for accurate multi-class evaluation.
- **Early stopping** halts training when validation mIoU stops improving for N epochs (`EARLY_STOPPING_PATIENCE`, default 10).
- **Learning rate tracking** records actual LR per epoch for accurate schedule plots.
- Saves the **best model** (by validation mIoU) and the **final model** as `.pth` checkpoints containing model weights, optimizer state, scheduler state, class names, and architecture metadata.
- Generates a 4-panel matplotlib figure: Train/Val Loss, Train/Val mIoU, LR schedule history, and a text summary.

#### Section 5 — Reusable Evaluation Metrics (Wilk et al. 2022–compatible)
Provides benchmark-comparable evaluation using the four standard metrics from the semantic segmentation literature:

| Metric | Description |
|--------|-------------|
| **mIoU** | Mean Intersection over Union across all classes |
| **OA** | Overall pixel-level accuracy |
| **mAcc** | Mean per-class accuracy (recall) |
| **mF1** | Mean per-class F1-score |

Key components:
- `SegmentationMetrics` class — confusion-matrix-based accumulator that derives all four metrics.
- `evaluate_model()` — runs the trained model on a DataLoader and returns full metrics.
- `plot_training_curves()` — 4-panel figure with corrected LR tracking.
- `plot_confusion_matrix()` — normalized heatmap per class.
- `plot_wilk_comparison()` — grouped bar chart comparing results against Wilk et al. (2022).
- CSV export of per-class metrics and experiment summaries.

#### Section 6 — Post-Training Evaluation & Benchmark Comparison
Runs the full evaluation pipeline after training:
1. Loads the best checkpoint and evaluates on the validation set.
2. Prints and exports per-class metrics to `logs/metrics_per_class.csv`.
3. Exports a one-row experiment summary to `logs/metrics_summary.csv` (appends across runs).
4. Plots confusion matrix and Wilk et al. comparison chart.
5. Prints a delta table showing how our results compare to the published benchmarks.

---

## Core Concepts

### Semantic Segmentation
Assigns a class label to **every pixel** in an image (e.g., "road", "building", "vegetation"). Unlike object detection (bounding boxes) or instance segmentation (per-object masks), semantic segmentation produces a single dense label map.

### Architectures Used
- **DeepLabv3+** — Uses atrous (dilated) convolutions and an encoder-decoder structure with Atrous Spatial Pyramid Pooling (ASPP) for multi-scale context.
- **PSPNet** (Pyramid Scene Parsing Network) — Applies a Pyramid Pooling Module to aggregate context at multiple scales before final classification.
- Both use a **ResNet** backbone (pre-trained on ImageNet) as the feature extractor.

### Loss Function — DiceLoss
Measures the overlap between predicted and ground-truth masks. Defined as $1 - \frac{2|P \cap G|}{|P| + |G|}$. Works well for class-imbalanced segmentation tasks.

### IoU (Intersection over Union)
The primary evaluation metric: $\text{IoU} = \frac{|P \cap G|}{|P \cup G|}$. Ranges from 0 (no overlap) to 1 (perfect overlap). Values above **0.7** generally indicate good segmentation quality.

### Benchmark Reference — Wilk et al. (2022)
Results are compared against the metrics reported in:
> Wilk, Ł. et al. (2022). "Automatic classification of roof types from 3D building models using semantic segmentation." *ISPRS Archives*, XLIII-B2-2022, 485–492.

| Metric | Wilk (image) | Wilk (texture voting) |
|--------|-------------:|----------------------:|
| mIoU   | 60.3%        | 63.8%                 |
| OA     | 84.1%        | 89.9%                 |
| mAcc   | 81.4%        | 89.8%                 |
| mF1    | 72.0%        | 73.5%                 |

### Data Pipeline
```
Google Drive: segpack_v1.zip  (private; file ID in configs/datasets/pascual.yaml)
          │  segpipe.data_io.ensure_dataset()
          │  download with the user's account (Drive API) │ or mounted Drive │ or SEGPACK_ZIP/SEGPACK_DIR
          │  verify size + sha256 → extract → validate against manifest.csv
          ▼
  VM local disk: /content/data/segpack_v1/{images,masks,classes.json,manifest.csv,...}
          │
          ▼
  Dataset  →  Albumentations  →  DataLoader  →  Model
```

### Dataset

`segpack_v1.zip` (560,108,771 bytes, sha256 `1ef52bcb…f41ab`) contains 66 oblique UAV photos (4032×2268) of the Pascual Guerrero stadium area, their CVAT RGB masks, `classes.json` (ordered list of `index`, `name`, `rgb`; source of truth for the class schema), `labelmap.txt`, `manifest.csv` (hashes, sizes, DJI GPS/gimbal metadata), `class_stats.csv` and `REPORT.txt`. It was built with [`tools/pack_dataset.py`](tools/pack_dataset.py). A manual backup copy exists on OneDrive (link kept by the author; not automated).

### RGB Mask Handling
CVAT exports masks where each class is encoded as a distinct RGB color. The notebook:
1. Parses `labelmap.txt` to extract `ClassName:R,G,B` entries.
2. Converts RGB triplets to BGR (OpenCV convention) for lookup.
3. Pre-builds a flat 256³ NumPy lookup table for O(1) vectorized per-pixel conversion.
4. Falls back to vectorized nearest-color matching when exact pixel colors aren't found in the map.

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Runtime** | Google Colab with GPU (uses the PyTorch build preinstalled by Colab) |
| **Data** | Read access to `segpack_v1.zip` in Google Drive with the Google account used in Colab |
| **Annotation Tool** | CVAT with Segmentation Mask 1.1 export (only needed to rebuild the pack) |

## Quick Start

1. Open the notebook in Google Colab using the badge links above.
2. Clone this repo in the VM and run the dataset cell: it authenticates with your Google account and downloads/validates the pack.
3. Run the remaining cells.

## Dependencies

See [requirements.txt](requirements.txt) (minimum versions only; PyTorch is not pinned). Local CPU tests: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt && pytest`.

## License

See individual notebook headers for attribution. SAM 2 components are adapted from [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) under their original license.
