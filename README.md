# SAM Image Processing — Drone Imagery Semantic Segmentation

End-to-end pipeline for **semantic segmentation** of drone imagery, combining Meta's Segment Anything Model (SAM 2) for automatic mask generation with supervised training of **DeepLabv3+** / **PSPNet** models on CVAT-annotated datasets stored in Azure Blob Storage.

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
| **Environment setup** | Installs SAM 2, PyTorch (CUDA 11.8), pycocotools, Azure SDK |
| **Grid-based prompting** | Samples single-point prompts across the image |
| **Mask filtering** | Quality scoring + non-maximal suppression to deduplicate |
| **COCO export** | Converts masks to COCO-format JSON for ingestion by CVAT |
| **Azure integration** | Reads/writes imagery and annotations from Azure Blob Storage |

**Use case:** Pre-annotate drone images before manual refinement in CVAT.

---

### 2. `semantic_segmentation_training.ipynb` — Supervised Segmentation Training

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datasantana/sam-image-processing/blob/main/notebooks/semantic_segmentation_training.ipynb)

Trains pixel-level classification models on CVAT-exported masks. The notebook is divided into four major sections:

#### Section 1 — Azure Blob Connection & Download
- Connects to Azure Blob Storage using a connection string (stored as a Colab secret).
- Downloads drone images and their corresponding PASCAL-VOC-style annotation masks.
- Downloads the `labelmap.txt` that maps class names to RGB colors.

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
Azure Blob Storage
  ├── drone-imagery/              (source images)
  ├── pascual_annotation_masks/
  │     ├── SegmentationClass/    (CVAT-exported RGB masks)
  │     └── labelmap.txt          (class name → RGB mapping)
  └──────────────────────────────────────
          │  download_blobs()
          ▼
  Local: assets/images/   assets/masks/   assets/labelmap.txt
          │
          ▼
  RGBSegmentationDataset  →  Albumentations  →  DataLoader  →  Model
```

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
| **Runtime** | Google Colab with GPU (CUDA 11.8 recommended) |
| **Azure** | A Blob Storage account with a `general-purpose` container containing the imagery and CVAT masks |
| **Secret** | `AZURE_CONNECTION_STRING` stored as a Colab user secret |
| **Annotation Tool** | CVAT with PASCAL VOC / Segmentation Mask export |

## Quick Start

1. Open either notebook in Google Colab using the badge links above.
2. Set your `AZURE_CONNECTION_STRING` in Colab Secrets.
3. Run all cells sequentially from top to bottom.
4. For the training notebook, monitor the Loss/IoU plots to check for overfitting.
5. Download the saved `.pth` model checkpoint from `models/`.

## Dependencies

See [requirements.txt](requirements.txt) for the full list. Key packages:

```
azure-storage-blob >= 12.17.0
opencv-python      >= 4.8.0
pillow             >= 10.0.0
numpy              >= 1.24.0
matplotlib         >= 3.7.0
torch + torchvision (CUDA 11.8)
segmentation-models-pytorch >= 0.3.3
albumentations     >= 1.3.0
timm               >= 0.9.0
scikit-learn       >= 1.3.0
tqdm               >= 4.65.0
tensorboard        >= 2.13.0
```

## License

See individual notebook headers for attribution. SAM 2 components are adapted from [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) under their original license.
