# INSTRUCTIONS — Deep Context for Future Improvements

This document captures the architectural decisions, known limitations, and concrete directions for improving the `sam-image-processing` project. It is written for contributors (human or AI) who will iterate on the codebase.

---

## 1. Current State Summary

### What exists today

| Artifact | Purpose | Maturity |
|----------|---------|----------|
| `automatic_mask_generator_example.ipynb` | Zero-shot mask generation with SAM 2 → COCO JSON export for CVAT | Working prototype; adapted from Meta's reference notebook |
| `semantic_segmentation_training.ipynb` | End-to-end training of DeepLabv3+/PSPNet on CVAT-annotated drone imagery stored in Azure Blob Storage | Functional pipeline; no automated tests, no CLI, no experiment tracking beyond TensorBoard |

### What does NOT exist yet
- No Python package / module structure — everything lives inside notebooks.
- No inference script or deployment pipeline.
- No CI/CD, linting, or automated testing.
- ~~No `requirements.txt` or `pyproject.toml`.~~ → `requirements.txt` added (2026-03-03).
- No Dockerfile or reproducible environment beyond "run on Colab".
- No model registry or versioned artifact storage.
- ~~No per-class IoU evaluation or confusion matrix.~~ → Per-class mIoU in training loop + `SegmentationMetrics` class for post-training evaluation (2026-03-03).

---

## 2. Architecture Decisions & Rationale

### 2.1 Why notebooks instead of scripts?
The primary user is running on **Google Colab** with a GPU. Notebooks provide immediate visual feedback (plots, printed tables) and require zero local setup. The trade-off is that notebooks are hard to version-control, test, and reuse.

**Future direction:** Extract reusable logic into a `src/` Python package and keep notebooks as thin orchestration layers that import from it.

### 2.2 Why Azure Blob Storage?
The team already stores drone imagery and CVAT exports in Azure. The notebook downloads data at runtime instead of committing large files to Git. This keeps the repo lightweight.

**Future direction:** Abstract the storage backend behind a simple interface so it can be swapped for S3, GCS, or a local path.

### 2.3 Why DiceLoss instead of CrossEntropy?
DiceLoss directly optimizes the overlap metric and handles class imbalance better than pixel-wise cross-entropy for segmentation tasks where background dominates. The current implementation uses `smp.losses.DiceLoss(mode='multiclass')`.

**Future direction:** Experiment with combined losses like `DiceLoss + FocalLoss` or `DiceLoss + CrossEntropy` for improved boundary precision.

### 2.4 Why a fixed train/val split (first 53)?
The split is deterministic to ensure reproducibility. However, 53 is hardcoded and assumes a specific dataset size.

**Future direction:** Replace with a configurable ratio-based split (e.g., 80/20) using `sklearn.model_selection.train_test_split` with a fixed `random_state`.

### 2.5 Why pixel-level RGB→index conversion at runtime?
CVAT exports masks as RGB images. Converting at training time keeps the raw data intact and avoids a preprocessing step. The downside is significant overhead — the current `rgb_to_class_indices()` method uses nested Python loops over every pixel.

**Future direction:** Vectorize the conversion using NumPy broadcasting or pre-convert masks to index format offline and store them as single-channel PNGs.

---

## 3. Known Limitations & Technical Debt

### 3.1 Performance bottleneck: RGB mask conversion

> **✅ RESOLVED** (2026-03-03)
> Replaced per-pixel Python loop with a pre-built 256³ NumPy lookup table (`_build_lookup_table()`) that converts the entire mask in a single vectorized indexing operation. The slow-path fallback now also uses vectorized broadcasting instead of nested loops. Expected ~10-50× speedup on 512×512 masks.

### 3.2 IoU calculation is binary, not per-class

> **✅ RESOLVED** (2026-03-03)
> Replaced `calculate_iou()` with `calculate_miou()` that computes IoU independently for each class and returns the mean. Training loop now reports per-class mean IoU (mIoU) instead of binary IoU.

### 3.3 No early stopping

> **✅ RESOLVED** (2026-03-03)
> Added `EARLY_STOPPING_PATIENCE` (default: 10 epochs). Training loop now tracks `epochs_without_improvement` and breaks early when patience is exhausted. Configurable via the `EARLY_STOPPING_PATIENCE` variable in the training cell.

### 3.4 Labelmap parsing is fragile
The parser assumes `ClassName:R,G,B` format but does not handle headers, comments, or alternate delimiters. If the file starts with a header line (e.g., CVAT's default `label_name:r:g:b:a::` format), parsing silently produces wrong mappings.

**Fix:** Add robust parsing that detects the format and handles edge cases (header rows, alpha channels, different delimiters).

### 3.5 No data validation
There is no check that image dimensions match mask dimensions, that mask pixel values are within the expected class range, or that all classes in the labelmap appear in the masks.

**Fix:** Add a pre-training validation pass that scans the full dataset and reports anomalies.

### 3.6 Learning rate plot is incorrect

> **✅ RESOLVED** (2026-03-03)
> The training loop now appends `current_lr` to a `learning_rates` list at the start of each epoch. Cell 26 plots this list directly, showing the actual ReduceLROnPlateau schedule.

### 3.7 No standardized evaluation metrics (Wilk et al. 2022 comparison)
The current pipeline only tracks a single binary IoU during training. It does **not** compute the four standard metrics used in the semantic segmentation literature — specifically the ones reported by Wilk et al. (2022) for UAV roof-type classification:

| Metric | Wilk et al. (image) | Wilk et al. (texture voting) |
|--------|---------------------|------------------------------|
| mIoU   | 60.3%               | 63.8%                        |
| OA     | 84.1%               | 89.9%                        |
| mAcc   | 81.4%               | 89.8%                        |
| mF1    | 72.0%               | 73.5%                        |

Without these, results cannot be compared to the literature.

**Fix priority: HIGH**

Implement a reusable `SegmentationMetrics` class + comparison utilities (see §12 for full specification). The fix involves:
1. Replacing `calculate_iou()` with a confusion-matrix-based metric computer that yields mIoU, OA, mAcc, mF1 per epoch.
2. Adding a post-training evaluation section that runs the trained model on the full val set, computes per-class metrics, and exports results to CSV.
3. Adding a benchmark comparison chart that overlays our results against Wilk et al. (2022).

---

## 4. Recommended Future Improvements

### 4.1 Short-Term (next 1–2 iterations)

All short-term improvements have been completed. See §4.4 Completed.

### 4.4 Completed

| # | Improvement | Status | Date |
|---|-------------|--------|------|
| 1 | **Vectorize RGB mask conversion** (§3.1) | ✅ Done — 256³ LUT in `_build_lookup_table()` | 2026-03-03 |
| 2 | **Per-class IoU metrics** (§3.2) | ✅ Done — `calculate_miou()` replaces binary IoU | 2026-03-03 |
| 3 | **Early stopping** (§3.3) | ✅ Done — `EARLY_STOPPING_PATIENCE` configurable | 2026-03-03 |
| 4 | **Fix LR plot** (§3.6) | ✅ Done — `learning_rates` list tracked per epoch | 2026-03-03 |
| 5 | **Add `requirements.txt`** | ✅ Done — root-level `requirements.txt` | 2026-03-03 |
| 6 | **Configurable train/val split ratio** | ✅ Done — `TRAIN_RATIO` + `RANDOM_SEED` + `train_test_split` | 2026-03-03 |
| 6b | **Wilk et al. (2022) benchmark metrics** (§3.7, §12) | ✅ Done — `SegmentationMetrics` class + CSV export + comparison charts | 2026-03-03 |

### 4.2 Medium-Term (next 3–5 iterations)

| # | Improvement | Impact | Effort |
|---|-------------|--------|--------|
| 7 | **Extract code into `src/` package** | Testable, reusable code; thin notebooks | Medium |
| 8 | **Add a CLI inference script** | Run predictions without a notebook | Medium |
| 9 | **Experiment with combined losses** (DiceLoss + FocalLoss) | Better boundary precision, minority class recall | Low–Medium |
| 10 | **Add a confusion matrix visualization** | Visual per-class error analysis | Low |
| 11 | **Mixed-precision training** (`torch.cuda.amp`) | ~2× speedup, lower GPU memory | Low |
| 12 | **TensorBoard integration** (already installed, not wired) | Live training monitoring in Colab | Low |
| 13 | **Abstract storage backend** | Support S3, GCS, local | Medium |
| 14 | **Pre-convert masks to index format** | Eliminate runtime RGB conversion entirely | Medium |
| 23 | **DeepLabV3+ ASPP dilation tuning + boundary loss** (§13.1) | Fine-grained edge recovery on small roof elements | Medium |
| 24 | **PSPNet multi-scale inference + mosaic tiling** (§13.2) | Reduce tile-edge artifacts, capture global context | Medium |
| 25 | **UNet++ deep supervision + CCE+Focal Tversky loss** (§13.3) | Improve minority-class recall via auxiliary losses + targeted sampling | Medium |

### 4.3 Long-Term

| # | Improvement | Impact | Effort |
|---|-------------|--------|--------|
| 15 | **Hydra / OmegaConf config system** | Declarative experiment configuration | Medium |
| 16 | **Weights & Biases or MLflow tracking** | Full experiment lineage, hyperparameter sweeps | Medium |
| 17 | **CI pipeline** (GitHub Actions) | Lint, test, notebook smoke tests on every push | Medium |
| 18 | **Docker image for training** | Reproducible environment beyond Colab | Medium |
| 19 | **ONNX / TorchScript export** | Deploy model to edge devices or web services | Medium |
| 20 | **Active learning loop** | Use SAM 2 to pre-annotate → human review in CVAT → retrain | High |
| 21 | **Multi-GPU / distributed training** | Scale to larger datasets | High |
| 22 | **Test with additional architectures** (UNet++, SegFormer, Mask2Former) | Find best accuracy/speed trade-off for drone imagery | Medium |

---

## 5. How to Add a New Model Architecture

The current design uses `segmentation-models-pytorch` (SMP), which makes this straightforward:

1. In cell 13 (Model Configuration), add a new `elif` branch:
   ```python
   elif MODEL_ARCH == "UnetPlusPlus":
       model_class = smp.UnetPlusPlus
   ```
2. No other changes are needed — the rest of the pipeline (loss, optimizer, dataset) is architecture-agnostic.
3. Available SMP architectures: `Unet`, `UnetPlusPlus`, `MAnet`, `Linknet`, `FPN`, `PSPNet`, `DeepLabV3`, `DeepLabV3Plus`, `PAN`.

---

## 6. How to Add a New Dataset

1. Upload images and masks to Azure Blob Storage under a known prefix.
2. Update `LOCAL_IMG_DIR`, `LOCAL_MASK_DIR`, and `LOCAL_LABELMAP_DIR` in cell 6.
3. Ensure the `labelmap.txt` follows the `ClassName:R,G,B` format (one class per line, no header).
4. If masks are grayscale (pixel value = class index), the pipeline auto-detects this and skips RGB conversion.
5. The train/val split is controlled by `TRAIN_RATIO` (default 0.8) and `RANDOM_SEED` (default 42) in cell 8. Adjust as needed.

---

## 7. Hyperparameter Tuning Guide

| Parameter | Current Value | Suggested Range | Notes |
|-----------|--------------|-----------------|-------|
| `MODEL_ARCH` | `PSPNet` | `DeepLabv3+`, `PSPNet`, `UnetPlusPlus` | Start with DeepLabv3+ for best general segmentation quality |
| `BACKBONE` | `resnet50` | `resnet34` (faster) to `resnet101` (more capacity) | `resnet50` is a good default; `resnet101` if you have enough data |
| `BATCH_SIZE` | `8` | `4`–`16` | Limited by GPU memory; reduce if OOM |
| `LEARNING_RATE` | `1e-4` | `1e-5` to `5e-4` | Lower if training is unstable; higher if convergence is too slow |
| `NUM_EPOCHS` | `50` | `30`–`150` | Use early stopping to find the right number |
| `EARLY_STOPPING_PATIENCE` | `10` | `5`–`20` | Epochs without val mIoU improvement before stopping |
| `TRAIN_RATIO` | `0.8` | `0.7`–`0.9` | Fraction of data for training (rest is validation) |
| `RANDOM_SEED` | `42` | Any integer | Fixed seed for reproducible train/val split |
| `IMAGE_SIZE` | `512` | `256`–`1024` | Larger = better detail but more memory; 512 is a solid default |
| Augmentation strength | Moderate | — | Increase if overfitting; decrease if underfitting |
| Loss function | DiceLoss | DiceLoss, FocalLoss, CE+Dice | Combined losses may improve boundary precision |

---

## 8. Glossary of Key Concepts

| Term | Definition |
|------|-----------|
| **Semantic segmentation** | Classifying every pixel in an image into a predefined category |
| **SAM 2** | Meta's Segment Anything Model v2 — a foundation model for zero-shot segmentation |
| **DeepLabv3+** | Encoder-decoder architecture using atrous convolutions and ASPP for multi-scale segmentation |
| **PSPNet** | Pyramid Scene Parsing Network — uses a pyramid pooling module for global context |
| **ResNet backbone** | Pre-trained convolutional network (ImageNet) used as the feature extractor/encoder |
| **DiceLoss** | Overlap-based loss function: $1 - \frac{2\|P \cap G\|}{\|P\| + \|G\|}$ |
| **IoU** | Intersection over Union — primary metric for segmentation quality |
| **CVAT** | Computer Vision Annotation Tool — open-source labeling platform |
| **PASCAL VOC masks** | Segmentation masks where each class is a distinct RGB color in a PNG image |
| **Albumentations** | Fast image augmentation library with mask-aware transforms |
| **SMP** | `segmentation-models-pytorch` — library providing pre-built segmentation architectures |
| **Atrous/Dilated convolution** | Convolution with spacing between kernel elements to increase receptive field without losing resolution |
| **ASPP** | Atrous Spatial Pyramid Pooling — parallel atrous convolutions at multiple rates for multi-scale features |
| **ReduceLROnPlateau** | Scheduler that halves the learning rate when the monitored metric stops improving |
| **Overfitting** | Model memorizes training data but fails to generalize to unseen data (train metric >> val metric) |

---

## 9. File-Level Reference

### `notebooks/semantic_segmentation_training.ipynb`

| Cell(s) | Section | Key Variables / Functions |
|---------|---------|--------------------------|
| 1–2 | Title & Colab badge | — |
| 3 | Install deps | `pip install` commands |
| 4 | Intro markdown | — |
| 5–6 | Azure connection & download | `AZURE_CONN_STR`, `CONTAINER_NAME`, `download_blobs()` |
| 7–8 | Dataset preparation | `TRAIN_RATIO`, `RANDOM_SEED`, `image_files`, `mask_files`, `train_pairs`, `val_pairs`, `train_dataset`, `val_dataset` |
| 9–11 | DL env setup | PyTorch, SMP, albumentations install |
| 12–13 | Model config | `MODEL_ARCH`, `BACKBONE`, `BATCH_SIZE`, `LEARNING_RATE`, `NUM_EPOCHS`, `IMAGE_SIZE` |
| 14 | Beginner guide (Spanish) | Explains Loss, IoU, overfitting in plain language |
| 15–16 | Label map & RGB detection | `class_names`, `NUM_CLASSES`, `rgb_to_class`, `class_to_idx` |
| 17–18 | Dataset & transforms | `RGBSegmentationDataset` (vectorized LUT), `train_transforms`, `val_transforms`, `train_loader`, `val_loader` |
| 19–20 | Model init | `model`, `criterion` (DiceLoss), `optimizer` (Adam), `scheduler` (ReduceLROnPlateau) |
| 21–22 | Forward pass test | Validates shapes with `model.eval()` |
| 23–24 | Training loop | `train_one_epoch()`, `validate_one_epoch()`, `calculate_miou()`, `EARLY_STOPPING_PATIENCE`, `learning_rates`, checkpoint saving |
| 25–26 | Visualization | matplotlib 4-panel figure, saved to `logs/training_results.png` |

### `notebooks/automatic_mask_generator_example.ipynb`

| Section | Purpose |
|---------|---------|
| Environment setup | Installs SAM 2, PyTorch, pycocotools, Azure SDK |
| Model loading | Loads SAM 2 checkpoint |
| Mask generation | `SAM2AutomaticMaskGenerator` with grid-based prompts |
| COCO export | Converts masks to COCO JSON for CVAT import |
| Azure integration | Reads/writes imagery and annotations from Blob Storage |

---

## 10. Conventions for Contributors

1. **Notebook cells should be idempotent** — re-running a cell should not produce side effects.
2. **Print emoji prefixes** — use the established convention: ✅ success, ❌ error, ⚠️ warning, 🔍 informational, 📊 metrics, 💾 save, 🎯 key outputs.
3. **Keep notebooks thin** — when extracting to `src/`, notebooks should only contain imports, configuration, and visualization. All logic should live in importable modules.
4. **Azure secrets** — never commit connection strings. Always use Colab Secrets (`userdata.get()`) or environment variables.
5. **Model checkpoints** — always save architecture name, backbone, `NUM_CLASSES`, and `class_names` alongside weights so checkpoints are self-describing.

---

## 11. Contribution Rules

### 11.1 Branching & Commits

1. **Never push directly to `main`.** Create a feature or fix branch (`fix/vectorize-rgb-mask`, `feat/early-stopping`, etc.) and open a Pull Request.
2. **One logical change per PR.** Do not bundle unrelated fixes. If a PR fixes the IoU calculation and also adds early stopping, split it into two PRs.
3. **Write descriptive commit messages.** Use the format:
   ```
   <type>(<scope>): <short summary>

   <optional body explaining why, not just what>
   ```
   Types: `fix`, `feat`, `docs`, `refactor`, `perf`, `test`, `chore`.  
   Example: `fix(dataset): vectorize RGB mask conversion to eliminate per-pixel loop`

### 11.2 Code Quality

1. **Test before committing.** At minimum, run the affected notebook end-to-end on Colab (or locally with a small data subset) and verify no cells error out.
2. **No hardcoded secrets or paths.** All credentials must come from environment variables or Colab Secrets. Local paths must be relative or configurable.
3. **Type hints and docstrings.** Any new function must include a docstring (Google style) and type hints for parameters and return values.
4. **Keep backward compatibility.** If changing a function signature, preserve the old interface with a deprecation warning for at least one release cycle.

### 11.3 Documentation Update After Every Fix (Mandatory)

> **Rule: Every code change must be accompanied by a corresponding documentation update in the same PR. PRs that modify behavior without updating documentation should not be merged.**

After completing any fix, feature, or refactor, update **all** of the following that apply:

| What changed | What to update |
|-------------|----------------|
| **Bug fix or behavior change** | Update the relevant cell's markdown header/comments in the notebook. If the fix resolves a known limitation listed in INSTRUCTIONS.md §3, mark it as resolved with the PR reference and date. |
| **New feature or parameter** | Add it to the notebook's markdown documentation cells, the README.md (Concepts / Quick Start sections), and the INSTRUCTIONS.md hyperparameter table (§7) or architecture guide (§5). |
| **New dependency** | Add to the `!pip install` cell in the notebook, the README.md Dependencies section, and `requirements.txt` (once it exists). |
| **Removed or deprecated feature** | Note the removal in INSTRUCTIONS.md with the date and rationale. Update README.md to remove references. |
| **Performance improvement** | Log before/after benchmarks in the PR description and update INSTRUCTIONS.md §3 (Known Limitations) to reflect the resolved bottleneck. |
| **New notebook or script** | Add an entry to README.md (Notebooks Overview) and INSTRUCTIONS.md (File-Level Reference, §9). |

#### Documentation update checklist (copy into every PR description)

```markdown
### Documentation Checklist
- [ ] Notebook markdown cells updated (comments, headers, descriptions)
- [ ] README.md updated (if user-facing behavior changed)
- [ ] INSTRUCTIONS.md updated (limitations resolved, new improvements, reference tables)
- [ ] Inline code comments explain *why*, not just *what*
- [ ] No stale references remain (old parameter names, removed functions, outdated paths)
```

### 11.4 Resolving Known Limitations

When fixing a limitation listed in §3 of this document:

1. Implement the fix on a dedicated branch.
2. In INSTRUCTIONS.md §3, **do not delete** the limitation entry. Instead, update it:
   ```markdown
   ### 3.1 ~~Performance bottleneck: RGB mask conversion~~ ✅ RESOLVED
   **Resolved in:** PR #<number> (<date>)  
   **Solution:** <one-line summary of what was done>
   
   <keep the original description below for historical context>
   ```
3. If the fix introduces new tunable parameters or trade-offs, add them to §7 (Hyperparameter Tuning Guide).
4. Move the corresponding item in §4 (Recommended Future Improvements) to a new **§4.4 Completed** table.

### 11.5 Reviewing & Merging

1. **Self-review first.** Re-read your own diff before requesting review. Check for leftover debug prints, commented-out code, and missing documentation updates.
2. **At least one approval required** before merging to `main`.
3. **Squash-merge** feature branches to keep `main` history linear and readable.
4. **Delete the branch** after merging.

### 11.6 Issue Tracking

1. When discovering a new bug or limitation during development, **open a GitHub Issue** immediately — even if you plan to fix it in the same session.
2. Reference the Issue number in your commit messages and PR description (`Fixes #12`).
3. For improvement ideas, add them to INSTRUCTIONS.md §4 *and* create a GitHub Issue labeled `enhancement` so they are discoverable outside this file.

---

## 12. Wilk et al. (2022) Benchmark Metrics — Full Specification

> **Reference:** Wilk, Ł. et al. (2022). "Automatic classification of roof types from 3D building models using semantic segmentation." *ISPRS Archives*, XLIII-B2-2022, 485–492.

This section specifies the evaluation framework needed to produce results directly comparable with Wilk et al. (2022). All functions are designed to be **reusable** — they accept a model, a DataLoader, and class metadata, and return structured outputs.

### 12.1 Metric Definitions

Given a confusion matrix $C$ of shape $(K \times K)$ where $K$ is the number of classes and $C_{ij}$ counts pixels with true class $i$ predicted as class $j$:

| Metric | Formula | Description |
|--------|---------|-------------|
| **OA** (Overall Accuracy) | $\frac{\sum_i C_{ii}}{\sum_{i,j} C_{ij}}$ | Fraction of all pixels correctly classified |
| **Per-class Accuracy** | $\text{Acc}_i = \frac{C_{ii}}{\sum_j C_{ij}}$ | Recall for class $i$ |
| **mAcc** (Mean Accuracy) | $\frac{1}{K} \sum_i \text{Acc}_i$ | Average recall across classes |
| **Per-class IoU** | $\text{IoU}_i = \frac{C_{ii}}{\sum_j C_{ij} + \sum_j C_{ji} - C_{ii}}$ | Jaccard index for class $i$ |
| **mIoU** (Mean IoU) | $\frac{1}{K} \sum_i \text{IoU}_i$ | Average IoU across classes |
| **Per-class Precision** | $\text{Prec}_i = \frac{C_{ii}}{\sum_j C_{ji}}$ | Precision for class $i$ |
| **Per-class F1** | $\text{F1}_i = \frac{2 \cdot \text{Prec}_i \cdot \text{Acc}_i}{\text{Prec}_i + \text{Acc}_i}$ | Harmonic mean of precision and recall |
| **mF1** (Mean F1) | $\frac{1}{K} \sum_i \text{F1}_i$ | Average F1 across classes |

### 12.2 Implementation Plan

The implementation adds **three new sections** to the training notebook after the existing visualization cell:

#### Section 5: Reusable Metrics Functions

**Cell 5.1 — `SegmentationMetrics` class**

```python
class SegmentationMetrics:
    """
    Accumulates a confusion matrix across batches
    and derives mIoU, OA, mAcc, mF1 (per-class and mean).
    """
    def __init__(self, num_classes, class_names=None, device='cpu'):
        self.num_classes = num_classes
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        self.device = device
        self.confusion_matrix = torch.zeros(num_classes, num_classes,
                                            dtype=torch.int64, device=device)

    def reset(self): ...
    def update(self, preds, targets): ...  # accumulate batch into CM
    def compute(self) -> dict:             # return all metrics
    def to_dataframe(self) -> pd.DataFrame # per-class + mean row
    def export_csv(self, path): ...        # save DataFrame
```

Key design decisions:
- Uses a **full confusion matrix** (not running averages) for numerical stability.
- `update()` accepts raw logits or argmax predictions and long targets.
- `compute()` returns a dict: `{'mIoU', 'OA', 'mAcc', 'mF1', 'per_class': {cls: {iou, acc, prec, f1}}}` .
- `to_dataframe()` returns a pandas DataFrame suitable for display and CSV export.

**Cell 5.2 — `evaluate_model()` helper**

Runs the model on an entire DataLoader in `eval()` mode and returns a `SegmentationMetrics` result.

**Cell 5.3 — `plot_training_curves()` helper**

Generates the standard 4-panel figure (Loss, IoU, LR, summary text) from history lists. This replaces the current ad-hoc cell 26 and **fixes the LR plot bug** (§3.6) by accepting a `learning_rates` list.

**Cell 5.4 — `plot_wilk_comparison()` helper**

Grouped bar chart comparing our metrics against Wilk et al. (2022) baselines.

**Cell 5.5 — `plot_confusion_matrix()` helper**

Normalized heatmap of the confusion matrix with class labels.

#### Section 6: Post-Training Evaluation

After training completes, this section:
1. Loads the best checkpoint.
2. Runs `evaluate_model()` on `val_loader`.
3. Prints and exports per-class metrics to `logs/metrics_per_class.csv`.
4. Plots the confusion matrix → `logs/confusion_matrix.png`.
5. Plots the Wilk et al. comparison chart → `logs/wilk_comparison.png`.

### 12.3 Wilk et al. (2022) Reference Values

These constants should be defined in the notebook for programmatic comparison:

```python
WILK_2022_BENCHMARKS = {
    'image': {
        'mIoU': 60.3,
        'OA':   84.1,
        'mAcc': 81.4,
        'mF1':  72.0,
    },
    'texture_voting': {
        'mIoU': 63.8,
        'OA':   89.9,
        'mAcc': 89.8,
        'mF1':  73.5,
    },
}
```

### 12.4 CSV Output Format

The exported `metrics_per_class.csv` should have this structure:

| class | iou | accuracy | precision | f1 |
|-------|-----|----------|-----------|----|
| class_0 | 0.XX | 0.XX | 0.XX | 0.XX |
| ... | ... | ... | ... | ... |
| **mean** | **0.XX** | **0.XX** | **0.XX** | **0.XX** |

An additional `metrics_summary.csv` should contain:

| model | backbone | epochs | mIoU | OA | mAcc | mF1 | best_val_iou | training_hours |
|-------|----------|--------|------|-----|------|-----|--------------|----------------|

This makes it trivial to aggregate results across experiments.

### 12.5 Integration with Training Loop

The training loop (cell 24) should be updated to:
1. Track `learning_rates` list (fixes §3.6).
2. Optionally compute full metrics every N epochs (configurable, default = only at the end) to avoid slowing down training.
3. Save the `SegmentationMetrics` result in the final checkpoint alongside model weights.

### 12.6 Comparison Interpretation Guide

For the beginner-guide markdown cell (currently in Spanish), add a subsection explaining:
- What each metric means in plain language.
- How to read the Wilk comparison chart.
- What "good" values look like (contextualized to drone imagery).
- Why mIoU is typically lower than OA (class imbalance; OA is dominated by majority classes).

---

## 13. Architecture-Specific Refinements — Implementation Plans

This section specifies per-architecture improvements that go beyond the default SMP configuration. Each subsection includes **rationale, implementation plan, PyTorch code, and new hyperparameters** to add to §7. All refinements preserve the existing metrics pipeline (§5–§6, §12).

> **Design principle:** The notebook's Model Configuration cell (§3.2) already switches on `MODEL_ARCH`. Each refinement below adds architecture-specific logic gated behind that same switch, so the notebook remains a single unified pipeline.

---

### 13.1 DeepLabV3+ Refinements

#### 13.1.1 Custom ASPP Dilation Rates

**Problem:** The default SMP `DeepLabV3Plus` uses ASPP dilation rates `(12, 24, 36)` tuned for natural-scene images at ~513×513. For drone imagery with fine-grained structures (roof edges, small annexes), these rates can be too coarse.

**Goal:** Expose configurable dilation rates that capture smaller spatial details.

**Implementation:**

```python
# ── DeepLabV3+ ASPP dilation configuration ──────────────────────────
# Add to the Model Configuration cell, inside the DeepLabV3+ branch

# Configurable dilation rates (tune per dataset)
ASPP_DILATIONS = (6, 12, 18)  # Default SMP: (12, 24, 36)
                               # Smaller rates → finer detail
                               # Suggested experiments: (6,12,18), (3,6,9), (6,12,24)

if MODEL_ARCH == "DeepLabv3+":
    model = smp.DeepLabV3Plus(
        encoder_name=BACKBONE,
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
        decoder_atrous_rates=ASPP_DILATIONS,
    )
    print(f"🎯 DeepLabV3+ created with ASPP dilations: {ASPP_DILATIONS}")
```

**New hyperparameter for §7:**

| Parameter | Current Value | Suggested Range | Notes |
|-----------|--------------|-----------------|-------|
| `ASPP_DILATIONS` | `(12, 24, 36)` | `(3,6,9)` to `(12,24,36)` | Smaller = finer detail; must divide `IMAGE_SIZE` evenly for best results |

#### 13.1.2 Learning Rate Scheduler Options

**Problem:** The current `ReduceLROnPlateau` reacts only after stagnation. `CosineAnnealingWarmRestarts` can provide smoother convergence with periodic warm restarts.

**Goal:** Make the scheduler configurable per architecture.

```python
# ── Scheduler selection ─────────────────────────────────────────────
SCHEDULER_TYPE = "cosine"  # Options: "plateau", "cosine"

if SCHEDULER_TYPE == "plateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
elif SCHEDULER_TYPE == "cosine":
    # T_0 = restart period in epochs; T_mult = period multiplier after each restart
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
else:
    raise ValueError(f"Unknown scheduler: {SCHEDULER_TYPE}")

print(f"⚙️ Scheduler: {SCHEDULER_TYPE}")

# ── Training loop integration ───────────────────────────────────────
# Replace `scheduler.step(val_loss)` with:
if SCHEDULER_TYPE == "plateau":
    scheduler.step(val_loss)
elif SCHEDULER_TYPE == "cosine":
    scheduler.step()  # CosineAnnealing steps per epoch, not per metric
```

**New hyperparameters for §7:**

| Parameter | Current Value | Suggested Range | Notes |
|-----------|--------------|-----------------|-------|
| `SCHEDULER_TYPE` | `"plateau"` | `"plateau"`, `"cosine"` | Cosine works well for longer training; plateau for short runs |
| `T_0` (cosine) | `10` | `5`–`20` | Initial restart period |
| `T_mult` (cosine) | `2` | `1`–`3` | Period multiplier after each restart |

#### 13.1.3 Boundary-Aware Loss (Dice + Boundary IoU)

**Problem:** DiceLoss alone treats all pixels equally. Boundary pixels are critical for roof-edge delineation but are vastly outnumbered by interior pixels.

**Goal:** Add a boundary-focused term that upweights edge pixels.

```python
# ── Boundary-aware combined loss ────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


class BoundaryLoss(nn.Module):
    """
    Computes a boundary-focused loss by extracting GT boundary pixels
    (via morphological erosion) and penalizing errors on those pixels
    more heavily.
    """
    def __init__(self, num_classes: int, kernel_size: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def _extract_boundaries(self, masks: torch.Tensor) -> torch.Tensor:
        """Extract boundary pixels from ground-truth masks."""
        masks_np = masks.cpu().numpy().astype(np.uint8)
        boundaries = np.zeros_like(masks_np, dtype=np.float32)
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)

        for i in range(masks_np.shape[0]):
            eroded = cv2.erode(masks_np[i], kernel, iterations=1)
            boundaries[i] = (masks_np[i] != eroded).astype(np.float32)

        return torch.from_numpy(boundaries).to(masks.device)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Boundary-weighted cross-entropy loss."""
        boundary_mask = self._extract_boundaries(targets)  # (N, H, W)
        ce_per_pixel = self.ce(logits, targets)             # (N, H, W)

        # Weight: 1.0 for interior, BOUNDARY_WEIGHT for boundary pixels
        weight_map = 1.0 + boundary_mask * (BOUNDARY_WEIGHT - 1.0)
        return (ce_per_pixel * weight_map).mean()


class CombinedDiceBoundaryLoss(nn.Module):
    """
    Combined loss: α * DiceLoss + β * BoundaryLoss
    """
    def __init__(self, num_classes: int, dice_weight: float = 0.7,
                 boundary_weight: float = 0.3, boundary_kernel: int = 3,
                 boundary_pixel_weight: float = 5.0):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode='multiclass')
        self.boundary = BoundaryLoss(num_classes, kernel_size=boundary_kernel)
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        # Store globally so BoundaryLoss.forward() can access it
        global BOUNDARY_WEIGHT
        BOUNDARY_WEIGHT = boundary_pixel_weight

    def forward(self, logits, targets):
        return (self.dice_weight * self.dice(logits, targets) +
                self.boundary_weight * self.boundary(logits, targets))


# ── Usage (in Model Initialization cell) ────────────────────────────
USE_BOUNDARY_LOSS = True  # Set to False to use plain DiceLoss

if MODEL_ARCH == "DeepLabv3+" and USE_BOUNDARY_LOSS:
    criterion = CombinedDiceBoundaryLoss(
        num_classes=NUM_CLASSES,
        dice_weight=0.7,
        boundary_weight=0.3,
        boundary_kernel=3,
        boundary_pixel_weight=5.0,
    )
    print("⚖️ Loss: DiceLoss(0.7) + BoundaryLoss(0.3) — edge-aware")
else:
    criterion = smp.losses.DiceLoss(mode='multiclass')
    print("⚖️ Loss: DiceLoss (standard)")
```

**New hyperparameters for §7:**

| Parameter | Current Value | Suggested Range | Notes |
|-----------|--------------|-----------------|-------|
| `USE_BOUNDARY_LOSS` | `False` | `True`/`False` | Enable for DeepLabV3+ edge refinement |
| `dice_weight` | `0.7` | `0.5`–`0.8` | Weight for DiceLoss in combined loss |
| `boundary_weight` | `0.3` | `0.2`–`0.5` | Weight for BoundaryLoss |
| `boundary_pixel_weight` | `5.0` | `3.0`–`10.0` | How much more boundary pixels matter vs interior |
| `boundary_kernel` | `3` | `3`–`7` | Erosion kernel; larger = thicker boundary band |

#### 13.1.4 Metrics Integration

No changes needed — the `SegmentationMetrics` class (§12) and post-training evaluation (Section 6) are architecture-agnostic. Just ensure the training loop passes `learning_rates` to `plot_training_curves()` regardless of scheduler type.

---

### 13.2 PSPNet Refinements

#### 13.2.1 Multi-Scale Testing at Inference

**Problem:** PSPNet's Pyramid Pooling Module captures context at fixed scales during training. At inference, running the model at multiple input scales and averaging predictions can recover fine details lost at the default resolution.

**Goal:** Implement a `multi_scale_predict()` function that runs inference at several scales and fuses the results.

```python
# ── Multi-scale inference for PSPNet ────────────────────────────────
import torch
import torch.nn.functional as F
from typing import List


def multi_scale_predict(model: torch.nn.Module,
                        image: torch.Tensor,
                        scales: List[float] = [0.5, 0.75, 1.0, 1.25, 1.5],
                        flip: bool = True,
                        num_classes: int = None,
                        device: str = 'cuda') -> torch.Tensor:
    """
    Multi-scale inference with optional horizontal flip augmentation.

    Args:
        model: Trained segmentation model.
        image: (1, 3, H, W) normalized input tensor.
        scales: List of scale factors to test.
        flip: Whether to also test horizontally-flipped inputs.
        num_classes: Number of classes (auto-detected from model output if None).
        device: Target device.

    Returns:
        (H, W) tensor of predicted class indices.
    """
    model.eval()
    _, _, H, W = image.shape
    accumulated = torch.zeros(1, num_classes or 1, H, W, device=device)

    with torch.no_grad():
        for scale in scales:
            new_h, new_w = int(H * scale), int(W * scale)
            # Ensure dimensions are divisible by 8 (PSPNet requirement)
            new_h = (new_h // 8) * 8
            new_w = (new_w // 8) * 8

            scaled = F.interpolate(image, size=(new_h, new_w),
                                   mode='bilinear', align_corners=True)
            scaled = scaled.to(device)

            # Forward pass
            output = model(scaled)
            if num_classes is None:
                num_classes = output.shape[1]
                accumulated = torch.zeros(1, num_classes, H, W, device=device)

            # Upsample back to original size
            output = F.interpolate(output, size=(H, W),
                                   mode='bilinear', align_corners=True)
            accumulated += F.softmax(output, dim=1)

            if flip:
                # Horizontal flip
                flipped = torch.flip(scaled, dims=[3])
                output_f = model(flipped)
                output_f = torch.flip(output_f, dims=[3])
                output_f = F.interpolate(output_f, size=(H, W),
                                         mode='bilinear', align_corners=True)
                accumulated += F.softmax(output_f, dim=1)

    return accumulated.argmax(dim=1).squeeze(0)  # (H, W)


# ── Usage in evaluation ─────────────────────────────────────────────
MULTI_SCALE_INFERENCE = True  # Enable for PSPNet
INFERENCE_SCALES = [0.5, 0.75, 1.0, 1.25, 1.5]
INFERENCE_FLIP = True

def evaluate_model_multiscale(model, dataloader, num_classes, class_names,
                              scales, flip, device):
    """Evaluate model with multi-scale inference."""
    metrics = SegmentationMetrics(num_classes, class_names, device)
    model.eval()
    for images, masks in tqdm(dataloader, desc="🔬 Multi-scale eval"):
        for i in range(images.shape[0]):
            img = images[i:i+1].to(device)
            pred = multi_scale_predict(
                model, img, scales=scales, flip=flip,
                num_classes=num_classes, device=device
            )
            metrics.update(pred.unsqueeze(0), masks[i:i+1].to(device))
    return metrics
```

**New hyperparameters for §7:**

| Parameter | Current Value | Suggested Range | Notes |
|-----------|--------------|-----------------|-------|
| `MULTI_SCALE_INFERENCE` | `False` | `True`/`False` | Enable for PSPNet at eval time (slower but better mIoU) |
| `INFERENCE_SCALES` | `[0.5, 0.75, 1.0, 1.25, 1.5]` | 3–7 values in `[0.5, 2.0]` | More scales = better accuracy but slower |
| `INFERENCE_FLIP` | `True` | `True`/`False` | Horizontal flip TTA |

#### 13.2.2 Mosaic Tiling with 50% Overlap and Central Weighting

**Problem:** When the input image is larger than `IMAGE_SIZE`, tiles are processed independently. Hard tile boundaries cause visible seams in the prediction.

**Goal:** Implement a tiling strategy with 50% overlap and a Gaussian center-weighting kernel that blends overlapping predictions smoothly.

```python
# ── Mosaic tiling with overlapping + Gaussian weighting ─────────────
import torch
import torch.nn.functional as F
import numpy as np


def create_gaussian_weight(tile_size: int, sigma_factor: float = 0.25) -> np.ndarray:
    """
    Create a 2D Gaussian weight map for tile blending.
    Center pixels get weight ~1.0, edges approach ~0.0.
    """
    sigma = tile_size * sigma_factor
    ax = np.arange(tile_size) - tile_size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.max()  # Normalize peak to 1.0


def predict_mosaic(model: torch.nn.Module,
                   image: np.ndarray,
                   tile_size: int = 512,
                   overlap: float = 0.5,
                   num_classes: int = 14,
                   transform=None,
                   device: str = 'cuda') -> np.ndarray:
    """
    Predict on a large image using overlapping tiles with Gaussian blending.

    Args:
        model: Trained segmentation model.
        image: (H, W, 3) numpy array, uint8, BGR or RGB.
        tile_size: Size of each square tile.
        overlap: Fraction of overlap between adjacent tiles (0.5 = 50%).
        num_classes: Number of output classes.
        transform: Albumentations transform (must include Normalize + ToTensorV2).
        device: Target device.

    Returns:
        (H, W) numpy array of predicted class indices.
    """
    model.eval()
    H, W, _ = image.shape
    stride = int(tile_size * (1 - overlap))

    # Accumulators
    prob_sum = np.zeros((num_classes, H, W), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)
    gaussian_weight = create_gaussian_weight(tile_size)

    # Generate tile coordinates
    y_starts = list(range(0, H - tile_size + 1, stride))
    x_starts = list(range(0, W - tile_size + 1, stride))

    # Ensure coverage of bottom-right edges
    if y_starts[-1] + tile_size < H:
        y_starts.append(H - tile_size)
    if x_starts[-1] + tile_size < W:
        x_starts.append(W - tile_size)

    with torch.no_grad():
        for y in y_starts:
            for x in x_starts:
                tile = image[y:y+tile_size, x:x+tile_size]

                # Apply transform
                if transform:
                    augmented = transform(image=tile)
                    tile_tensor = augmented['image'].unsqueeze(0).to(device)
                else:
                    tile_tensor = torch.from_numpy(
                        tile.transpose(2, 0, 1).astype(np.float32) / 255.0
                    ).unsqueeze(0).to(device)

                output = model(tile_tensor)
                probs = F.softmax(output, dim=1).squeeze(0).cpu().numpy()  # (C, H, W)

                # Accumulate with Gaussian weighting
                for c in range(num_classes):
                    prob_sum[c, y:y+tile_size, x:x+tile_size] += probs[c] * gaussian_weight
                weight_sum[y:y+tile_size, x:x+tile_size] += gaussian_weight

    # Normalize
    weight_sum = np.maximum(weight_sum, 1e-8)
    for c in range(num_classes):
        prob_sum[c] /= weight_sum

    return prob_sum.argmax(axis=0)  # (H, W)


# ── Configuration ───────────────────────────────────────────────────
USE_MOSAIC_TILING = True   # Enable for full-image prediction
TILE_OVERLAP = 0.5         # 50% overlap
GAUSSIAN_SIGMA = 0.25      # Gaussian weight sigma as fraction of tile_size
```

**New hyperparameters for §7:**

| Parameter | Current Value | Suggested Range | Notes |
|-----------|--------------|-----------------|-------|
| `USE_MOSAIC_TILING` | `False` | `True`/`False` | Enable for images larger than `IMAGE_SIZE` |
| `TILE_OVERLAP` | `0.5` | `0.25`–`0.75` | Higher overlap = smoother blending but slower |
| `GAUSSIAN_SIGMA` | `0.25` | `0.15`–`0.35` | Controls fall-off; smaller = sharper center weighting |

#### 13.2.3 Adaptive Pyramid Pooling Module Kernel Sizes

**Problem:** SMP's PSPNet uses fixed pooling bin sizes `(1, 2, 3, 6)` regardless of input resolution. When `IMAGE_SIZE` changes, these bins may not align optimally.

**Goal:** Dynamically adjust the PPM output sizes relative to the actual input feature map size.

**Implementation note:** SMP's `PSPNet` constructor accepts `psp_out_channels` and `psp_use_batchnorm` but does **not** expose the pooling bin sizes directly. To customize them, we need to **replace the PPM module** after model creation.

```python
# ── Adaptive PPM kernel sizes for PSPNet ────────────────────────────
import torch.nn as nn


class AdaptivePPM(nn.Module):
    """
    Pyramid Pooling Module with configurable bin sizes.
    Replaces the default SMP PPM to allow tuning:
      pool_sizes relative to input spatial dimensions.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 pool_sizes: tuple = (1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList()
        for size in pool_sizes:
            self.stages.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=size),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ))

    def forward(self, x):
        h, w = x.shape[2:]
        pyramids = [x]
        for stage in self.stages:
            out = stage(x)
            out = F.interpolate(out, size=(h, w), mode='bilinear',
                                align_corners=True)
            pyramids.append(out)
        return torch.cat(pyramids, dim=1)


# ── Usage ───────────────────────────────────────────────────────────
# After creating the PSPNet model, replace the PPM:
PSP_POOL_SIZES = (1, 2, 3, 6)  # Default
# For IMAGE_SIZE=512: try (1, 2, 4, 8)
# For IMAGE_SIZE=256: try (1, 2, 3, 6)
# For IMAGE_SIZE=1024: try (1, 3, 6, 12)

if MODEL_ARCH == "PSPNet":
    # Get the encoder output channels
    encoder_out_ch = model.encoder.out_channels[-1]
    psp_out_ch = model.decoder.psp.stages[0][1].out_channels  # per-branch channels

    # Replace PPM
    model.decoder.psp = AdaptivePPM(
        in_channels=encoder_out_ch,
        out_channels=psp_out_ch,
        pool_sizes=PSP_POOL_SIZES,
    )
    model = model.to(device)
    print(f"🏗️ PSPNet PPM replaced with pool sizes: {PSP_POOL_SIZES}")
```

**New hyperparameters for §7:**

| Parameter | Current Value | Suggested Range | Notes |
|-----------|--------------|-----------------|-------|
| `PSP_POOL_SIZES` | `(1,2,3,6)` | Adapt to `IMAGE_SIZE` | Larger images → larger max bin. General rule: max bin ≈ `IMAGE_SIZE / 64` |

#### 13.2.4 Metrics Integration

When `MULTI_SCALE_INFERENCE` is enabled, the post-training evaluation cell (Section 6) should call `evaluate_model_multiscale()` instead of `evaluate_model()`. Both return a `SegmentationMetrics` instance, so the rest of the pipeline (CSV export, Wilk comparison) is unchanged.

```python
# ── In the post-training evaluation cell ────────────────────────────
if MODEL_ARCH == "PSPNet" and MULTI_SCALE_INFERENCE:
    val_metrics = evaluate_model_multiscale(
        model, val_loader, NUM_CLASSES, class_names,
        scales=INFERENCE_SCALES, flip=INFERENCE_FLIP, device=device
    )
else:
    val_metrics = evaluate_model(model, val_loader, NUM_CLASSES, class_names, device)
```

---

### 13.3 UNet++ Refinements

#### 13.3.1 Deep Supervision on Intermediate Nodes

**Problem:** UNet++ has dense skip connections forming intermediate "nodes" at each decoder level. By default, only the final output is supervised. Activating **deep supervision** adds auxiliary loss on intermediate outputs, regularizing the network and improving gradient flow.

**Goal:** Create the UNet++ with deep supervision enabled and average the auxiliary losses during training.

```python
# ── UNet++ with deep supervision ────────────────────────────────────
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F


# Model creation — enable auxiliary output
if MODEL_ARCH == "UnetPlusPlus":
    DEEP_SUPERVISION = True  # Toggle deep supervision

    model = smp.UnetPlusPlus(
        encoder_name=BACKBONE,
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
        # SMP UNet++ aux_params adds a classification head, not per-node supervision.
        # For true deep supervision we wrap the model.
    )

    if DEEP_SUPERVISION:
        class UNetPPDeepSupervision(nn.Module):
            """
            Wraps SMP UNet++ to add auxiliary heads on intermediate decoder nodes.
            During training: returns (main_output, [aux1, aux2, ...]).
            During eval: returns main_output only.
            """
            def __init__(self, base_model, num_classes):
                super().__init__()
                self.base = base_model
                self.encoder = base_model.encoder
                self.decoder = base_model.decoder
                self.seg_head = base_model.segmentation_head

                # Add auxiliary heads for intermediate decoder stages
                # SMP UNet++ decoder blocks output to .blocks attribute
                decoder_channels = base_model.decoder.blocks  # nested ModuleList
                self.aux_heads = nn.ModuleList()
                for level_blocks in decoder_channels:
                    # Each level's last block output channels
                    last_block = level_blocks[-1]
                    # Get output channels from the block's convolutions
                    for m in reversed(list(last_block.modules())):
                        if isinstance(m, nn.Conv2d):
                            ch = m.out_channels
                            break
                    self.aux_heads.append(
                        nn.Conv2d(ch, num_classes, kernel_size=1)
                    )

            def forward(self, x):
                features = self.encoder(x)
                decoder_output = self.decoder(*features)

                # Main output
                main = self.seg_head(decoder_output)

                if self.training:
                    # Auxiliary outputs from intermediate nodes
                    aux_outputs = []
                    for i, level_blocks in enumerate(self.decoder.blocks):
                        # Get output from the last block in each level
                        node_feat = level_blocks[-1].output  # May need hook
                        aux = self.aux_heads[i](node_feat)
                        aux = F.interpolate(aux, size=main.shape[2:],
                                            mode='bilinear', align_corners=True)
                        aux_outputs.append(aux)
                    return main, aux_outputs
                return main

        # NOTE: The above is a reference implementation. The exact approach
        # depends on SMP internals. An alternative is to use forward hooks:

        class UNetPPWithHooks(nn.Module):
            """
            Alternative: uses forward hooks to capture intermediate features
            without modifying SMP internals.
            """
            def __init__(self, base_model, num_classes):
                super().__init__()
                self.base = base_model
                self.hooks = []
                self.intermediate_features = []

                # Register hooks on decoder blocks
                for i, level_blocks in enumerate(base_model.decoder.blocks):
                    block = level_blocks[-1]  # Last block in each level
                    self.hooks.append(
                        block.register_forward_hook(self._make_hook(i))
                    )

                # Auxiliary classification heads
                self.aux_heads = nn.ModuleList()
                for level_blocks in base_model.decoder.blocks:
                    last_block = level_blocks[-1]
                    for m in reversed(list(last_block.modules())):
                        if isinstance(m, nn.Conv2d):
                            ch = m.out_channels
                            break
                    self.aux_heads.append(nn.Conv2d(ch, num_classes, 1))

            def _make_hook(self, idx):
                def hook(module, input, output):
                    if self.training:
                        self.intermediate_features.append(output)
                return hook

            def forward(self, x):
                self.intermediate_features = []
                main_out = self.base(x)

                if self.training and len(self.intermediate_features) > 0:
                    target_size = main_out.shape[2:]
                    aux_outputs = []
                    for feat, head in zip(self.intermediate_features, self.aux_heads):
                        aux = head(feat)
                        aux = F.interpolate(aux, size=target_size,
                                            mode='bilinear', align_corners=True)
                        aux_outputs.append(aux)
                    return main_out, aux_outputs

                return main_out

        model = UNetPPWithHooks(model, NUM_CLASSES)
        print("🧠 UNet++ with deep supervision via forward hooks")

    model = model.to(device)
```

**Deep supervision loss integration:**

```python
# ── Training loop modification for deep supervision ─────────────────
DEEP_SUP_WEIGHT = 0.4  # Total weight for all auxiliary losses

# Inside train_one_epoch(), replace the loss computation:
if MODEL_ARCH == "UnetPlusPlus" and DEEP_SUPERVISION:
    outputs = model(images)  # returns (main, [aux1, aux2, ...])
    if isinstance(outputs, tuple):
        main_out, aux_outputs = outputs
        # Main loss
        loss = criterion(main_out, masks)
        # Auxiliary losses (equally weighted)
        if len(aux_outputs) > 0:
            aux_loss = sum(criterion(aux, masks) for aux in aux_outputs)
            aux_loss = aux_loss / len(aux_outputs)
            loss = loss + DEEP_SUP_WEIGHT * aux_loss
        outputs_for_iou = main_out
    else:
        loss = criterion(outputs, masks)
        outputs_for_iou = outputs
else:
    outputs = model(images)
    loss = criterion(outputs, masks)
    outputs_for_iou = outputs
```

**New hyperparameters for §7:**

| Parameter | Current Value | Suggested Range | Notes |
|-----------|--------------|-----------------|-------|
| `DEEP_SUPERVISION` | `False` | `True`/`False` | Enable for UNet++ to improve gradient flow |
| `DEEP_SUP_WEIGHT` | `0.4` | `0.2`–`0.6` | Total weight given to auxiliary losses |

#### 13.3.2 CCE + Focal Tversky Loss for Minority Classes

**Problem:** DiceLoss alone does not penalize false negatives on rare classes strongly enough. Focal Tversky Loss generalizes Dice by letting you control the FN/FP trade-off and applying a focal exponent to down-weight easy examples.

**Goal:** Combine Categorical Cross-Entropy (for pixel stability) with Focal Tversky (for minority-class recall).

```python
# ── CCE + Focal Tversky combined loss ───────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for semantic segmentation.

    Tversky index: TI = TP / (TP + α*FP + β*FN)
    Focal Tversky:  FT = (1 - TI)^γ

    Higher β → more penalty on false negatives (improves recall).
    Higher γ → more focus on hard (poorly classified) classes.
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 gamma: float = 0.75, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha  # FP weight
        self.beta = beta    # FN weight (higher → recall-oriented)
        self.gamma = gamma  # Focal exponent
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N, C, H, W) raw model output.
            targets: (N, H, W) class indices.
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)  # (N, C, H, W)

        # One-hot encode targets
        targets_oh = F.one_hot(targets.long(), num_classes)  # (N, H, W, C)
        targets_oh = targets_oh.permute(0, 3, 1, 2).float()  # (N, C, H, W)

        # Per-class Tversky index
        tp = (probs * targets_oh).sum(dim=(0, 2, 3))
        fp = (probs * (1 - targets_oh)).sum(dim=(0, 2, 3))
        fn = ((1 - probs) * targets_oh).sum(dim=(0, 2, 3))

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = (1 - tversky) ** self.gamma

        return focal_tversky.mean()


class CCEFocalTverskyLoss(nn.Module):
    """
    Combined loss: λ_cce * CCE + λ_ft * FocalTversky
    """
    def __init__(self, num_classes: int, cce_weight: float = 0.5,
                 ft_weight: float = 0.5, alpha: float = 0.3,
                 beta: float = 0.7, gamma: float = 0.75,
                 class_weights: torch.Tensor = None):
        super().__init__()
        self.cce = nn.CrossEntropyLoss(
            weight=class_weights  # Optional per-class weighting
        )
        self.focal_tversky = FocalTverskyLoss(alpha, beta, gamma)
        self.cce_weight = cce_weight
        self.ft_weight = ft_weight

    def forward(self, logits, targets):
        return (self.cce_weight * self.cce(logits, targets) +
                self.ft_weight * self.focal_tversky(logits, targets))


# ── Usage (in Model Initialization cell) ────────────────────────────
if MODEL_ARCH == "UnetPlusPlus":
    # Optional: compute class weights from training data
    # class_weights = compute_class_weights(train_loader, NUM_CLASSES, device)
    criterion = CCEFocalTverskyLoss(
        num_classes=NUM_CLASSES,
        cce_weight=0.5,
        ft_weight=0.5,
        alpha=0.3,   # FP penalty
        beta=0.7,    # FN penalty (high → favor recall)
        gamma=0.75,  # Focal exponent
    )
    print("⚖️ Loss: CCE(0.5) + Focal Tversky(0.5) — minority-class aware")
```

**New hyperparameters for §7:**

| Parameter | Current Value | Suggested Range | Notes |
|-----------|--------------|-----------------|-------|
| `cce_weight` (UNet++) | `0.5` | `0.3`–`0.7` | Weight for CCE in combined loss |
| `ft_weight` (UNet++) | `0.5` | `0.3`–`0.7` | Weight for Focal Tversky |
| `alpha` (Tversky) | `0.3` | `0.2`–`0.5` | FP penalty; lower = less FP |
| `beta` (Tversky) | `0.7` | `0.5`–`0.8` | FN penalty; higher = better recall |
| `gamma` (Focal) | `0.75` | `0.5`–`2.0` | Exponent; higher = more focus on hard classes |

#### 13.3.3 Oversampling Tiles with Small Objects

**Problem:** In drone imagery, small structures (chimneys, dormers, small annexes) occupy few pixels. Random tile sampling under-represents these during training.

**Goal:** Implement a weighted sampler that over-samples tiles containing rare/small objects.

```python
# ── Small-object oversampling ───────────────────────────────────────
import numpy as np
from torch.utils.data import WeightedRandomSampler
from PIL import Image


def compute_sample_weights(dataset, rare_classes: list = None,
                           rare_boost: float = 3.0,
                           min_pixel_fraction: float = 0.001) -> list:
    """
    Compute per-sample weights for a WeightedRandomSampler.

    Samples containing rare-class pixels (above min_pixel_fraction)
    are boosted by rare_boost factor.

    Args:
        dataset: List of dicts with 'seg_map_path' keys.
        rare_classes: List of class indices considered rare.
                      If None, auto-detect from dataset (bottom 25% by pixel count).
        rare_boost: Weight multiplier for tiles with rare objects.
        min_pixel_fraction: Minimum fraction of tile pixels that must belong
                           to a rare class for the boost to apply.

    Returns:
        List of float weights, one per sample.
    """
    print("🔍 Computing sample weights for oversampling...")
    weights = []

    # Pass 1: count pixels per class across all samples
    if rare_classes is None:
        class_pixel_counts = {}
        for item in dataset:
            mask = np.array(Image.open(item['seg_map_path']))
            # If RGB mask, this counts unique RGB tuples; for index mask, unique values
            for val in np.unique(mask.reshape(-1) if mask.ndim == 2
                                 else mask.reshape(-1, 3), axis=0):
                key = int(val) if mask.ndim == 2 else tuple(val)
                class_pixel_counts[key] = class_pixel_counts.get(key, 0) + 1

        # Bottom 25% of classes by pixel count are "rare"
        sorted_classes = sorted(class_pixel_counts.items(), key=lambda x: x[1])
        cutoff = max(1, len(sorted_classes) // 4)
        rare_classes_set = {cls for cls, _ in sorted_classes[:cutoff]}
        print(f"   Auto-detected {len(rare_classes_set)} rare classes: {rare_classes_set}")
    else:
        rare_classes_set = set(rare_classes)

    # Pass 2: assign weights
    for item in dataset:
        mask = np.array(Image.open(item['seg_map_path']))
        total_pixels = mask.size if mask.ndim == 2 else mask.shape[0] * mask.shape[1]

        # Check if this sample contains enough rare-class pixels
        has_rare = False
        if mask.ndim == 2:
            for cls in rare_classes_set:
                if isinstance(cls, int):
                    frac = (mask == cls).sum() / total_pixels
                    if frac >= min_pixel_fraction:
                        has_rare = True
                        break

        weights.append(rare_boost if has_rare else 1.0)

    print(f"   Boosted {sum(1 for w in weights if w > 1.0)}/{len(weights)} samples")
    return weights


# ── Usage (in DataLoader creation) ──────────────────────────────────
USE_OVERSAMPLING = True  # Enable for UNet++ with small objects
RARE_BOOST = 3.0         # How much more often to sample rare-object tiles
RARE_CLASSES = None      # None = auto-detect; or list of class indices

if USE_OVERSAMPLING and MODEL_ARCH == "UnetPlusPlus":
    sample_weights = compute_sample_weights(
        train_dataset, rare_classes=RARE_CLASSES, rare_boost=RARE_BOOST
    )
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    # Replace train_loader (shuffle must be False when using a sampler)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    print(f"🎯 Oversampling enabled: rare objects boosted {RARE_BOOST}×")
```

**New hyperparameters for §7:**

| Parameter | Current Value | Suggested Range | Notes |
|-----------|--------------|-----------------|-------|
| `USE_OVERSAMPLING` | `False` | `True`/`False` | Enable for datasets with small/rare objects |
| `RARE_BOOST` | `3.0` | `2.0`–`5.0` | Oversample factor; too high causes overfitting on rare tiles |
| `RARE_CLASSES` | `None` (auto) | List of ints or `None` | Manually specify which classes to boost |

#### 13.3.4 Metrics Integration

No special integration needed — `SegmentationMetrics` (§12) works with any model that produces `(N, C, H, W)` logits. When deep supervision is active, use only the main output for evaluation:

```python
# In evaluate_model(), handle deep supervision output:
outputs = model(images)
if isinstance(outputs, tuple):
    outputs = outputs[0]  # Use main output only
preds = torch.argmax(outputs, dim=1)
```

---

### 13.4 Summary: Architecture → Refinement Matrix

| Refinement | DeepLabV3+ | PSPNet | UNet++ | Shared |
|-----------|:----------:|:------:|:------:|:------:|
| Custom ASPP dilation rates | ✅ | — | — | — |
| Configurable LR scheduler (Plateau / Cosine) | ✅ | ✅ | ✅ | ✅ |
| Boundary-aware loss (Dice + Boundary IoU) | ✅ | — | — | — |
| Multi-scale inference | — | ✅ | — | — |
| Mosaic tiling with Gaussian blending | — | ✅ | — | Optional |
| Adaptive PPM kernel sizes | — | ✅ | — | — |
| Deep supervision | — | — | ✅ | — |
| CCE + Focal Tversky loss | — | — | ✅ | Optional |
| Small-object oversampling | — | — | ✅ | Optional |
| Metrics export (mIoU, OA, mAcc, mF1) | ✅ | ✅ | ✅ | ✅ |
| Training curves + Wilk comparison | ✅ | ✅ | ✅ | ✅ |

### 13.5 Implementation Order

Recommended order for implementing these refinements:

| Phase | Task | Branch Name | Depends On |
|-------|------|-------------|------------|
| 1 | Configurable LR scheduler (shared) | `feat/lr-scheduler-options` | — |
| 2 | DeepLabV3+ ASPP dilation tuning | `feat/deeplabv3-aspp-tuning` | — |
| 3 | DeepLabV3+ boundary loss | `feat/boundary-loss` | Phase 1 |
| 4 | PSPNet adaptive PPM | `feat/pspnet-adaptive-ppm` | — |
| 5 | PSPNet multi-scale inference | `feat/pspnet-multiscale` | — |
| 6 | PSPNet mosaic tiling | `feat/mosaic-tiling` | Phase 5 |
| 7 | UNet++ deep supervision | `feat/unetpp-deep-supervision` | — |
| 8 | UNet++ CCE + Focal Tversky loss | `feat/cce-focal-tversky` | Phase 7 |
| 9 | UNet++ small-object oversampling | `feat/rare-class-oversampling` | Phase 8 |
