# INSTRUCTIONS — Deep Context for Future Improvements

This document captures the architectural decisions, known limitations, and concrete directions for improving the `sam-image-processing` project. It is written for contributors (human or AI) who will iterate on the codebase.

> **Major restructuring (2026-11-14, branch `feat/segpipe-ablation`).** Training logic moved out of the notebook into the tested `segpipe/` package; the old cloud bucket (no longer exists) was replaced by a verified Google Drive archive; the random 80/20 split was replaced by a spatial block split; `background` is now ignored instead of trained. Sections 12–14 below are **historical plans written for the original notebook** (removed from the tree; available in git history) and are kept for context; each one has a banner with its current status.

---

## 1. Current State Summary

### What exists today

| Artifact | Purpose | Maturity |
|----------|---------|----------|
| `segpipe/` | Class-schema-configurable segmentation pipeline: dataset fetch/verify/validate, spatial block split, training with per-epoch checkpoint + resume, native-resolution metrics with photo-bootstrap CIs, comparison report | Tested on CPU with `pytest` (synthetic data + real pack for data/split); **not yet run on GPU** |
| `configs/datasets/pascual.yaml` | Dataset config (Drive file ID, sha256, sizes, paths, ignored classes, geometry columns) | Verified against the real `segpack_v1.zip` |
| `configs/runs/*.yaml` | Ablation steps J0–J3, M-b3 (+ optional J2-loss, PSPNet, UNetPP) with `base:` inheritance | — |
| `notebooks/train_ablation.ipynb` | Thin Colab orchestration: setup, study, smoke run, queue, comparison table | Syntax-checked; not executed on Colab yet |
| `notebooks/automatic_mask_generator_example.ipynb` | SAM 2 zero-shot masks → COCO JSON for CVAT; images come from the segpack (`ensure_dataset`), results to local disk or mounted Drive | I/O cells rewritten (SAM logic untouched); not re-executed |
| `tools/pack_dataset.py` | Builds the segpack from a CVAT "Segmentation mask 1.1" export | Used to build `segpack_v1.zip` |
| `tests/` | pytest suite (CPU) | See README "Pruebas locales" |

### What does NOT exist yet
- No GPU run of the new pipeline (timings, memory, AMP behaviour unverified).
- No re-mapping of predictions of a fine-schema model into a coarse common schema at evaluation time (the fine→coarse mapping is applied to labels before training).
- No CI; tests are run manually (`pytest`).
- No inference/export script for new, unlabeled photos.
- Old checkpoints (`2025/training/*.pth`) were trained with 14 classes due to the labelmap-header bug and are **incompatible**; do not load them.

---

## 2. Architecture Decisions & Rationale

### 2.1 Package + thin notebook
Colab is the only GPU runtime (no local GPU). Logic lives in `segpipe/` so it can be unit-tested on CPU; the notebook only clones the repo, sets parameters and calls `prepare_study` / `run_queue` / `report.build`.

### 2.2 Dataset storage: verified Drive archive
> **Changed** (2026-11-14). The original cloud bucket no longer exists and images cannot be re-downloaded from CVAT. The dataset is a single archive (`segpack_v1.zip`) in Google Drive. `segpipe.data_io.ensure_dataset` fetches it (validated local copy → Drive API with the user's account → mounted Drive → `SEGPACK_DIR`/`SEGPACK_ZIP`), checks size + sha256, extracts to the VM's local disk and validates against `manifest.csv`. A Colab kernel cannot see the user's local disk/OneDrive, so nothing depends on them. No secrets or connection strings are stored anywhere.

### 2.3 Loss
J0 keeps the original loss (Dice + inverse-frequency weighted CE). Other losses (`ce`, `focal`, `dice_focal`, `sqrt_inverse_freq` weights) are config options. All losses use `ignore_index=255` for unannotated pixels.

### 2.4 ~~Configurable random split~~ → Spatial block split
> **Replaced** (2026-11-14). `train_test_split(0.8, seed 42)` leaked information: neighbouring oblique photos see the same ground. `segpipe.split` projects the 4 image corners to a flat ground plane using the Metashape calibration (f 6.72 mm, pixel 0.0026188 mm, 4032×2268 → FOV 76.3° × 47.7°; `geometry.camera` in the dataset YAML) and the gimbal yaw/pitch/roll. The photo centre is the polygon centroid (≈ 96 m ahead of the drone). Centres go into 150 m grid blocks; the flight date plays no role (9 of 19 blocks mix both flights). There are k = 5 block folds (test = f, val = f+1) and a 50 m centre buffer that excludes train photos near val/test. Each fold reports the min train↔test distance and the **residual footprint overlap** (area of test footprints covered by train footprints, 2 m raster). Call this "reduced leakage" ("fuga reducida"), never "no leakage". Rationale and numbers: README "Protocolo de evaluación".

### 2.5 Class schema from `classes.json`
`classes.json` (ordered `index, name, rgb`) is the single source of truth; `labelmap.txt` is no longer parsed. RGB→index is vectorized and **fails on unknown colours**. `background` (black, no CVAT polygon) is ignored in loss, metrics and confusion matrix in every step, including J0 — a correction vs. the original notebook, which trained it as a class.

### 2.6 Metrics
Per-photo confusion matrices at native resolution (4032×2268); mIoU/OA/mAcc/mF1 + per class; classes present in < K photos (default 5) of the evaluated partition are "not evaluable" and excluded from the means (listed explicitly); 95 % CIs by bootstrapping photos. With block CV the test predictions of all folds are pooled (each photo tested once).

### 2.7 Reproducibility and resumption
Per-sample randomness derives from (seed, epoch, index); `last.pt` stores model/optimizer/scheduler/scaler/epoch/best/patience/history/RNG and is written atomically every epoch; re-running the same cell resumes. A test asserts that an interrupted + resumed run produces the same log, weights and metrics as an uninterrupted one. Each run records config, split, library versions and git commit.

---

## 3. Known Limitations & Technical Debt

Items 3.1–3.8 describe the original training notebook (removed from the tree; see git history, e.g. `git show 9c3f171:notebooks/semantic_segmentation_training.ipynb`); their status in `segpipe` is noted.

- **3.1 RGB mask conversion speed** — ✅ vectorized (`ClassSchema.rgb_to_index`).
- **3.2 Binary IoU** — ✅ per-class from confusion matrices (`segpipe.metrics`).
- **3.3 Early stopping** — ✅ `train.patience`.
- **3.4 Fragile labelmap parsing** — ✅ RESOLVED (2026-11-14): `classes.json` is used instead; the original labelmap-header bug produced 14 classes.
- **3.5 No data validation** — ✅ RESOLVED (2026-11-14): archive sha256/size, pair completeness, image/mask sizes vs. manifest, unknown mask colours.
- **3.6 LR plot** — ✅ `lr` logged per epoch in `epoch_log.csv`.
- **3.7 Standard metrics** — ✅ mIoU/OA/mAcc/mF1 + bootstrap CIs.
- **3.8 Recipe under-optimized** — now studied as the J0–J3 ablation. Original-notebook Run 1–5 numbers below were measured with the leaky random split, 14 classes and `background` as a class; **they are not comparable** with new results.

New limitations:
- **3.9 Residual footprint overlap (reduced, not removed leakage).** Footprints are ~150 m deep and 90–240 m wide, so a centre buffer cannot remove overlap without emptying train. With the real pack, train footprints cover 38–60 % of test footprint area at 50 m and 32–46 % at 75 m. Report `min_dist_train_test_m` and `overlap_train_test_pct` per fold; J0-b75 measures sensitivity.
- **3.10 Small dataset.** 66 photos → train 33–39 per fold at 50 m (23–32 at 75 m). The default is one seed (42); add seeds with `EXTRA_SEEDS` if the budget allows. Report photo-bootstrap CIs.
- **3.11 M-b3 is approximate.** Several hyperparameters are inferred; see `docs/mauricio_config.md`.
- **3.12 J1 confound (resolution + number of steps).** J0 epochs are one pass over ~35 resized photos (~4 optimizer steps); J1+ epochs are 880 native crops (110 steps). The control **J0-pasos** (J0 with 880 samples/epoch) has the same step budget as J1, so J0-pasos → J1 isolates resolution. Every run logs cumulative optimizer steps (`opt_steps`), and the final table reports total steps and steps at the best checkpoint.
- **3.13 Flat-ground footprint.** Footprints assume flat ground at `RelativeAltitude`, a centred principal point and no distortion; buildings and terrain relief shift the true footprint.

<details><summary>Original-notebook run log (historical, not comparable)</summary>

| Run | Model | Loss | Best val mIoU | Notes |
|-----|-------|------|---------------|-------|
| 1 | PSPNet R50 | Dice | 0.3354 | 50/50 epochs |
| 2 | PSPNet R50 | Dice | 0.3609 | early stopped at 67 |
| 3 | DeepLabv3+ R50 | Dice | 0.3738 | OA 70.9 % |
| 4 | DeepLabv3+ R50 | Dice + CE (unweighted), strong aug | 0.3800 | Fence/Lightning Posts collapsed to 0 |
| 5 | DeepLabv3+ R50 | Dice + weighted CE, CoarseDropout 16 px | — | = J0 recipe |

</details>

---

## 4. Recommended Future Improvements

| # | Improvement | Notes |
|---|-------------|-------|
| 1 | Run the smoke run on Colab, review measured epoch times and the cost estimate, then the queue J0 → J3 → J1 → J2 → M-b3 → J0-pasos → J0-b75 (`docs/smoke_run.md`) | First GPU validation; adjust `train.micro_batch` if the heuristic over/under-fills memory |
| 2 | **After the smoke run:** evaluate a 12-class model in the common coarse schema by aggregating its confusion matrix fine→coarse | Needed to compare with Mauricio's 10-class model; per-photo matrices (`photo_confusion.npz`) are already saved, no retraining needed |
| 3 | Add Mauricio's dataset (`configs/datasets/mauricio.yaml`) | Needs its segpack (same format) and an agreed common schema |
| 4 | Inference script for unlabeled photos (+ GeoTIFF/point-cloud projection) | — |
| 5 | CI running `pytest` on push | — |
| 6 | §13 refinements (ASPP dilations, boundary loss, multi-scale inference, UNet++ deep supervision) | Add as new configs with `base:` so they stay single-change ablations |

---

## 5. How to Add a New Model Architecture

1. Register it in `segpipe/models.py::build_model` (and its input multiple in `input_multiple`).
2. Create `configs/runs/<Name>.yaml` with `base: J2.yaml` and a `model:` block.
3. Add a forward/backward test in `tests/test_train.py`.

Registered today: `deeplabv3plus`, `pspnet`, `unetplusplus` (smp, any smp encoder) and `segformer` (transformers, `hf_name`).

---

## 6. How to Add a New Dataset

See README "Añadir otro dataset". In short: build a segpack with `tools/pack_dataset.py`, upload it to Drive, copy `configs/datasets/pascual.yaml`, set `archive.*`, `local.extract_dir`, `classes.ignore` and (optionally) `classes.target` + `classes.mapping` for a fine→coarse common schema, and use a new `STUDY_DIR`. The manifest must contain the GPS/gimbal columns named in `geometry:` for the spatial split.

---

## 7. Hyperparameter Reference (run YAML keys)

| Key | J0 default | Notes |
|-----|-----------|-------|
| `model.arch` / `encoder` / `hf_name` | `deeplabv3plus` / `resnet50` / — | `segformer` uses `hf_name` (`nvidia/mit-b3`) |
| `data.mode` | `resize` | `crops` = native-resolution random crops |
| `data.resize` / `crop` | 512 / 1024 | |
| `data.samples_per_epoch` | — | samples per epoch (880 in J1+ and J0-pasos); in `resize` mode photos repeat in a cycle |
| `data.scale_range` | — | random rescale before cropping (M-b3: [0.5, 1.0]) |
| `data.sampler`, `rare_p`, `rare_classes`, `rare_px_frac` | `uniform`, 0.5, `auto`, 0.01 | `rare` = crops centred on rare-class pixels |
| `data.augment` | `legacy` | `hflip`, `none` |
| `infer.mode`, `window`, `overlap`, `scale` | `resize`, 1024, 0.5, 1.0 | `sliding` for J1+ |
| `loss.name`, `weights` | `dice_wce`, `inverse_freq` | `ce`, `focal`, `dice_focal`; `none`, `sqrt_inverse_freq` |
| `optim.name`, `lr`, `weight_decay` | `adam`, 1e-4, 1e-4 | `adamw` |
| `sched.name` | `plateau` (×0.5, patience 5, on val loss) | `poly` (per iteration), `none` |
| `train.batch` / `micro_batch` | 8 / auto | effective batch kept with gradient accumulation |
| `train.epochs`, `patience`, `select` | 50, 10, `miou` | `maxmin` = maximize min per-class val IoU |
| `train.amp` | true | bf16 if supported, else fp16 + GradScaler |
| `eval.min_photos`, `n_boot` | 5, 2000 | "not evaluable" threshold K; bootstrap replicates |
| `split.buffer_m` | — (study's 50 m) | per-config buffer with the same block assignment (J0-b75: 75) |
| `reference` | — | config used only by `describe_ladder` for the diff (M-b3 → J3) |

Study-level parameters (`prepare_study`): `protocol` (`cv`/`fixed`), `k` = 5, `block_m` = 150, `buffer_m` = 50, `split_seed` = 0. Dataset-level geometry (`geometry.camera`: `focal_mm`, `pixel_mm`, `width`, `height`, `max_range_m`; `flight_regex` for diagnostics).

Default queue (`runner.DEFAULT_QUEUE`, run config → seed → fold): J0, J3, J1, J2, M-b3, J0-pasos, J0-b75 with seed 42 → 7 × 1 × 5 = **35 runs** (+ 5 short smoke runs in a separate study).

---

## 8. Glossary of Key Concepts

| Term | Definition |
|------|-----------|
| **Semantic segmentation** | Classifying every pixel in an image into a predefined category |
| **Footprint / centre** | Ground polygon of an oblique photo (4 image corners projected to a flat plane with the calibrated camera and gimbal angles); centre = polygon centroid |
| **Residual overlap ("fuga reducida")** | % of the area of held-out footprints covered by train footprints |
| **Block CV** | Cross-validation where whole spatial blocks go to the same fold, to limit spatial leakage |
| **Buffer** | Minimum footprint-centre distance between a train photo and any val/test photo |
| **Not evaluable** | Class present in fewer than K photos of the evaluated partition; excluded from mean metrics |
| **Photo bootstrap** | Resampling photos (not pixels) with replacement to obtain confidence intervals |
| **Sliding-window inference** | Predicting overlapping native-resolution tiles and averaging logits |
| **SAM 2** | Meta's Segment Anything Model v2 — a foundation model for zero-shot segmentation |
| **DeepLabv3+** | Encoder-decoder with atrous convolutions and ASPP |
| **SegFormer (mit-b3)** | Transformer encoder (Mix Transformer) with a lightweight MLP decoder |
| **PSPNet / UNet++** | Pyramid pooling network / nested U-Net (registered options) |
| **SMP** | `segmentation-models-pytorch` |
| **AMP** | Automatic mixed precision (bf16/fp16) |
| **ReduceLROnPlateau / poly** | LR schedules: halve on plateau / polynomial decay per iteration |

---

## 9. File-Level Reference

| File | Key contents |
|------|-------------|
| `segpipe/data_io.py` | `ensure_dataset`, `download_from_drive`, `download_ranges`, `verify_archive`, `safe_extract`, `validate_extracted` |
| `segpipe/classes.py` | `ClassSchema.from_json`, `rgb_to_index`, `IGNORE_INDEX = 255` |
| `segpipe/split.py` | `camera_fov`, `footprint_polygons`, `footprints`, `FootprintRaster`, `assign_blocks`, `make_folds`, `folds_from_assignment`, `flight_mix`, `save_folds`, `split_report` |
| `segpipe/data.py` | `PhotoStore`, `build_augment` (legacy aug, albumentations 1.x/2.x), `ResizeDataset`, `CropDataset` |
| `segpipe/models.py` | `build_model`, `SegFormer` wrapper |
| `segpipe/losses.py` | `class_weights`, `build_loss` |
| `segpipe/infer.py` | `predict_logits` (resize / sliding window, optional scale) |
| `segpipe/metrics.py` | `confusion`, `per_class`, `summarize`, `bootstrap`, `evaluate`, `save_report` |
| `segpipe/train.py` | `train_run` (checkpoint/resume, final evaluation), `evaluate_photos` |
| `segpipe/runner.py` | `prepare_study`, `Study.folds_for` (buffer variants), `write_ignored_report`, `run_queue`, `run_smoke`, `smoke_epoch_times`, `estimate_cost`, `describe_ladder` |
| `segpipe/report.py` | `build`, `default_pairs`, `to_markdown` |
| `segpipe/config.py` | `load_run_config` (`base:` inheritance), `diff` |
| `segpipe/env.py` | `gpu_info`, `micro_batch`, `num_workers`, `versions`, `git_commit` |
| `notebooks/train_ablation.ipynb` | Sections: 1 code/deps, 2 parameters, 3 dataset+split, 4 smoke run + epoch times + cost estimate, 5 queue, 6 comparison, 7 curves |
| `notebooks/automatic_mask_generator_example.ipynb` | SAM 2 setup, image source (`ensure_dataset`), mask generation, COCO batch export |
| `docs/smoke_run.md`, `docs/mauricio_config.md` | Smoke-run procedure; M-b3 provenance |

---

## 10. Conventions for Contributors

1. **Notebook cells must be idempotent and resumable** — re-running a cell continues where it left off (finished runs are skipped).
2. **Print emoji prefixes** — use the established convention: ✅ success, ❌ error, ⚠️ warning, 🔍 informational, 📊 metrics, 💾 save, 🎯 key outputs.
3. **Keep notebooks thin** — notebooks only contain imports, configuration, and visualization. All logic lives in `segpipe/` and is covered by `pytest`.
4. **No secrets, no scattered paths** — dataset IDs, hashes and paths live in `configs/datasets/*.yaml`; Drive access uses the user's own Colab authentication. Never read training images from Drive (extract to the VM's local disk).
5. **Runs are self-describing** — each run directory stores the resolved config, split, class names, library versions and git commit next to the weights.

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

1. **Test before committing.** Run `pytest` locally (CPU). For changes that affect GPU behaviour, run the smoke run on Colab (`docs/smoke_run.md`).
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
| **New dependency** | Add it to `requirements.txt` (minimum version only) and to the install-if-missing cell of `notebooks/train_ablation.ipynb`. Never pin torch to a CUDA build. |
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

> **Historical (original notebook, see git history).** Implemented in `segpipe/metrics.py` (confusion matrix → mIoU/OA/mAcc/mF1, per class, CSV/JSON) with photo-bootstrap CIs and "not evaluable" classes. Cell numbers below refer to the original notebook in git history (commit `9c3f171`).

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

> **Historical (original notebook, see git history), not implemented.** To try any of these, add a config with `base: J2.yaml` (or the relevant step) so it remains a single-change ablation. Sliding-window inference (§13.2.2-like, without Gaussian weighting) and UNet++ registration exist in `segpipe`.

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

---

## 14. Training Recipe Optimization — Implementation Plan

> **Historical (original notebook, see git history).** Changes A–C are part of the J0 recipe (`configs/runs/J0.yaml`). Expected results quoted here were measured with the leaky random split and are not comparable.

> **Status:** Planned (2026-03-04).
> **Goal:** Raise per-class mIoU from ~0.36 to ~0.42–0.50 on 14-class drone/roof segmentation.
> **Prerequisite:** All §4.4 Completed items are already in the notebook.

This section specifies three changes to the training recipe. All three should be applied **together** in a single iteration — they are complementary and their effects compound.

### 14.1 Change A: Combined Loss (DiceLoss + CrossEntropyLoss)

#### What to change

**Cell 20** (Model Initialization — the cell containing `criterion = smp.losses.DiceLoss(...)`).

#### Why

DiceLoss optimizes global overlap but does not penalize individual pixel misclassifications. When a pixel of class "flat roof" is predicted as "gable roof," DiceLoss only sees a slight overlap decrease. CrossEntropyLoss computes a per-pixel log-likelihood loss that directly penalizes this confusion. Together they provide both macro-level overlap optimization and micro-level class discrimination.

#### Exact code change

Replace:
```python
# Loss function - DiceLoss works well for segmentation
criterion = smp.losses.DiceLoss(mode='multiclass')
```

With:
```python
# Loss function — Combined DiceLoss + CrossEntropyLoss
# DiceLoss handles class imbalance (optimizes overlap metric)
# CrossEntropyLoss provides per-pixel gradients (discriminates similar classes)
# See INSTRUCTIONS.md §2.3, §3.8, §14.1 for rationale.
dice_loss = smp.losses.DiceLoss(mode='multiclass')
ce_loss = nn.CrossEntropyLoss()

def combined_loss(pred, target):
    """Combined Dice + CE loss for multi-class segmentation."""
    return dice_loss(pred, target) + ce_loss(pred, target)

criterion = combined_loss
```

#### What NOT to change in this cell

- `optimizer` — keep `Adam(lr=1e-4, weight_decay=1e-4)`.
- `scheduler` — keep `ReduceLROnPlateau(mode='min', factor=0.5, patience=5)`. It monitors `val_loss`; the combined loss is still a scalar that decreases when the model improves.
- Print statements — update the loss description line to read `"DiceLoss + CrossEntropyLoss (combined)"`.

#### Interaction with other cells

- **Training loop (cell 24):** No change needed. The loop calls `criterion(outputs, masks)` — this works identically whether `criterion` is a function, a `nn.Module`, or a lambda.
- **Checkpoint format:** No change. The checkpoint stores `model_state_dict` and `optimizer_state_dict`, not the loss function.
- **Post-training evaluation (cell 35):** No change. Evaluation uses `SegmentationMetrics`, not the loss function.

---

### 14.2 Change B: Stronger Data Augmentation

#### What to change

**Cell 18** (Data Transforms and Dataset — the cell containing `train_transforms = A.Compose([...])`).

#### Why

With ~65 training images, the model overfits quickly. The current augmentations (flip, rotate90, mild brightness/contrast, mild noise) are conservative. Drone imagery benefits from:
- **ShiftScaleRotate** — simulates camera angle variations and zoom.
- **HueSaturationValue** — handles lighting/color temperature variability across flight sessions.
- **OneOf** blocks — apply exactly one augmentation from a set, adding diversity without stacking too many distortions.
- **CoarseDropout** — simulates occlusions and forces the model to use context.

#### Exact code change

Replace the `train_transforms` definition:
```python
train_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.RandomGamma(p=0.3),
    A.GaussNoise(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

With:
```python
train_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
        A.RandomGamma(gamma_limit=(70, 130), p=1.0),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
    ], p=0.5),
    A.OneOf([
        A.GaussNoise(var_limit=(10, 50), p=1.0),
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
    ], p=0.3),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

#### What NOT to change in this cell

- `val_transforms` — validation transforms must stay unchanged (Resize + Normalize + ToTensorV2 only). Augmenting validation data would invalidate metrics.
- `RGBSegmentationDataset` class — no changes. Albumentations handles masks automatically via the `mask=mask` kwarg.
- `DataLoader` creation — no changes to batch size, num_workers, or pin_memory.
- Data loading test block — no changes.

#### Interaction with other cells

- **albumentations API:** All transforms used (`ShiftScaleRotate`, `OneOf`, `HueSaturationValue`, `GaussianBlur`, `CoarseDropout`) are available in `albumentations>=1.3.0` which is already installed.
- **Mask consistency:** All spatial transforms (flip, rotate, shift/scale) are automatically applied to both `image` and `mask` by `A.Compose`. Color transforms only affect the image (they ignore the mask channel). `CoarseDropout` drops rectangular regions from images only by default — masks are unaffected unless `mask_fill_value` is set.

---

### 14.3 Change C: Default Architecture → DeepLabv3+

#### What to change

**Cell 13** (Model Configuration — the line `MODEL_ARCH = "PSPNet"`).

#### Why

DeepLabv3+'s ASPP (Atrous Spatial Pyramid Pooling) module applies parallel dilated convolutions at multiple rates, capturing both local detail and global context. For fine-grained multi-class segmentation (14 roof types with varying sizes), this typically outperforms PSPNet's fixed-size pyramid pooling. The SMP library handles all architecture differences internally.

#### Exact code change

Replace:
```python
MODEL_ARCH = "PSPNet"  # Options: "DeepLabv3+", "PSPNet"
```

With:
```python
MODEL_ARCH = "DeepLabv3+"  # Options: "DeepLabv3+", "PSPNet"
```

#### What NOT to change

- **Everything else in the cell** — `BACKBONE`, `BATCH_SIZE`, `LEARNING_RATE`, `NUM_EPOCHS`, `IMAGE_SIZE`, `NUM_WORKERS`, device config, and output directories must remain unchanged.
- **The `if/elif` branching logic** — it already handles `"DeepLabv3+"` correctly and maps to `smp.DeepLabV3Plus`.

#### Interaction with other cells

- **Model Initialization (cell 20):** The `model_class(encoder_name=..., ...)` call is architecture-agnostic. No change needed.
- **Forward pass test (cell 22):** Works for any SMP model. No change needed.
- **Training loop (cell 24):** Architecture-agnostic. No change needed.
- **Checkpoint naming:** Uses `MODEL_ARCH` in the filename (e.g., `best_DeepLabv3+_resnet50_epoch_X.pth`), which automatically updates.
- **Post-training evaluation (cell 35):** Architecture-agnostic. No change needed.
- **Wilk comparison chart:** Uses `MODEL_ARCH` in the legend label automatically.

---

### 14.4 Safety Constraints — What Must NOT Break

Any contributor implementing §14.1–14.3 **must verify** the following invariants hold after their changes:

| Invariant | How to verify |
|-----------|---------------|
| `calculate_miou(predicted, target, num_classes)` returns a float in [0, 1] | Training loop prints mIoU per epoch; should not be NaN or >1 |
| `SegmentationMetrics.compute()` returns dict with keys `mIoU`, `OA`, `mAcc`, `mF1`, `per_class`, `confusion_matrix` | Post-training evaluation cell (35) prints all metrics |
| `learning_rates` list has exactly `len(train_losses)` entries | Visualization cell (26) plots LR curve without IndexError |
| Early stopping triggers when `epochs_without_improvement >= EARLY_STOPPING_PATIENCE` | Check training output shows `🛑 Early stopping triggered` when plateau is long enough |
| Checkpoint loads with `torch.load(path, map_location=device, weights_only=False)` | Post-training evaluation cell (35) loads without error |
| `train_test_split(pairs, train_size=TRAIN_RATIO, random_state=RANDOM_SEED)` produces the same split on rerun | First 3 filenames in `train_pairs` should be identical across runs with same seed |
| Mask conversion uses vectorized LUT path (not Python loops) | Data loading test in cell 18 completes in <5 seconds |
| CSV exports (`metrics_per_class.csv`, `metrics_summary.csv`) contain valid data | Check files exist in `logs/` and contain numeric values |

### 14.5 Expected Results

| Config | mIoU (estimated) | Notes |
|--------|-------------------|-------|
| Current (PSPNet, DiceLoss, moderate aug) | 0.36 | Baseline measured 2026-03-04 |
| +Combined loss only | 0.39–0.44 | Biggest single lever |
| +Combined loss + strong aug | 0.41–0.47 | Better generalization on small dataset |
| +Combined loss + strong aug + DeepLabv3+ | **0.42–0.50** | Full recipe; target for this iteration |
| Wilk et al. (2022) image baseline | 0.603 | Long-term target; may require more data or advanced techniques from §13 |

### 14.6 Post-Implementation Checklist

After applying all three changes and running a full training cycle on Colab:

- [ ] Training completes without errors (all cells run)
- [ ] Best mIoU > 0.40 (improved over 0.36 baseline)
- [ ] Post-training evaluation cell produces metrics CSV and comparison charts
- [ ] Confusion matrix shows improved diagonal density
- [ ] LR plot shows actual scheduler steps (not a flat line)
- [ ] Early stopping triggers at a reasonable epoch (not too early, not at max)
- [ ] Update §3.8 in INSTRUCTIONS.md with new baseline results
- [ ] Move §4.1 items A/B/C to §4.4 Completed table
- [ ] Update §7 hyperparameter table with actual values used
- [ ] Update §9 cell references if any cells were added or reordered

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
