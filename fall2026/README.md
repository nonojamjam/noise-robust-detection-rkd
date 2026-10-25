# Fall 2026 follow-up: heterogeneous spatial KD (DINOv2 → YOLOv8s-seg) on the official iSAID split

This branch holds the work done **after** the June 2026 capstone presentation. The `main` branch keeps the June presentation version (with its Known issues). Numbers here are **not comparable** to the June numbers. Everything in this branch is work in progress.

## Why this branch exists
An audit after June found two problems that affect the June numbers (see `main` → Known issues 3 and 4):
- **Train/val overlap.** In the Kaggle conversion we used, 7,926 of 8,919 validation tiles are byte-identical to training tiles.
- **Label prefix.** Every label line starts with `cls cx cy w h` before the polygon. The loader read those four values as extra vertices, which inflated the ground-truth boxes.

The 10-epoch runs in the results table use the **official iSAID train/val scene split**: 1,411 train scenes / 458 val scenes, 18,732 / 6,048 tiles, 0 shared scenes. The lists are in `data_split/` (sha256 `246efa0a…` / `be763a31…`). Labels have the prefix removed, and evaluation uses `last.pt`.

## Method (what changed from June)
- **Teacher.** Frozen DINOv2 ViT-B/14. Input is 448 px with ImageNet normalization, and the teacher uses **patch tokens (32×32)** instead of the CLS token.
- **Student.** YOLOv8s-seg. KD is applied at neck P4 (`model[18]`, 256×40×40) through a training-only adapter (Conv3×3 → GN → GELU → Conv1×1 → resize to 32×32). The adapter is removed at inference.
- **Loss.** `L = L_det + λ·L_sp`, where `L_sp` is a per-position cosine distance after a fixed LayerNorm, averaged over positions except the 1 % highest-norm teacher tokens per image. The teacher always sees the clean image; half of each batch gets Gaussian noise σ = 50/255.
- **Conditions.**

| Condition | Description |
|---|---|
| S0 | Same path, λ = 0 (no KD) |
| S1 | Spatial KD, plain loss, λ = 0.5 |
| S2 | S1 with teacher tokens randomly shuffled across positions (control: is spatial correspondence what matters?) |
| X1 | Exploratory: S1 with a **centered** loss (per-image mean removed before cosine) and λ = 2.5 |
| X2 | X1 with shuffled teacher tokens (pending) |

  X1 changes both the loss form and λ at once, so an effect cannot be attributed to either one alone.
- **Code.** `code/seed_sweep_v2.py`. Each condition is selected by the settings the notebook generators write (`EXP`, `LAM_SP_SHARED`, `KD_LOSS_VARIANT`); the script's own defaults are not one of these conditions. The other loss variants and training options in that file are opt-in and exploratory.

## Results — official split, 10 epochs, **seed 0 only** (box mAP50, noisy = Gaussian σ = 0.3)

| Condition | clean | noisy | noisy − S0 |
|---|---|---|---|
| S0 (no KD) | 0.5974 | 0.3827 | — |
| S1 (plain, λ 0.5) | 0.5971 | 0.3749 | −0.0078 |
| S2 (shuffled) | 0.5908 | 0.3868 | +0.0041 |
| X1 (centered, λ 2.5) | 0.6114 | 0.3995 | +0.0168 |

The raw result files are in `results/`. These runs used `seed_sweep_v2.py` sha256 `8465ec15…`; the copy in `code/` (`64b55605…`) adds opt-in loss variants and training options whose defaults are meant to leave these conditions unchanged (covered by the unit tests, not by re-running). **This is a single seed.** We have not yet measured seed-to-seed spread at this scale. In 20 %-data, 1-epoch screening runs, noisy mAP varied by 0.023 (S0) to 0.045 (centered) across three seeds, and the sign of centered − S0 flipped (+0.016, +0.025, −0.018). X2 (the shuffle control for X1) has not run yet, so we make no claim that spatial correspondence explains X1. Note also that the shuffled control S2 scored above both S0 and S1 on noisy mAP in this seed; with one seed we cannot tell whether that is noise or a real effect.

## Diagnostics
- **KD signal was small in the plain loss.** The ratio `λ·L_sp / (per-image detection loss)` was about 0.02 for S1 (plain, λ = 0.5; 10-epoch log) — and about 0.04 for the centered loss at λ = 0.5 in 1-epoch diagnostic runs. X1 used λ = 2.5, which put the ratio at about 0.18 (10-epoch log).
- **Alignment did not coincide with robustness in this run.** In the 10-epoch S1 run, the adapter output retrieves the matching teacher token at 0.055 P@1 (chance 0.001), yet noisy mAP is 0.008 below S0 (one seed).
- **Backbone linear probe.** We fit a ridge regression from student P4 to the teacher tokens. These checkpoints were trained for 1 epoch on the *old* split, so treat this as a diagnostic only. Files: `results/probe_1ep_ctx{1,3}.json`.

| Checkpoint | P@1 (1×1 context) | P@1 (3×3 context) |
|---|---|---|
| S2 (shuffled) | 0.0126 | 0.0213 |
| S0 (no KD) | 0.0129 | 0.0219 |
| S1 plain | 0.0153 | 0.0254 |
| COCO-pretrained (before iSAID) | 0.0167 | 0.0254 |
| S1 centered | 0.0208 | 0.0332 |

  In this table (1-epoch, old-split checkpoints, one seed), the shuffled-KD checkpoint is at the S0 level, plain KD is slightly higher, and the centered loss is highest. These are not the 10-epoch models above, and other loss variants we screened later (e.g. a contrastive one) reached a higher probe score without better noisy mAP — so probe scores should not be read as predicting robustness.

## Reproduce
- `kaggle/make_v23.py LAM INTERP CONDS SEEDS` generates the main Kaggle notebooks.
- `kaggle/make_v23x.py LAM INTERP CONDS SEEDS VARIANT` generates the exploratory ones.
- Each notebook embeds `code/seed_sweep_v2.py` and the split lists, and checks their sha256 before training.
- Dataset: Kaggle `redzapdos123/isaid-dataset-yolo11-seg-format`. The code builds a symlink mirror from the official lists and strips the label prefix itself.
- Tests: `code/tests/test_spatial_kd.py` (some tests need reference files that are not in this repo) and `analysis/test_probe_core.py`.

## Known limitations / next
- One seed per condition so far. We are planning additional seeds and the X2 control.
- Mosaic augmentation was effectively off in every run, because `close_mosaic = 10` equals the 10-epoch length. This holds for all conditions, so comparisons are fair, but absolute mAP may be lower than it could be.
- Planned next: a teacher comparison (DINOv2 resolution and size vs. a task-trained CNN), and a stronger baseline (mosaic on, wider training-noise range) with KD on top.
