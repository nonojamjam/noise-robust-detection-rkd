# Noise-Robust Object Detection via Heterogeneous ViT → CNN Knowledge Distillation

**DINOv2 (ViT) → YOLOv8s-seg (CNN): mixed-noise training plus centred spatial distillation to keep a lightweight detector accurate under sensor noise.**

🔗 **[Live demo →](https://nonojamjam.github.io/noise-robust-detection-rkd/)** — drag the noise slider, then see the official-split results.

> Capstone research project, Chungbuk National University (Dept. of Electronic Engineering).
> ML and all code: **Juhyeok Park**. Data preparation and documentation: Jiwon Lee. Advisor: Prof. Hyeongwon Kim.

---

## TL;DR

Lightweight detectors degrade sharply under satellite/aerial sensor noise. A frozen self-supervised foundation model (DINOv2 ViT-B/14) teaches a small CNN detector (YOLOv8s-seg) across a **heterogeneous ViT→CNN gap**, combined with 50:50 mixed-noise training. The teacher and the alignment adapter are removed at inference, so there is **no additional inference cost**.

**Final results (graduation thesis v2.1, 2026.11)** — official iSAID split (18,732 / 6,048 patches), labels fixed, 10 epochs, box mAP50, Gaussian σ=0.3, every checkpoint re-evaluated on the same noisy images:

| Condition (seed 0) | Clean | Noisy | Noisy − S0 |
|---|---|---|---|
| BL0 — no noise training* | 0.631 | 0.018 | — |
| S0 — noise training, no KD | 0.597 | 0.381 | — |
| N1 — + relational KD (the June method on `main`) | 0.593 | 0.382 | +0.001 |
| S1 — + plain spatial KD, λ 0.5 | 0.596 | 0.374 | −0.008 |
| **X1 — + centred spatial KD, λ 2.5** | **0.611** | **0.401** | **+0.020** |
| X2 — X1 with teacher positions shuffled | 0.587 | 0.375 | −0.006 |

\* different GPU/software stack. Seed 1: X1 − S0 = +0.010, N1 − S0 = −0.002. Noise training does most of the work; relational KD adds nothing; centred spatial KD adds a smaller, repeated gain that disappears when the teacher's positions are shuffled (seed 0). Full table (incl. S2 and P), σ sweep, bootstrap, code and raw numbers: [`fall2026-heterogeneous-kd`](https://github.com/nonojamjam/noise-robust-detection-rkd/tree/fall2026-heterogeneous-kd/fall2026) branch.

> **The code on `main` and the June numbers further below are the June 2026 presentation version** (relational KD). They were measured on a leaky public split with inflated boxes — see [Known issues](#known-issues-found-after-the-june-presentation) — and are kept only as a record.

| June 2026 condition (50 epochs; **not hyperparameter-matched**, superseded) | box mAP50 (Clean) | box mAP50 (Noisy, σ=0.3) |
|---|---|---|
| BaseLine0 (plain transfer learning) | 0.2775 | 0.0090 |
| N1_MIXED | 0.4264 | 0.1961 |

Final deployed model: `YOLOv8s-seg`, **11.8M params / 22MB** — teacher + adapter fully removed at inference.

---

## Problem

- Satellite/aerial imagery is critical infrastructure (defense, disaster response, urban planning) and demands **real-time edge inference**.
- Targets are **tiny objects** (tens of pixels); long-range optics + atmospheric interference cause **inevitable image degradation**.
- Edge deployment forces **lightweight models** → limited representational capacity → **fragile under noise**.
- Existing KD assumes **homogeneous** teacher/student architectures, and there is little work on KD **specifically for noise robustness**. ← the gap this project targets.

## Method

| Component | Choice | Rationale |
|---|---|---|
| Teacher | DINOv2 ViT-B/14 (frozen) | self-supervised general spatial-topological features; CLS-token representation |
| Teacher input | 448px resize of [0,1] image | intentionally no ImageNet mean/std normalization |
| Student | YOLOv8s-seg | real-time, lightweight (11.8M); features hooked at SPPF |
| Alignment | bottleneck MLP adapter (`C → 256 → 768`) | bridges ViT↔CNN dimensionality; **detached at inference** |
| Adapter LR | `lr0` (default 5e-5) | model uses the `optimizer=auto` LR; unchanged from the June original |
| Loss | **distance-wise RKD** + YOLO det/seg loss | preserves **pairwise sample-distance structure** even when per-pixel features are noise-corrupted |
| Data | iSAID (16 cls) + **50:50 mixed-noise** training | drives Clean and Noisy domains jointly; evaluated at σ=0.3 |

Training noise is σ=50/255 while evaluation uses σ=0.3, deliberately testing robustness under noise stronger than training.

**Core idea:** noise corrupts individual feature magnitudes, but the *relational geometry* between samples (encoded by the foundation model) is more stable. Transferring that geometry via RKD keeps the small student generalizing under degradation.

![Training pipeline (fall 2026 thesis version)](docs/assets/pipeline.png)

*Training pipeline as drawn in the fall 2026 graduation thesis (Korean labels). This is the **follow-up spatial KD** version: DINOv2 patch tokens (768×32×32) aligned to the student's neck P4 feature through a training-only conv adapter. The June method described on this page uses the teacher CLS token, the SPPF feature, an MLP adapter and a distance-wise RKD loss instead — see the diagram below and the [`fall2026-heterogeneous-kd`](https://github.com/nonojamjam/noise-robust-detection-rkd/tree/fall2026-heterogeneous-kd/fall2026) branch for the spatial version. In both, the teacher and adapter are removed at inference.*

```
Student (YOLOv8s) ──SPPF feature──► Adapter (MLP) ──┐
                                                     ├──► distance-wise RKD loss
Teacher (DINOv2, frozen, noise-free augmented input) ──CLS token───┘
Total loss = YOLO(det+seg) + λ · RKD     (λ = 0.5)
At inference: teacher + adapter removed → plain YOLOv8s-seg.
```

## June 2026 results (superseded)

**Ablation (10 epochs) — isolating KD vs. noise contributions:**

| Condition | KD | Noise | Clean box mAP50 | Noisy box mAP50 | KD Δ (Clean) |
|---|---|---|---|---|---|
| BaseLine0 ⚠️ | ✗ | ✗ | ~~0.1243~~ | ~~0.0152~~ | — (trained 2 epochs, not 10 — invalid; Known issues 1) |
| C1 | ✅ | ✗ | 0.2849 | 0.0390 | — |
| A_NOISY | ✗ | ✅ | 0.1386 | 0.1728 | — |
| **N1_MIXED** | ✅ | ✅ | 0.2277 | 0.1799 | +0.0897 (vs. noise-only) |

- Comparisons against the BaseLine0 row are withdrawn (Known issues 1); the remaining rows are as presented and share the data issues listed below.

**Qualitative (P1130 aircraft tile, σ=0.05):** BaseLine0 detects **0** objects (total failure); N1_MIXED detects **2** aircraft correctly.

## Known issues (found after the June presentation)

The June 2026 numbers on this page are the capstone presentation results, kept as presented except for the two corrections marked below. Later audits found problems that affect **all** of them; they are superseded by the official-split re-run in the TL;DR.

1. **10-epoch baseline was not 10 epochs.** The BaseLine0 checkpoint behind the ablation row was trained for 2 epochs (`train_args.epochs=2`, batch 16). Comparisons against that row — including "KD alone +0.1606" — are invalid and have been removed.
2. **The 50-epoch pair is not hyperparameter-matched** (BaseLine0: batch 2, imgsz 800, AdamW; N1_MIXED: batch 8, imgsz 640, auto). It is not an effect size.
3. **Train/validation overlap in the dataset version used.** In the public Kaggle conversion we used, most validation tiles are byte-identical copies of training tiles, so validation mAP was largely measured on images seen during training.
4. **Label-format issue.** Each label line carries a `cx cy w h` box prefix before the polygon; the loader read it as extra polygon vertices, inflating ground-truth boxes.

All conditions were re-run on the official iSAID train/val split with the label fix and matched settings; see the TL;DR table and the `fall2026-heterogeneous-kd` branch.

## Contributions

All ML and code by **Juhyeok Park**; data preparation and documentation by Jiwon Lee.

- **Heterogeneous RKD loss** — formulated and implemented the distance-wise relational loss to align a ViT teacher with a CNN student (`src/n1mixed_kd_trainer.py:rkd_loss`).
- **Custom Ultralytics trainer** (`N1MixedKDTrainer`) — SPPF forward-hook feature extraction, MLP adapter, and 50:50 mixed-noise injection inside the training loop, with the teacher seeing noise-free (augmented) input.
- **Resolved 3 integration issues with the Ultralytics training loop** that blocked training:
  - Ultralytics 8.4.x `model.loss` wrapping
  - teacher/student optimizer **param-group mismatch on resume** (auto optimizer-strip)
  - adapter weights lost on checkpoint save → **adapter sidecar** save/load
- **Experiment design** — 4-condition ablation, σ-sweep noise evaluation, qualitative demo.

## Limitations & next directions

- Two training seeds for the main comparison (S0, N1, X1); one for the shuffled controls and P.
- Centring and λ changed together in X1; one plain λ 2.5 run (P = 0.391) lands in between, so they are not separated.
- Gaussian noise + iSAID only; at σ = 0.4 the sign of X1 − S0 depends on the seed. 10 epochs, mosaic effectively off.

**Next:** more seeds and a λ grid for the centred loss, compound degradation and other datasets (DOTA/xView), and **federated-learning integration** for data that cannot leave the node (*a SCAFFOLD/FedAvg prototype exists; exploratory, not part of the benchmark*).

## Repository

```
src/n1mixed_kd_trainer.py   # core: RKD loss + custom KD trainer + DINOv2 teacher
eval/noise_sweep.py         # σ-sweep evaluation on iSAID
results/                    # comparison figures / tables
configs/                    # hyperparameters
```

### Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python src/n1mixed_kd_trainer.py --data /path/to/data.yaml --device 0
```
The DINOv2 teacher is fetched automatically via `torch.hub` on first run. A CUDA GPU is strongly recommended; CPU is supported for smoke tests or small runs (`--device cpu`) but will be very slow. Use a batch size of at least 3: the distance-wise RKD loss normalizes by the mean pairwise distance, so with 2 samples it is identically zero (default batch is 8).

`--resume --weights /path/to/epochN.pt` is only for an interrupted run whose checkpoint retains its epoch state (and matching `_kd_adapter.pt` sidecar). A completed `last.pt` is intentionally rejected; start a new run with `--weights /path/to/last.pt` and no `--resume` instead.

### Data preparation

Use the Kaggle dataset [`redzapdos123/isaid-dataset-yolo11-seg-format`](https://www.kaggle.com/datasets/redzapdos123/isaid-dataset-yolo11-seg-format), or provide iSAID in standard YOLO-seg layout: a `data.yaml`, `images/{train,val}`, and matching `labels/{train,val}` polygon files. This repository expects 16 classes.

The cited Kaggle conversion has the Known issue 4 label variant: every line begins with `cls cx cy w h` before the polygon points. Ultralytics YOLO-seg expects `cls x1 y1 x2 y2 ...`; using those prefixed files unchanged makes boxes appear inflated. Remove the four box values or use corrected labels before treating any metric as meaningful.

### Evaluation and qualitative visualization

```bash
python eval/noise_sweep.py --data_yaml /path/to/data.yaml \
  --models n1mixed:runs/n1mixed/N1_MIXED/weights/best.pt \
  --sigmas 0.0 0.3 --max_images 50 --device 0

python results/visualize_comparison.py --baseline /path/to/baseline.pt \
  --proposed runs/n1mixed/N1_MIXED/weights/best.pt --images /path/to/image.png
```

`visualize_comparison.py` accepts both a normal Ultralytics checkpoint (including this trainer's `best.pt`) and the legacy checkpoint dictionary format containing a `student` key.

> **Reproducibility note:** trained on Kaggle A100. The public quick-start path was tested with Python 3.12, PyTorch 2.14.1, torchvision 0.29.1, Ultralytics 8.4.184, NumPy 2.5.4, and PyYAML 6.0.3 (other Python versions untested; PyTorch had no wheel for Python 3.15 at test time). There is no environment-specific CONFIG path to edit: pass `--data`, `--project`, `--name`, and `--device` on the command line. Key settings remain 50 epochs, batch 8, imgsz 640, λ=0.5, mixed-noise ratio 0.5, eval σ=0.3.
