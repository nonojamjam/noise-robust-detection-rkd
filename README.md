# Noise-Robust Object Detection via Heterogeneous Relational Knowledge Distillation

**DINOv2 (ViT) → YOLOv8s-seg (CNN): transferring relational structure to keep a lightweight detector accurate under sensor noise.**

🔗 **[Live demo →](https://nonojamjam.github.io/noise-robust-detection-rkd/)** — drag the noise slider and watch the baseline go blind while the distilled model holds.

> Capstone research project, Chungbuk National University (Dept. of Electronic Engineering).
> ML and all code: **Juhyeok Park**. Data preparation and documentation: Jiwon Lee. Advisor: Prof. Hyeongwon Kim.

---

## TL;DR

Lightweight detectors degrade sharply under satellite/aerial sensor noise. I distill the **relational (pairwise-distance) structure** of a frozen self-supervised foundation model (DINOv2 ViT-B/14) into a small CNN detector (YOLOv8s-seg) across a **heterogeneous ViT→CNN gap**, combined with 50:50 mixed-noise training.

On iSAID (16 classes), under strong Gaussian noise (σ=0.3), this improves detection under noise and, in our experiments, also gains on the clean domain — with **no additional inference cost** (the teacher and the alignment adapter are removed at inference).

> All reported numbers are single-seed (seed=0) runs; see [Limitations](#honest-limitations--next-directions).

| Condition (50 epochs) | mAP50 (Clean) | mAP50 (Noisy, σ=0.3) |
|---|---|---|
| BaseLine0 (plain transfer learning) | 0.2775 | 0.0090 |
| **N1_MIXED (proposed)** | **0.4264** | **0.1961** |

> ⚠️ **These two runs do not share hyperparameters** and should not be read as a controlled
> comparison (checked 2026-09-08 by reading `train_args` off both checkpoints):
> BaseLine0 was trained with `batch=2, imgsz=800, optimizer=AdamW`; N1_MIXED with
> `batch=8, imgsz=640, optimizer=auto`. The 10-epoch ablation below **is** controlled —
> all four conditions share hardware, hyperparameters and code.

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
| Student | YOLOv8s-seg | real-time, lightweight (11.8M); features hooked at SPPF |
| Alignment | bottleneck MLP adapter (`C → 256 → 768`) | bridges ViT↔CNN dimensionality; **detached at inference** |
| Loss | **distance-wise RKD** + YOLO det/seg loss | preserves **pairwise sample-distance structure** even when per-pixel features are noise-corrupted |
| Data | iSAID (16 cls) + **50:50 mixed-noise** training | drives Clean and Noisy domains jointly; evaluated at σ=0.3 |

**Core idea:** noise corrupts individual feature magnitudes, but the *relational geometry* between samples (encoded by the foundation model) is more stable. Transferring that geometry via RKD keeps the small student generalizing under degradation.

<!-- TODO: add Method Overview figure -->
<!-- ![Method Overview](docs/pipeline.png) -->

```
Student (YOLOv8s) ──SPPF feature──► Adapter (MLP) ──┐
                                                     ├──► distance-wise RKD loss
Teacher (DINOv2, frozen, clean input) ──CLS token───┘
Total loss = YOLO(det+seg) + λ · RKD     (λ = 0.5)
At inference: teacher + adapter removed → plain YOLOv8s-seg.
```

## Results

**Ablation (10 epochs) — isolating KD vs. noise contributions:**

| Condition | KD | Noise | Clean mAP50 | Noisy mAP50 | KD Δ (Clean) |
|---|---|---|---|---|---|
| BaseLine0 | ✗ | ✗ | 0.2840 | 0.0149 | — |
| C1 | ✅ | ✗ | 0.2974 | 0.0179 | **+0.0134** |
| A_NOISY | ✗ | ✅ | 0.2252 | 0.1776 | — |
| **N1_MIXED** | ✅ | ✅ | 0.2312 | 0.1821 | +0.0060 (vs. noise-only) |

- **KD alone** (C1 vs. BaseLine0): **+0.0134** Clean mAP50 — heterogeneous ViT→CNN transfer helps, but modestly.
- **Mixed-noise training is the dominant factor** under noise: +0.163 (A_NOISY vs. BaseLine0), versus +0.005 from KD on top of it.
- The two are **additive under noise** (interaction ≈ +0.0015) but mixed-noise training **costs clean accuracy**: −0.060 ± 0.002 across two seeds.
- **At 10 epochs the two are not free of each other**: mixed-noise training trades ~0.06 clean mAP50
  for +0.163 under noise (−0.060 ± 0.002 across two seeds).
  Whether this is fundamental or an artifact of undertraining is **open** — all three conditions
  were still improving monotonically at epoch 10, and none had converged.

**Qualitative (P1130 aircraft tile, σ=0.05):** BaseLine0 detects **0** objects (total failure); N1_MIXED detects **2** aircraft correctly.

## Contributions

All ML and code by **Juhyeok Park**; data preparation and documentation by Jiwon Lee.

- **Heterogeneous RKD loss** — formulated and implemented the distance-wise relational loss to align a ViT teacher with a CNN student (`src/n1mixed_kd_trainer.py:rkd_loss`).
- **Custom Ultralytics trainer** (`N1MixedKDTrainer`) — SPPF forward-hook feature extraction, MLP adapter, and 50:50 mixed-noise injection inside the training loop, with the teacher always seeing clean input.
- **Resolved 3 framework-level bugs** that blocked training:
  - Ultralytics 8.4.x `model.loss` wrapping
  - teacher/student optimizer **param-group mismatch on resume** (auto optimizer-strip)
  - adapter weights lost on checkpoint save → **adapter sidecar** save/load
- **Experiment design** — 4-condition ablation, σ-sweep noise evaluation, qualitative demo.

## Honest limitations & next directions

This was completed under fixed compute/schedule, so:
- single seed (seed=0) → no confidence intervals yet;
- Gaussian noise + iSAID only → no cross-dataset (DOTA/xView) or JPEG/blur validation;
- λ fixed at 0.5 → no sensitivity sweep.

**Research roadmap** (these limitations *are* the next steps):
- multi-seed + λ grid search + σ robustness curve;
- real satellite raw data + compound degradation;
- **federated-learning integration** — for defense/security domains where data cannot leave the node. *A SCAFFOLD/FedAvg prototype exists (the original motivation behind the "Fed" name); it is exploratory and not part of the benchmark above.*

## Repository

```
src/n1mixed_kd_trainer.py   # core: RKD loss + custom KD trainer + DINOv2 teacher
eval/noise_sweep.py         # σ-sweep evaluation on iSAID
results/                    # comparison figures / tables
configs/                    # hyperparameters
```

### Quick start

```bash
pip install -r requirements.txt
# Edit the CONFIG block at the top of src/n1mixed_kd_trainer.py
# (checkpoint path, dataset yaml) for your environment, then:
python -c "from src.n1mixed_kd_trainer import run_n1mixed; run_n1mixed()"
```
The DINOv2 teacher is fetched automatically via `torch.hub`. Training was run on a Kaggle A100; see the reproducibility note below.

> **Reproducibility note:** trained on Kaggle A100 (Ultralytics 8.4.62+, PyTorch, iSAID YOLO-seg format). Paths in `src/n1mixed_kd_trainer.py` (CONFIG block) are set for that environment; adjust them to run elsewhere. Key settings: 50 epochs, batch 8, imgsz 640, λ=0.5, mixed-noise ratio 0.5, eval σ=0.3.
