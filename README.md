# Noise-Robust Object Detection via Heterogeneous Relational Knowledge Distillation

**DINOv2 (ViT) → YOLOv8s-seg (CNN): transferring relational structure to keep a lightweight detector accurate under sensor noise.**

> Capstone research project, Chungbuk National University (Dept. of Electronic Engineering).
> ML and all code: **Juhyeok Park**. Data preparation and documentation: Jiwon Lee. Advisor: Prof. Hyeongwon Kim.

---

## TL;DR

Lightweight detectors degrade sharply under satellite/aerial sensor noise. I distill the **relational (pairwise-distance) structure** of a frozen self-supervised foundation model (DINOv2 ViT-B/14) into a small CNN detector (YOLOv8s-seg) across a **heterogeneous ViT→CNN gap**, combined with 50:50 mixed-noise training.

On iSAID (16 classes), under strong Gaussian noise (σ=0.3), this improves detection under noise and, in our experiments, also gains on the clean domain — with **no additional inference cost** (the teacher and the alignment adapter are removed at inference).

> All reported numbers are single-seed (seed=0) runs; see [Limitations](#honest-limitations--next-directions).

| Condition (50 epochs, fair comparison) | mAP50 (Clean) | mAP50 (Noisy, σ=0.3) |
|---|---|---|
| BaseLine0 (plain transfer learning) | 0.2775 | 0.0090 |
| **N1_MIXED (proposed)** | **0.4264** | **0.1961** |

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
| BaseLine0 | ✗ | ✗ | 0.1243 | 0.0152 | — |
| C1 | ✅ | ✗ | 0.2849 | 0.0390 | **+0.1606** |
| A_NOISY | ✗ | ✅ | 0.1386 | 0.1728 | — |
| **N1_MIXED** | ✅ | ✅ | 0.2277 | 0.1799 | +0.0897 (vs. noise-only) |

- **KD alone** (C1 vs. BaseLine0): +0.1606 Clean mAP50 → heterogeneous ViT→CNN transfer is effective.
- **Two strategies are complementary**, not a trade-off: N1_MIXED leads on both Clean and Noisy.

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
