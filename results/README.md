# Results

All numbers are from the submitted capstone benchmark (iSAID, 16 classes, standalone eval `half=True`).

## 50-epoch fair comparison

| Condition | Epoch | mAP50 (Clean) | mAP50 (Noisy, σ=0.3) |
|---|---|---|---|
| BaseLine0 | 50 | 0.2775 | 0.0090 |
| **N1_MIXED (proposed)** | 50 | **0.4264** | **0.1961** |

## 10-epoch ablation

| Condition | KD | Noise | Clean mAP50 | Noisy mAP50 | KD Δ (Clean) |
|---|---|---|---|---|---|
| BaseLine0 | ✗ | ✗ | 0.2840 | 0.0149 | — |
| C1 | ✅ | ✗ | 0.2974 | 0.0179 | +0.0134 |
| A_NOISY | ✗ | ✅ | 0.2252 | 0.1776 | — |
| N1_MIXED | ✅ | ✅ | 0.2312 | 0.1821 | +0.0060 (vs. noise-only) |

> **Correction (2026-09-08).** An earlier version of this table reported BaseLine0 as
> `0.1243 / 0.0152`. That checkpoint was trained for **2 epochs**, not 10 — the label said
> otherwise and it went unnoticed for three months. Every derived figure (`+0.1606`, and the
> `×11.x` recovery ratios that used it as denominator) was inflated as a result.
> All four conditions were re-measured on identical hardware and hyperparameters
> (10 epochs, batch 8, seed 0, sha `eada6d2f`) with a declared-vs-measured gate on the
> checkpoint's `train_args`. The numbers above are from that re-measurement.
>
> **Ratios are no longer reported.** A second seed moved the noise-recovery ratio from
> ×11.9 to ×15.5 — the denominator (BaseLine0 under noise, 0.011–0.015) is small enough that
> the ratio is unstable. The absolute difference is not: +0.1627 vs. +0.1637 across seeds.


## Qualitative (P1130 aircraft tile, σ=0.05)

| Model | Detections |
|---|---|
| BaseLine0 | 0 (total failure under noise) |
| N1_MIXED | 2 aircraft correctly detected |

> `visualize_comparison.py` regenerates the comparison figures.
> TODO: add side-by-side demo image (BaseLine0 vs. N1_MIXED) here.
