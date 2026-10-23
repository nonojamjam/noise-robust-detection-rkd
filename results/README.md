# Results

All numbers are from the June 2026 capstone presentation (iSAID, 16 classes, standalone eval `half=True`) and are **preliminary** — see [Known issues](../README.md#known-issues-found-after-the-june-presentation).

## 50-epoch comparison (not hyperparameter-matched)

| Condition | Epoch | mAP50 (Clean) | mAP50 (Noisy, σ=0.3) |
|---|---|---|---|
| BaseLine0 | 50 | 0.2775 | 0.0090 |
| **N1_MIXED (proposed)** | 50 | **0.4264** | **0.1961** |

## 10-epoch ablation

| Condition | KD | Noise | Clean mAP50 | Noisy mAP50 | KD Δ (Clean) |
|---|---|---|---|---|---|
| BaseLine0 ⚠️ | ✗ | ✗ | ~~0.1243~~ | ~~0.0152~~ | — (trained 2 epochs, not 10 — invalid) |
| C1 | ✅ | ✗ | 0.2849 | 0.0390 | — |
| A_NOISY | ✗ | ✅ | 0.1386 | 0.1728 | — |
| N1_MIXED | ✅ | ✅ | 0.2277 | 0.1799 | +0.0897 (vs. noise-only) |

## Qualitative (P1130 aircraft tile, σ=0.05)

| Model | Detections |
|---|---|
| BaseLine0 | 0 (total failure under noise) |
| N1_MIXED | 2 aircraft correctly detected |

> `visualize_comparison.py` regenerates the comparison figures.
> TODO: add side-by-side demo image (BaseLine0 vs. N1_MIXED) here.
