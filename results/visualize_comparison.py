"""
졸업발표용 정성 시각화 — COCO 이미지 기반 Baseline vs KD × Clean vs Noisy

목적:
  - COCO 실험 결과 (+3.6% clean, -2.7% noisy)를 정성적으로 뒷받침
  - 슬라이드용 2×2 그리드: [clean|noisy] × [baseline|KD]
  - 두 모델 모두 COCO 학습 → COCO 이미지에서 실제 검출 가능

사용:
  python visualize_kd_noise_comparison.py
  → figures/qualitative/kd_noise_2x2.png  (슬라이드용)
  → figures/qualitative/kd_noise_summary.png (1행 요약)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import ASSETS

OUT_DIR  = Path("figures/qualitative")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONF     = 0.25
IMG_SIZE = 640

# ── 모델 로드 ──────────────────────────────────────────────────────────────────
def load_student_model():
    """v9_best.pt에서 student 백본 추출 → yolov8s-seg 구조에 주입."""
    ckpt = torch.load("weights/v9_best.pt", map_location="cpu")
    student_sd = ckpt["student"]
    model = YOLO("yolov8s-seg.pt")
    result = model.model.load_state_dict(student_sd, strict=False)
    print(f"  [KD] missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}")
    return model


def add_noise(img_bgr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma == 0:
        return img_bgr.copy()
    noisy = img_bgr.astype(np.float32) / 255.0
    noisy += np.random.randn(*noisy.shape).astype(np.float32) * sigma
    return (noisy.clip(0, 1) * 255).astype(np.uint8)


def run_detect(model: YOLO, img_bgr: np.ndarray) -> tuple[np.ndarray, int]:
    """검출 실행 → (박스 그린 BGR, 검출수)"""
    vis = img_bgr.copy()
    res = model(img_bgr, imgsz=IMG_SIZE, conf=CONF, verbose=False)[0]
    boxes = res.boxes
    n = 0
    if boxes is not None and len(boxes):
        for box in boxes:
            conf = float(box.conf[0])
            if conf < CONF:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            label = f"{res.names[cls_id]} {conf:.2f}" if cls_id in res.names else f"cls{cls_id}"
            cv2.rectangle(vis, (x1, y1), (x2, y2), (30, 200, 50), 2)
            cv2.putText(vis, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 200, 50), 1)
            n += 1
    return vis, n


# ── 2×2 메인 그리드 ───────────────────────────────────────────────────────────
def make_2x2(baseline: YOLO, kd: YOLO, img_path: str, sigma_noisy: float = 0.30):
    orig = cv2.imread(img_path)
    orig = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))

    clean = add_noise(orig, 0.0)
    noisy = add_noise(orig, sigma_noisy)

    cells = [
        ("Baseline  (σ=0, clean)",   baseline, clean),
        ("KD Ours  (σ=0, clean)",    kd,       clean),
        (f"Baseline  (σ={sigma_noisy}, noisy)", baseline, noisy),
        (f"KD Ours  (σ={sigma_noisy}, noisy)",  kd,       noisy),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle(
        "Baseline vs KD — Clean / Noisy 검출 비교  (COCO val)",
        fontsize=14, fontweight="bold", y=0.99
    )

    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for (title, model, img), (r, c) in zip(cells, positions):
        vis_bgr, n = run_detect(model, img)
        rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

        ax = axes[r][c]
        ax.imshow(rgb)

        det_str = f"검출 {n}개" if n > 0 else "검출 없음"
        color   = "#1a7f37" if n > 0 else "#cf222e"
        ax.set_title(f"{title}\n{det_str}", fontsize=11,
                     color=color, fontweight="bold", pad=6)
        ax.axis("off")

        # 노이즈 행에 테두리 강조
        if r == 1:
            for spine in ax.spines.values():
                spine.set_edgecolor("#cf222e")
                spine.set_linewidth(2.5)
                spine.set_visible(True)

    # 범례
    legend = [
        mpatches.Patch(color="#1a7f37", label="검출 있음"),
        mpatches.Patch(color="#cf222e", label="검출 없음 / 저하"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2,
               fontsize=11, framealpha=0.9, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 0.99])
    out = OUT_DIR / "kd_noise_2x2.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\n★ 2×2 저장: {out}")
    return out


# ── 슬라이드용 1행 요약 (clean KD vs noisy baseline) ─────────────────────────
def make_summary_row(baseline: YOLO, kd: YOLO, img_path: str, sigma: float = 0.30):
    orig  = cv2.imread(img_path)
    orig  = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))
    clean = add_noise(orig, 0.0)
    noisy = add_noise(orig, sigma)

    vis_kd_clean, n_kd  = run_detect(kd,       clean)
    vis_bl_noisy, n_bl  = run_detect(baseline,  noisy)
    vis_kd_noisy, n_kd2 = run_detect(kd,        noisy)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Gram-KD: clean에서 우위, noisy(σ={sigma})에서 취약 — COCO 정성 확인",
                 fontsize=13, fontweight="bold")

    panels = [
        (vis_kd_clean,  f"KD (clean σ=0)\n검출 {n_kd}개",   "#1a7f37"),
        (vis_bl_noisy,  f"Baseline (noisy σ={sigma})\n검출 {n_bl}개", "#e36209"),
        (vis_kd_noisy,  f"KD (noisy σ={sigma})\n검출 {n_kd2}개",  "#cf222e" if n_kd2 <= n_kd else "#1a7f37"),
    ]
    for ax, (bgr, title, color) in zip(axes, panels):
        ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=12, color=color, fontweight="bold", pad=8)
        ax.axis("off")

    plt.tight_layout()
    out = OUT_DIR / "kd_noise_summary.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"★ 요약 1행 저장: {out}")
    return out


if __name__ == "__main__":
    print("=== Baseline vs KD 정성 시각화 ===")

    img_path = str(ASSETS / "bus.jpg")   # ultralytics 내장 COCO 버스 이미지
    print(f"이미지: {img_path}")

    print("\n[1/2] 모델 로드...")
    baseline = YOLO("weights/yolov8s-seg.pt")
    print("  [Baseline] yolov8s-seg.pt 로드 완료")
    kd = load_student_model()

    print("\n[2/2] 시각화 생성...")
    make_2x2(baseline, kd, img_path, sigma_noisy=0.30)
    make_summary_row(baseline, kd, img_path, sigma=0.30)

    print("\n완료! figures/qualitative/ 확인하세요.")
    print("  - kd_noise_2x2.png    (슬라이드 메인)")
    print("  - kd_noise_summary.png (1행 요약)")
