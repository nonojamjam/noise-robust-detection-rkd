"""Create clean/noisy qualitative comparisons for baseline and KD models."""

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # File output only; works on headless servers and CI.
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO


def load_student_model(checkpoint: str):
    """Load either a legacy ``{'student': state_dict}`` or Ultralytics checkpoint."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "student" not in ckpt:
        print("[KD] standard Ultralytics checkpoint detected")
        return YOLO(checkpoint)
    student_sd = ckpt["student"]
    model = YOLO("yolov8s-seg.pt")
    result = model.model.load_state_dict(student_sd, strict=False)
    print(f"[KD] missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}")
    return model


def add_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma == 0:
        return image.copy()
    noisy = image.astype(np.float32) / 255.0
    noisy += rng.standard_normal(noisy.shape, dtype=np.float32) * sigma
    return (noisy.clip(0, 1) * 255).astype(np.uint8)


def run_detect(model: YOLO, image: np.ndarray, conf: float, imgsz: int) -> tuple[np.ndarray, int]:
    """Run detection and return the annotated image and detection count."""
    visual = image.copy()
    result = model(image, imgsz=imgsz, conf=conf, verbose=False)[0]
    count = 0
    if result.boxes is not None and len(result.boxes):
        for box in result.boxes:
            score = float(box.conf[0])
            if score < conf:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls[0])
            label = f"{result.names[class_id]} {score:.2f}" if class_id in result.names else f"class {class_id}"
            cv2.rectangle(visual, (x1, y1), (x2, y2), (30, 200, 50), 2)
            cv2.putText(visual, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 200, 50), 1)
            count += 1
    return visual, count


def make_2x2(baseline, kd, image_path, out_dir, conf, imgsz, rng, sigma_noisy=0.30):
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")
    original = cv2.resize(original, (imgsz, imgsz))
    clean, noisy = add_noise(original, 0.0, rng), add_noise(original, sigma_noisy, rng)
    cells = [
        ("Baseline (clean)", baseline, clean), ("KD (clean)", kd, clean),
        (f"Baseline (sigma={sigma_noisy})", baseline, noisy), (f"KD (sigma={sigma_noisy})", kd, noisy),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for (title, model, image), axis in zip(cells, axes.flat):
        visual, count = run_detect(model, image, conf, imgsz)
        axis.imshow(cv2.cvtColor(visual, cv2.COLOR_BGR2RGB))
        axis.set_title(f"{title}\n{count} detections", color="#1a7f37" if count else "#cf222e", fontweight="bold")
        axis.axis("off")
    fig.legend(handles=[mpatches.Patch(color="#1a7f37", label="detections"), mpatches.Patch(color="#cf222e", label="no detections")], loc="lower center", ncol=2)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    output = out_dir / "kd_noise_2x2.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def make_summary_row(baseline, kd, image_path, out_dir, conf, imgsz, rng, sigma=0.30):
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")
    original = cv2.resize(original, (imgsz, imgsz))
    clean, noisy = add_noise(original, 0.0, rng), add_noise(original, sigma, rng)
    panels = [(kd, clean, "KD (clean)"), (baseline, noisy, "Baseline (noisy)"), (kd, noisy, "KD (noisy)")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for axis, (model, image, title) in zip(axes, panels):
        visual, count = run_detect(model, image, conf, imgsz)
        axis.imshow(cv2.cvtColor(visual, cv2.COLOR_BGR2RGB))
        axis.set_title(f"{title}\n{count} detections", fontweight="bold")
        axis.axis("off")
    fig.tight_layout()
    output = out_dir / "kd_noise_summary.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize baseline and KD predictions on one image.")
    parser.add_argument("--baseline", required=True, help="Baseline Ultralytics checkpoint.")
    parser.add_argument(
        "--proposed", required=True,
        help="KD checkpoint: a standard Ultralytics .pt (including this trainer's best.pt), or legacy dict with 'student' key.",
    )
    parser.add_argument("--images", required=True, help="Input image path.")
    parser.add_argument("--out-dir", default="results/qualitative")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0, help="Seed for deterministic Gaussian noise.")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = YOLO(args.baseline)
    kd = load_student_model(args.proposed)
    rng = np.random.default_rng(args.seed)
    grid = make_2x2(baseline, kd, args.images, out_dir, args.conf, args.imgsz, rng)
    summary = make_summary_row(baseline, kd, args.images, out_dir, args.conf, args.imgsz, rng)
    print(f"Saved: {grid}\nSaved: {summary}")


if __name__ == "__main__":
    main()
