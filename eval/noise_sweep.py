"""Evaluate iSAID box and mask mAP@50 over Gaussian noise levels.

Example:
  python eval/noise_sweep.py --data_yaml data.yaml \
    --models baseline:weights/baseline.pt kd:weights/kd.pt \
    --sigmas 0.0 0.1 0.2 0.3 --out_dir eval_out/isaid_sweep
"""
import argparse, csv, json, os, random, shutil, sys
from pathlib import Path

import cv2
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


def apply_noise_np(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Add clipped Gaussian noise to a float32 HWC image in [0, 1]."""
    if sigma == 0.0:
        return arr
    return (arr + sigma * np.random.randn(*arr.shape).astype(np.float32)).clip(0.0, 1.0)


def label_path_for(img_path: Path) -> Path:
    """Map an image path to its corresponding label path."""
    parts = list(img_path.parts)
    idxs = [i for i, x in enumerate(parts) if x == "images"]
    if not idxs:
        raise ValueError(f"'images' 디렉토리 없는 경로: {img_path}")
    idx = idxs[-1]
    return Path(*parts[:idx], "labels", *parts[idx + 1:]).with_suffix(".txt")


# Build a noisy validation cache below --out_dir.
def make_noisy_val(yaml_path: str, out_dir: str, sigma: float, max_n: int) -> str:
    import yaml

    done = os.path.join(out_dir, ".done")
    cached_yaml = os.path.join(out_dir, "noisy_eval.yaml")
    if os.path.exists(done) and os.path.exists(cached_yaml):
        print(f"    [cache] 재사용: {out_dir}")
        return cached_yaml

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    base = cfg.get("path", str(Path(yaml_path).parent))
    val_rel = cfg.get("val", "val/images")
    val_dir = Path(base) / val_rel
    if not val_dir.exists():
        val_dir = Path(yaml_path).parent / val_rel
    if not val_dir.exists():
        raise FileNotFoundError(f"val dir not found: {val_dir}")

    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        imgs += val_dir.rglob(ext)
    imgs = sorted(imgs)
    if len(imgs) > max_n:
        random.seed(42)
        imgs = random.sample(imgs, max_n)

    out_img = Path(out_dir) / "images"
    out_lbl = Path(out_dir) / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    ok = 0
    for p in imgs:
        try:
            lbl = label_path_for(p)
        except ValueError:
            continue
        if not lbl.exists():
            continue
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        if sigma > 0:
            arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            arr = apply_noise_np(arr, sigma)
            bgr = cv2.cvtColor((arr * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
        # Avoid JPEG re-encoding changes to the injected noise.
        cv2.imwrite(str((out_img / p.stem).with_suffix(".png")), bgr)
        shutil.copy(str(lbl), str(out_lbl / (p.stem + ".txt")))
        ok += 1

    print(f"    [noisy_val] σ={sigma:.2f}: {ok}장 생성")
    if ok == 0:
        raise RuntimeError(f"노이즈 val 0장 — val 이미지/라벨 경로 확인: {val_dir}")

    # Normalize names for Ultralytics.
    names = cfg.get("names")
    nc = int(cfg.get("nc", 16))
    if names is None:
        names = [str(i) for i in range(nc)]
    elif isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
        nc = len(names)
    else:
        nc = len(names)
    with open(cached_yaml, "w") as f:
        yaml.dump({
            "path": str(Path(out_dir).resolve()),
            "train": "images",
            "val": "images",
            "nc": nc,
            "names": names,
        }, f)
    with open(done, "w") as f:
        f.write(f"gaussian sigma={sigma}")
    return cached_yaml


def load_p2_model(path: str):
    from ultralytics import YOLO
    try:
        return YOLO(path)
    except Exception as e:
        raise RuntimeError(
            f"모델 로드 실패: {path}\n"
            f"Expected an iSAID-compatible Ultralytics checkpoint. Original error: {e}")


def eval_box_seg(model, yaml_path, sigma, out_dir, max_images, device):
    cache_key = f"s{sigma:.4f}_n{max_images}_png"
    cache_dir = os.path.join(out_dir, "noisy_val_cache", cache_key)
    eval_yaml = make_noisy_val(yaml_path, cache_dir, sigma, max_images)

    box50 = seg50 = -1.0
    try:
        # Keep validation artifacts under the requested output directory instead
        # of inheriting a possibly unavailable project path from the checkpoint.
        res = model.val(data=eval_yaml, imgsz=640, batch=8, verbose=False,
                        device=device, plots=False, save=False, project=out_dir,
                        name="ultralytics_val", exist_ok=True)
        try:
            box50 = float(res.box.map50)
        except Exception:
            pass
        try:
            seg50 = float(res.seg.map50)
        except Exception:
            pass
    except Exception as e:
        print(f"    [ERROR] val 실패: {e}")
    return box50, seg50


def plot_curve(results, sigmas, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib 없음 — 표만 사용: {e}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, title in [(axes[0], "box", "Box mAP@50"),
                              (axes[1], "seg", "Seg(Mask) mAP@50")]:
        for label, per_sigma in results.items():
            ys = [per_sigma.get(str(s), {}).get(metric, float("nan")) for s in sigmas]
            ax.plot(sigmas, ys, "o-", label=label)
        ax.set_xlabel("noise σ"); ax.set_ylabel(title)
        ax.set_title(f"iSAID noise robustness — {title}")
        ax.grid(alpha=0.3); ax.legend()
    out_png = os.path.join(out_dir, "isaid_noise_curve.png")
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)
    print(f"[saved] {out_png}")


def parse_args():
    p = argparse.ArgumentParser(description="iSAID noise sigma sweep (box and segmentation mAP)")
    p.add_argument("--data_yaml", required=True, help="iSAID yaml (nc=16)")
    p.add_argument("--models", nargs="+", required=True,
                   help="label:best.pt 쌍들. 예) baseline:.../best.pt kd:.../best.pt")
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3])
    p.add_argument("--max_images", type=int, default=300)
    p.add_argument("--out_dir", default="eval_out/isaid_sweep")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[iSAID sweep] device={args.device} sigmas={args.sigmas} max={args.max_images}")

    # Parse label:path model specifications.
    model_specs = []
    for item in args.models:
        if ":" not in item:
            print(f"[SKIP] 'label:path' 형식 아님: {item}")
            continue
        label, path = item.split(":", 1)
        if not os.path.exists(path):
            print(f"[SKIP] {label}: {path} 없음")
            continue
        model_specs.append((label, path))

    if not model_specs:
        print("[error] No usable model checkpoints were provided.")
        return

    results = {}  # label -> {str(sigma) -> {"box":, "seg":}}
    for label, path in model_specs:
        print(f"\n{'='*55}\n  {label.upper()}: {path}\n{'='*55}")
        model = load_p2_model(path)
        results[label] = {}
        for sigma in args.sigmas:
            print(f"\n  σ = {sigma:.2f} ...")
            box50, seg50 = eval_box_seg(model, args.data_yaml, sigma,
                                        args.out_dir, args.max_images, args.device)
            results[label][str(sigma)] = {"box": round(box50, 6), "seg": round(seg50, 6)}
            print(f"  → Box {box50:.4f} | Seg {seg50:.4f}")
        del model

    # Print segmentation and box metrics.
    print("\n\n" + "━" * 64)
    print("  Table. iSAID σ-sweep  (Seg mAP@50, 괄호=Box)")
    print("━" * 64)
    header = f"  {'σ':<6}" + "".join(f"{lab:<20}" for lab, _ in model_specs)
    print(header); print("  " + "-" * 58)
    for sigma in args.sigmas:
        row = f"  {sigma:<6.2f}"
        for lab, _ in model_specs:
            cell = results[lab].get(str(sigma), {})
            s, b = cell.get("seg", -1), cell.get("box", -1)
            row += f"{(f'{s:.4f} ({b:.3f})' if s >= 0 else '—'):<20}"
        print(row)
    # Report the relative segmentation change from the first to final sigma.
    print("  " + "-" * 58)
    print("  [clean 대비 Seg mAP 상대 변화]")
    for lab, _ in model_specs:
        c = results[lab].get(str(args.sigmas[0]), {}).get("seg", -1)
        if c and c > 0:
            last = results[lab].get(str(args.sigmas[-1]), {}).get("seg", -1)
            if last >= 0:
                rel = (last - c) / c * 100
                print(f"    {lab:<12}: σ{args.sigmas[0]}→σ{args.sigmas[-1]}  {rel:+.1f}%")
    print("━" * 64)

    # Save machine-readable results and a curve.
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sigma"] + [f"{l}_box" for l, _ in model_specs]
                   + [f"{l}_seg" for l, _ in model_specs])
        for sigma in args.sigmas:
            r = [sigma]
            for l, _ in model_specs:
                r.append(results[l].get(str(sigma), {}).get("box", ""))
            for l, _ in model_specs:
                r.append(results[l].get(str(sigma), {}).get("seg", ""))
            w.writerow(r)
    plot_curve(results, args.sigmas, args.out_dir)
    print(f"\n[saved] {args.out_dir}/results.json, table.csv")


if __name__ == "__main__":
    main()
