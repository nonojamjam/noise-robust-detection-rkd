"""
방산(iSAID) 노이즈 강건성 σ-sweep — P2로 학습된 모델들의 노이즈별 mAP 비교.

졸작 보완 #2 (검증 구멍 메우기): "COCO에서 -2.7% 악화"가 방산에서도 같은지 직접 측정.
COCO용 eval_kd_noise_sweep_coco.py를 iSAID(nc=16, seg)로 적응 + box·seg 둘 다 + 곡선 PNG.

⚠️ 입력은 'P2로 iSAID head까지 학습된 ultralytics best.pt' 여야 함 (backbone 단독 불가 — nc=16 head 필요).
   즉 finetune_isaid_head.py를 먼저 돌린 산출물(runs_*/ft/weights/best.pt)을 평가한다.

평가 방식: val 이미지에 가우시안 노이즈를 디스크에 캐싱(견고) → model.val() → box·seg mAP50.

사용:
  # P2 먼저 (COCO 백본 / NR 백본) → 각각 best.pt 생성된 뒤:
  python eval_isaid_noise_sweep.py \
    --data_yaml isaid_data.yaml \
    --models baseline:weights/runs_baseline_head/ft/weights/best.pt \
             kd:weights/runs_kd_nr_head/ft/weights/best.pt \
    --sigmas 0.0 0.1 0.2 0.3 \
    --max_images 300 \
    --out_dir eval_out/isaid_sweep
"""
import argparse, csv, json, os, random, shutil, sys
from pathlib import Path

import cv2
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


# ── 노이즈 ──────────────────────────────────────────────────────────────────
def apply_noise_np(arr: np.ndarray, sigma: float) -> np.ndarray:
    """float32 [0,1] HWC numpy → gaussian noise (clip 0~1)."""
    if sigma == 0.0:
        return arr
    return (arr + sigma * np.random.randn(*arr.shape).astype(np.float32)).clip(0.0, 1.0)


def label_path_for(img_path: Path) -> Path:
    """이미지 경로 → 라벨 경로. 문자열 replace 대신 path-part 치환 (GPT 검수: 견고)."""
    parts = list(img_path.parts)
    idxs = [i for i, x in enumerate(parts) if x == "images"]
    if not idxs:
        raise ValueError(f"'images' 디렉토리 없는 경로: {img_path}")
    idx = idxs[-1]  # 가장 안쪽 images 기준
    return Path(*parts[:idx], "labels", *parts[idx + 1:]).with_suffix(".txt")


# ── 노이즈 val 캐시 생성 (디스크에 노이즈 이미지 + 라벨 복사 + 임시 yaml) ──────
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
        random.seed(42)                 # σ 간 동일 이미지 집합 보장 (공정)
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
        # PNG로 저장 — JPEG 재인코딩 손실이 σ 신호를 오염시키는 것 방지 (GPT 검수)
        cv2.imwrite(str((out_img / p.stem).with_suffix(".png")), bgr)
        shutil.copy(str(lbl), str(out_lbl / (p.stem + ".txt")))
        ok += 1

    print(f"    [noisy_val] σ={sigma:.2f}: {ok}장 생성")
    if ok == 0:
        raise RuntimeError(f"노이즈 val 0장 — val 이미지/라벨 경로 확인: {val_dir}")

    # names 항상 채우기 (없으면 인덱스명) + dict→list 정규화 (GPT 검수)
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


# ── 모델 로드 (P2 ultralytics best.pt, nc=16 native) ────────────────────────
def load_p2_model(path: str):
    from ultralytics import YOLO
    try:
        return YOLO(path)            # P2 best.pt = 완전한 nc=16 ultralytics ckpt
    except Exception as e:
        raise RuntimeError(
            f"모델 로드 실패: {path}\n"
            f"  → 이 스크립트는 finetune_isaid_head.py가 만든 'runs_*/ft/weights/best.pt'(nc=16) "
            f"를 입력으로 받습니다. try4kd 포맷(weights/p2_*.pt)이 아니라 best.pt 경로를 주세요.\n"
            f"  원본 에러: {e}")


# ── 평가 (box·seg mAP50 둘 다) ───────────────────────────────────────────────
def eval_box_seg(model, yaml_path, sigma, out_dir, max_images, device):
    cache_key = f"s{sigma:.4f}_n{max_images}_png"  # 정밀도+포맷 반영(stale 방지)
    cache_dir = os.path.join(out_dir, "noisy_val_cache", cache_key)
    eval_yaml = make_noisy_val(yaml_path, cache_dir, sigma, max_images)

    box50 = seg50 = -1.0
    try:
        res = model.val(data=eval_yaml, imgsz=640, batch=8, verbose=False,
                        device=device, plots=False, save=False)
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


# ── 곡선 PNG ─────────────────────────────────────────────────────────────────
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
    print(f"[saved] {out_png}  ← 방산 노이즈 강건성 슬라이드용")


# ── 메인 ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="iSAID 방산 노이즈 σ-sweep (box·seg)")
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

    # "label:path" 파싱
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
        print("[error] 평가할 모델 없음 — P2(finetune_isaid_head.py) 먼저 돌려 best.pt 생성 필요")
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

    # ── 표 (seg 기준 + clean 대비 drop) ──────────────────────────────────────
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
    # clean 대비 drop (seg)
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

    # ── 저장 + 곡선 ──────────────────────────────────────────────────────────
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
    print("\n해석: baseline보다 KD의 σ0→σ0.3 하락폭이 작으면 → 방산서 KD가 노이즈 강건.")
    print("      크면 → COCO와 동일한 한계가 방산서도 확인됨 (정직한 negative, 둘 다 의미).")


if __name__ == "__main__":
    main()
