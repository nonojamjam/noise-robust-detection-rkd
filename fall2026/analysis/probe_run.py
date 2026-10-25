"""Linear probes of YOLO P4 features against the spatial DINOv2 teacher.

This is an exploratory diagnostic, not a preregistered decision procedure.
It deliberately execs ``code_v2/seed_sweep_v2.py`` only to reuse its locked
teacher, interpolation, normalization, and correspondence definitions.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox


LAYER, FEATURE_SHAPE, LAM_FACTORS = 18, (256, 40, 40), (1e-3, 1e-2, 1e-1, 1., 10., 100.)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_spatial_api(source):
    """Exec the training source without invoking its run() entry point."""
    g = {"EXP": "S1", "SEED": 0, "DATA_ROOT": None, "__name__": "__probe__",
         "__SRC_PATH__": str(source)}
    exec(compile(source.read_text(), str(source), "exec"), g)
    return g


def interpolate_features(features, mode):
    if tuple(features.shape[1:]) != FEATURE_SHAPE:
        raise RuntimeError(f"P4 must be Bx256x40x40, got {tuple(features.shape)}")
    if mode == "area":
        return F.interpolate(features.float(), size=(32, 32), mode="area")
    if mode == "bilinear_antialias":
        return F.interpolate(features.float(), size=(32, 32), mode="bilinear",
                             align_corners=False, antialias=True)
    raise ValueError(f"unsupported interpolation: {mode}")


class FeatureMoments:
    """Streaming per-channel moments over selected spatial cells."""
    def __init__(self, channels=256):
        self.sum = torch.zeros(channels, dtype=torch.float64)
        self.sq_sum = torch.zeros(channels, dtype=torch.float64)
        self.n = 0

    def add(self, s, mask, context=1):
        # 통계는 항상 256 채널 기준 — context 3 도 표준화한 뒤 이웃을 붙인다 (경계 0 패딩이 통계에 섞이지 않게, 검수 probe rev2 codex#1)
        x = s.permute(0, 2, 3, 1)[mask].double().cpu()
        self.sum += x.sum(0)
        self.sq_sum += x.square().sum(0)
        self.n += x.shape[0]

    def finish(self):
        if self.n == 0:
            raise RuntimeError("no valid cells for feature standardization")
        mean = self.sum / self.n
        var = (self.sq_sum / self.n - mean.square()).clamp_min(1e-12)
        return mean.float(), var.sqrt().float()


def contextual_features(s, context):
    """Return per-cell student features before standardization."""
    if context == 1:
        return s.permute(0, 2, 3, 1)
    if context == 3:
        return F.unfold(s, kernel_size=3, padding=1).transpose(1, 2).reshape(
            s.shape[0], s.shape[2], s.shape[3], -1)
    raise ValueError(f"unsupported context: {context}")


def design_rows(s, mask, mean, std, context=1):
    # Keep the context=1 path byte-for-byte equivalent in its tensor operations.
    if context == 1:
        x = s.permute(0, 2, 3, 1)[mask]
        x = (x - mean.to(x.device)) / std.to(x.device)
    else:
        z = (s - mean.to(s.device).view(1, -1, 1, 1)) / std.to(s.device).view(1, -1, 1, 1)
        x = contextual_features(z, context)[mask]   # 패딩 0 = 표준화된 평균
    return torch.cat((x, torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)), dim=1)


class NormalEquations:
    """Streaming X'X and X'Y accumulation; never retains all cell rows."""
    def __init__(self, inputs=257, outputs=768):
        self.xtx = torch.zeros((inputs, inputs), dtype=torch.float64)
        self.xty = torch.zeros((inputs, outputs), dtype=torch.float64)
        self.n = 0

    def add(self, x, y):
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError("ridge rows must be matching two-dimensional matrices")
        x, y = x.double().cpu(), y.double().cpu()
        self.xtx += x.T @ x
        self.xty += x.T @ y
        self.n += x.shape[0]

    def merged(self, other):
        result = NormalEquations(self.xtx.shape[0], self.xty.shape[1])
        result.xtx, result.xty, result.n = self.xtx + other.xtx, self.xty + other.xty, self.n + other.n
        return result


def ridge_weights(eq, lam):
    eye = torch.eye(eq.xtx.shape[0], dtype=torch.float64)
    return torch.linalg.solve(eq.xtx + float(lam) * eye, eq.xty).float()


def cache_path(cache_dir, image_path, source_sha):
    key = hashlib.sha256((str(image_path.resolve()) + "\0" + source_sha).encode()).hexdigest()
    return cache_dir / f"{key}.npz"


def read_images(root, names, start, count, device, letterbox):
    imgs = []
    for name in names[start:start + count]:
        path = root / "images" / name
        im = cv2.imread(str(path))
        if im is None:
            raise FileNotFoundError(path)
        x = letterbox(image=im)[..., ::-1].transpose(2, 0, 1).copy()
        imgs.append(torch.from_numpy(x).float().div(255))
    return torch.stack(imgs).to(device), names[start:start + count]


def teacher_targets(teacher, clean, image_paths, cache_dir, source_sha, api):
    raws, masks = [], []
    missing = []
    for i, path in enumerate(image_paths):
        cp = cache_path(cache_dir, path, source_sha)
        if cp.is_file():
            with np.load(cp) as data:
                raws.append(torch.from_numpy(data["raw"]).float())
                masks.append(torch.from_numpy(data["mask"].astype(bool)))
        else:
            missing.append(i)
            raws.append(None); masks.append(None)
    if missing:
        raw, mask = api["teacher_patches"](teacher, clean[missing])
        for j, i in enumerate(missing):
            cp = cache_path(cache_dir, image_paths[i], source_sha)
            payload = {"raw": raw[j].detach().cpu().numpy().astype(np.float16),
                       "mask": mask[j].detach().cpu().numpy().astype(np.uint8)}
            # Complete bytes are prepared before the only write, as required.
            import io
            buf = io.BytesIO(); np.savez_compressed(buf, **payload)
            with open(cp, "wb") as f:
                f.write(buf.getvalue())
            raws[i], masks[i] = raw[j].detach().cpu().half().float(), mask[j].detach().cpu().bool()  # 캐시 적중 경로와 같은 fp16 값 (검수 rev1 codex#2)
    raw = torch.stack(raws).to(clean.device)
    return api["normalize_spatial"](raw), torch.stack(masks).to(clean.device)


def load_p4_model(path, device):
    model = YOLO(str(path)).model.float().eval().to(device)
    captured = {}
    handle = model.model[LAYER].register_forward_hook(
        lambda module, inputs, output: captured.__setitem__("p4", output.detach()))
    return model, captured, handle


def batches(root, names, batch_size, device, letterbox, teacher, cache_dir, source_sha, api):
    for start in range(0, len(names), batch_size):
        clean, current = read_images(root, names, start, batch_size, device, letterbox)
        paths = [root / "images" / name for name in current]
        target, mask = teacher_targets(teacher, clean, paths, cache_dir, source_sha, api)
        yield clean, target, mask


def run_model(model, captured, clean, mode):
    with torch.no_grad():
        model(clean)
    return interpolate_features(captured.pop("p4"), mode)


def accumulate_moments(model, captured, data, mode, context=1):
    moments = FeatureMoments(256)
    for clean, _, mask in data:
        moments.add(run_model(model, captured, clean, mode), mask, context)
    return moments.finish()


def accumulate_equations(model, captured, data, mode, mean, std, context=1):
    eq = NormalEquations(inputs=256 * context * context + 1)
    for clean, target, mask in data:
        s = run_model(model, captured, clean, mode)
        eq.add(design_rows(s, mask, mean, std, context), target.permute(0, 2, 3, 1)[mask])
    return eq


def predict(s, mask, mean, std, weights, normalize, context=1):
    x = design_rows(s, mask, mean, std, context)
    out = torch.zeros((s.shape[0], 32, 32, weights.shape[1]), dtype=torch.float32, device=s.device)
    out[mask] = x @ weights.to(s.device)
    return normalize(out.permute(0, 3, 1, 2))


def mean_cosine(model, captured, data, mode, mean, std, weights, normalize, context=1):
    total, loss_sum = 0, 0.0
    for clean, target, mask in data:
        pred = predict(run_model(model, captured, clean, mode), mask, mean, std, weights, normalize, context)
        cosine = (pred * target).sum(1)
        total += int(mask.sum())
        loss_sum += float((1 - cosine[mask]).sum())
    return loss_sum / total


def collect_stats(model, captured, data, mode, mean, std, weights, normalize, correspondence, context=1):
    sums = {"correct": 0.0, "chance_correct": 0.0, "plain": 0.0, "centered": 0.0, "n": 0}
    for clean, target, mask in data:
        pred = predict(run_model(model, captured, clean, mode), mask, mean, std, weights, normalize, context)
        stat = correspondence(pred, target, mask)
        n = int(mask.sum())
        sums["correct"] += stat["p_at_1"] * n
        sums["chance_correct"] += stat["p_at_1_chance"] * n
        sums["plain"] += stat["plain_cos"] * n
        sums["centered"] += stat["centered_cos"] * n
        sums["n"] += n
    n = sums.pop("n")
    return {"p_at_1": sums["correct"] / n, "p_at_1_chance": sums["chance_correct"] / n,
            "plain_cos": sums["plain"] / n, "centered_cos": sums["centered"] / n,
            "n_val_cells": n}


def parse_args():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--train-root", required=True, type=Path)
    ap.add_argument("--val-root", required=True, type=Path)
    ap.add_argument("--val-list", default=here / "R_sample_500.txt", type=Path)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--context", type=int, choices=(1, 3), default=1)
    return ap.parse_args()


def main():
    a = parse_args()
    if a.n_train < 5 or a.batch_size < 1:
        raise ValueError("n-train must be at least 5 and batch-size must be positive")
    source = Path(__file__).resolve().parents[1] / "code_v2" / "seed_sweep_v2.py"
    source_sha = sha256(source)
    api = load_spatial_api(source)
    mode = api["INTERP_MODE"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a.cache.mkdir(parents=True, exist_ok=True)
    train_names = sorted(p.name for p in (a.train_root / "images").iterdir()
                         if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    selected = random.Random(a.seed).sample(train_names, a.n_train)
    cut = int(len(selected) * .8)
    fit_names, hold_names = selected[:cut], selected[cut:]
    val_names = [x.strip() for x in a.val_list.read_text().splitlines() if x.strip()]
    if not val_names or len(set(val_names)) != len(val_names):
        raise RuntimeError("val-list must contain unique image names")
    if any(not (a.val_root / "images" / x).is_file() for x in val_names):
        raise FileNotFoundError("a val-list image is absent from val-root/images")
    ckpts = {}
    for item in a.ckpt:
        if item.count("=") != 1:
            raise ValueError(f"--ckpt must be NAME=PATH: {item}")
        name, path = item.split("=", 1)
        if not name or name in ckpts or not Path(path).is_file():
            raise ValueError(f"invalid or duplicate checkpoint: {item}")
        ckpts[name] = Path(path)
    print(f"[probe] device={device} train={len(selected)} ({len(fit_names)} fit/{len(hold_names)} holdout) val={len(val_names)} mode={mode} context={a.context}", flush=True)
    teacher = api["load_spatial_teacher"](device)
    lb = LetterBox((640, 640), auto=False, center=True)
    make_data = lambda root, names: lambda: batches(root, names, a.batch_size, device, lb, teacher, a.cache, source_sha, api)
    results = {}
    for name, path in ckpts.items():
        print(f"[probe] {name}: loading {path}", flush=True)
        model, captured, hook = load_p4_model(path, device)
        try:
            print(f"[probe] {name}: feature moments", flush=True)
            mean, std = accumulate_moments(model, captured, make_data(a.train_root, fit_names)(), mode, a.context)
            fit_eq = accumulate_equations(model, captured, make_data(a.train_root, fit_names)(), mode, mean, std, a.context)
            hold_eq = accumulate_equations(model, captured, make_data(a.train_root, hold_names)(), mode, mean, std, a.context)
            scale = float(torch.trace(fit_eq.xtx) / fit_eq.xtx.shape[0])
            losses, candidates = {}, []
            for factor in LAM_FACTORS:
                lam = factor * scale
                w = ridge_weights(fit_eq, lam)
                loss = mean_cosine(model, captured, make_data(a.train_root, hold_names)(), mode, mean, std, w, api["normalize_spatial"], a.context)
                losses[f"{lam:.17g}"] = loss
                candidates.append((loss, lam))
            _, best_lam = min(candidates, key=lambda x: x[0])
            final_eq = fit_eq.merged(hold_eq)
            weights = ridge_weights(final_eq, best_lam)
            random_w = torch.randn((256 * a.context * a.context + 1, 768), generator=torch.Generator().manual_seed(a.seed), dtype=torch.float32)
            print(f"[probe] {name}: evaluating lambda={best_lam:.8g}", flush=True)
            stats = collect_stats(model, captured, make_data(a.val_root, val_names)(), mode, mean, std, weights,
                                  api["normalize_spatial"], api["correspondence_stats"], a.context)
            random_stats = collect_stats(model, captured, make_data(a.val_root, val_names)(), mode, mean, std, random_w,
                                         api["normalize_spatial"], api["correspondence_stats"], a.context)
            stats.update(lambda_=best_lam, holdout_loss_per_lambda=losses,
                         n_train_cells=final_eq.n, random_map=random_stats)
            stats["lambda"] = stats.pop("lambda_")
            results[name] = stats
            print(f"[probe] {name}: P@1={stats['p_at_1']:.5f}, random={random_stats['p_at_1']:.5f}", flush=True)
        finally:
            hook.remove(); del model
            if device.type == "cuda": torch.cuda.empty_cache()
    output = {"checkpoints": results, "meta": {"seed": a.seed, "n_train": len(selected), "n_val": len(val_names), "context": a.context,
              "INTERP_MODE": mode, "TEACHER_NORM": api["TEACHER_NORM"]}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(output, ensure_ascii=False, indent=2)
    with open(str(a.out) + ".json", "w") as f: f.write(text)
    prov = {"code_sha256": sha256(__file__), "seed_sweep_v2_sha256": source_sha,
            "checkpoint_sha256": {name: sha256(path) for name, path in ckpts.items()},
            "val_list_sha256": sha256(a.val_list)}
    prov_text = json.dumps(prov, ensure_ascii=False, indent=2)
    with open(str(a.out) + ".prov.json", "w") as f: f.write(prov_text)
    print(f"[probe] wrote {a.out}.json and {a.out}.prov.json", flush=True)


if __name__ == "__main__":
    main()
