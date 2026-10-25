"""seed_sweep_v2.py — SPEC v2.1 spatial KD plus unchanged legacy trainers.

Spatial notebook entry (no resume):
    EXP, SEED = 'S1', 0
    __SRC_PATH__ = '/kaggle/working/seed_sweep_v2.py'
    # Optional, fixed across ALL spatial runs after smoke:
    # LAM_SP_SHARED = 0.5; INTERP_MODE = 'bilinear_antialias'
    exec(compile(open(__SRC_PATH__).read(), __SRC_PATH__, 'exec'))
    run()

S1/seed-0 FRACTION<1 smoke writes lambda_smoke.json without changing lambda
mid-run. Lock its recommendation before main runs. Confirm DET_REDUCTION in
the smoke log when changing ultralytics versions. S1r uses an extra ImageNet
teacher pass for the common anomaly mask; its KD targets use input norm=none.

Original script description follows:

왜 있나
------
캡스톤 2x2 ablation 은 seed=0 단일이라 분산 추정이 없었다. 그 결과
노이즈 조건에서의 KD 순기여 +0.0071 (N1_MIXED 0.1799 − A_NOISY 0.1728) 이
실효인지 시드 요동인지 판정할 수 없다. 이 스크립트가 그 판정을 만든다.

설계 원칙 — 이 파일의 존재 이유
------------------------------
**노이즈 주입 경로를 하나만 둔다.** 기존 코드베이스에는 구현이 두 벌 있었다:
  kd_v7_5_n1mixed_fulltime.py : 이미지 단위 (randperm)   ← 헤드라인 50ep 를 만든 코드
  kd_appendix/kd_trainer.py   : 배치 전체 (random.random())  ← 보고서 부록에 실린 코드
서로 다른 구현으로 얻은 두 수를 빼면 그 차이에 '노이즈 커리큘럼 차이'가 섞인다.
여기서는 **실제로 헤드라인을 만든 이미지 단위 방식**으로 통일한다.

그리고 A_NOISY 와 N1_MIXED 는 **같은 트레이너 클래스**를 쓴다.
차이는 KD 손실 항 하나뿐이다. 그래야 뺄셈이 KD 만 남긴다.

⚠️ 원본 kd_v7_5 는 preprocess_batch 의 노이즈 주입을 `if self._teacher is not None`
   으로 감싸고 있어, teacher 를 빼면 노이즈까지 함께 꺼진다. 여기서는 분리했다.

사용법 (Kaggle 노트북 셀)
------------------------
    EXP = 'A_NOISY'   # 또는 'N1_MIXED'
    SEED = 0
    exec(open('/kaggle/working/seed_sweep.py').read())
    run()
"""

import os
import glob
import random
import json
import hashlib
import shutil
import pathlib
import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from ultralytics import YOLO
from ultralytics.models.yolo.segment import SegmentationTrainer

# ══════════════════════════════════════════════
# 설정 — 보고서 기준에 맞춘다 (imgsz 640)
# ══════════════════════════════════════════════
EXP           = globals().get('EXP', 'A_NOISY')     # BL0 | A_NOISY | N1_MIXED
SEED          = int(globals().get('SEED', 0))
EPOCHS        = int(globals().get('EPOCHS', 10))
BATCH         = 8
# 2026-09-06: 6월 원본 3조건 체크포인트의 train_args 가 전부 workers=8 이었다.
#   Kaggle 에서 0 으로 두면 에폭당 72분(실측) — 원본 9.3분의 7.8배다.
#   0 은 Colab/Windows 회피책이 ablation.py 를 통해 딸려온 값이지 원본 설정이 아니다.
WORKERS       = int(globals().get('WORKERS', 8))
# 배선 점검용. 1.0 미만이면 학습 데이터를 그만큼만 쓴다 — 값은 못 쓰고 경로만 확인한다.
FRACTION      = float(globals().get('FRACTION', 1.0))
IMGSZ         = 640          # 보고서 168행 '입력 해상도 640'
LR0           = 1e-3         # 신규 학습(resume 아님) — BASE_OVERRIDES 와 동일
NOISE_SIGMA   = 50 / 255     # ≈0.196. 평가는 σ=0.3
MIXED_RATIO   = 0.5          # 배치 '안'에서 noisy 이미지 비율
DINOV2_SIZE   = 448          # 14의 배수
LAM_KD        = 0.5
EVAL_SIGMA    = 0.3          # 노이즈 평가 강도
SPLIT_LISTS   = globals().get('SPLIT_LISTS')
ALLOW_SPLIT_OVERLAP = bool(globals().get('ALLOW_SPLIT_OVERLAP', False))
EVAL_CKPT     = globals().get('EVAL_CKPT', 'last')
if EVAL_CKPT not in {'last', 'best'}:
    raise ValueError("EVAL_CKPT must be 'last' or 'best'")

# 2×2 를 표로 못박는다. 부정 논리(EXP != 'BL0')로 두면 조건이 늘 때 조용히 틀린다 —
#   HARD-002 검수에서 grok 이 정확히 이 자리를 짚었다("C1 을 넣으면 게이팅이 부족하다").
#   모르는 이름은 여기서 죽는다. 오타로 BL0 가 A_NOISY 처럼 도는 일이 없다.
_COND = {                          # EXP : (KD 쓰나, 노이즈 주입하나)
    'BL0':          (False, False),   # 베이스라인
    'C1_CLEAN_RKD': (True,  False),   # KD 단독
    'A_NOISY':      (False, True),    # 노이즈 단독
    'N1_MIXED':     (True,  True),    # 둘 다 = 제안 기법
    'S0':           (True,  True),
    'S1':           (True,  True),
    'S2':           (True,  True),
    'S1r':          (True,  True),
}
assert EXP in _COND, f'모르는 조건 {EXP!r} — 가능한 값: {list(_COND)}'
USE_KD, USE_NOISE = _COND[EXP]
# The legacy trainer and its hyperparameters below remain unchanged.
_SPATIAL_COND = {
    'S0': (False, 'imagenet'), 'S1': (False, 'imagenet'),
    'S2': (True, 'imagenet'), 'S1r': (False, 'none'),
}
IS_SPATIAL = EXP in _SPATIAL_COND
KD_MODE = 'spatial' if IS_SPATIAL else 'cls_rkd'
SHUFFLE, TEACHER_NORM = _SPATIAL_COND.get(EXP, (False, 'none'))
KD_LAYER = 18
MASK_TOP = 0.01
# Set once from S1/seed-0 smoke, then use the SAME value in S1/S2/S1r.
LAM_SP_SHARED = float(globals().get('LAM_SP_SHARED', globals().get('LAM_SP', 0.5)))
LAM_SP = 0.0 if EXP == 'S0' else LAM_SP_SHARED
# 붕괴 진단 (2026-10-23, noise_robust_kd/.panel_collapse): plain = 명세 v2.1 그대로 / center = 이미지별 유효칸 평균을 뺀 뒤 cos
# / varfloor = plain + 학생 공간에너지 하한. 기본 plain 이면 학습 동작은 이전과 같다(로그만 늘어남).
KD_LOSS_VARIANT = str(globals().get('KD_LOSS_VARIANT', 'plain'))
assert KD_LOSS_VARIANT in ('plain', 'center', 'varfloor', 'relation', 'infonce', 'fgcenter'), KD_LOSS_VARIANT
VAR_FLOOR_GAMMA = float(globals().get('VAR_FLOOR_GAMMA', 0.5))
VAR_FLOOR_W = float(globals().get('VAR_FLOOR_W', 1.0))
RELATION_K = int(globals().get('RELATION_K', 256))
INFONCE_K = int(globals().get('INFONCE_K', 256))
INFONCE_TAU = float(globals().get('INFONCE_TAU', 0.1))
FG_ALPHA = float(globals().get('FG_ALPHA', 3.0))
LAM_AUTO_R = globals().get('LAM_AUTO_R')
LAM_AUTO_R = None if LAM_AUTO_R is None else float(LAM_AUTO_R)
LAM_AUTO_STEPS = int(globals().get('LAM_AUTO_STEPS', 50))
ADAPTER_LR_MULT = globals().get('ADAPTER_LR_MULT')
ADAPTER_LR_MULT = None if ADAPTER_LR_MULT is None else float(ADAPTER_LR_MULT)
ADAPTER_WD = globals().get('ADAPTER_WD')
ADAPTER_WD = None if ADAPTER_WD is None else float(ADAPTER_WD)
_prealign_raw = globals().get('ADAPTER_PREALIGN_STEPS', 0)
assert float(_prealign_raw) == int(_prealign_raw) and int(_prealign_raw) >= 0, _prealign_raw  # 검수 rev2: 음수 소수가 int() 로 0 이 되지 않게
ADAPTER_PREALIGN_STEPS = int(_prealign_raw)
LAM_SCHEDULE = str(globals().get('LAM_SCHEDULE', 'const'))
LAM_WARMUP_FRAC = float(globals().get('LAM_WARMUP_FRAC', 0.1))
LAM_END_FRAC = float(globals().get('LAM_END_FRAC', 0.1))
INTERP_MODE = globals().get('INTERP_MODE', 'bilinear_antialias')
# Installed 8.4.39 returns component * B; BaseTrainer backpropagates loss.sum().
# Smoke must confirm this contract before a different ultralytics version is used.
DET_REDUCTION = globals().get('DET_REDUCTION', 'batch_sum')
DIAG_EVERY = int(globals().get('DIAG_EVERY', 50))   # KD 기울기 진단 주기 (스텝)
assert ADAPTER_LR_MULT is None or (np.isfinite(ADAPTER_LR_MULT) and ADAPTER_LR_MULT >= 0)
assert ADAPTER_WD is None or (np.isfinite(ADAPTER_WD) and ADAPTER_WD >= 0)
assert ADAPTER_PREALIGN_STEPS >= 0
assert not (ADAPTER_PREALIGN_STEPS and EXP == 'S0'), 'prealign needs KD (S0 has lambda 0)'
assert LAM_SCHEDULE in ('const', 'warmup', 'warmup_cos'), LAM_SCHEDULE
assert np.isfinite(LAM_WARMUP_FRAC) and 0 <= LAM_WARMUP_FRAC <= 1
assert np.isfinite(LAM_END_FRAC) and LAM_END_FRAC >= 0
if IS_SPATIAL:
    assert SEED in (0, 1, 2), 'spatial seeds must be 0, 1, 2'
    assert np.isfinite(LAM_SP) and LAM_SP >= 0
    assert RELATION_K > 1 and INFONCE_K > 1 and INFONCE_TAU > 0 and FG_ALPHA >= 1
    assert LAM_AUTO_R is None or (np.isfinite(LAM_AUTO_R) and LAM_AUTO_R > 0 and LAM_AUTO_STEPS > 0)
    assert INTERP_MODE in ('bilinear_antialias', 'area')
    assert DET_REDUCTION in ('batch_sum', 'image_mean')
PROJECT = globals().get('PROJECT', '/kaggle/working/seed_sweep')
RUN_NAME = (f'{EXP}_{EPOCHS}ep_seed{SEED}' if FRACTION >= 1.0
            else f'SMOKE_{EXP}_{EPOCHS}ep_seed{SEED}_frac{FRACTION}')
if KD_LOSS_VARIANT != 'plain':
    RUN_NAME += f'_{KD_LOSS_VARIANT}'
if LAM_AUTO_R is not None:   # 검수 rev1(codex·agy): 자동 보정 실행이 고정 λ 실행 폴더를 덮지 않게
    RUN_NAME += f'_lamR{LAM_AUTO_R:g}'
if ADAPTER_LR_MULT is not None:
    RUN_NAME += f'_alr{ADAPTER_LR_MULT:g}'
if ADAPTER_WD is not None:
    RUN_NAME += f'_awd{ADAPTER_WD:g}'
if ADAPTER_PREALIGN_STEPS:
    RUN_NAME += f'_pa{ADAPTER_PREALIGN_STEPS}'
if LAM_SCHEDULE == 'warmup':
    RUN_NAME += '_lw'
elif LAM_SCHEDULE == 'warmup_cos':
    RUN_NAME += '_lwc'
if LAM_WARMUP_FRAC != 0.1:
    RUN_NAME += f'_lwf{LAM_WARMUP_FRAC:g}'
if LAM_END_FRAC != 0.1:
    RUN_NAME += f'_lef{LAM_END_FRAC:g}'


def set_all_seeds(s: int) -> None:
    """torch/numpy/random 전부. Ultralytics 의 seed 인자만으로는 부족하다."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ══════════════════════════════════════════════
# 라벨 캐시 미러 (2026-10-22) — 작업대 T-0123 workbench/kaggle_cache_mirror.py 를 그대로 옮긴 것 (커밋 ee42665)
#   /kaggle/input 은 읽기 전용이라 labels.cache 가 저장되지 않아 실행마다 전 라벨을 재스캔했다
#   (10/22 스모크: 디스크 0.1 MB/s 세션에서 val 스캔만 약 2시간). 파일 단위 심볼릭 링크 미러를 /tmp 에 만든다
#   — ultralytics 가 경로를 resolve() 하므로 폴더 링크는 안 된다. /kaggle/working 에 두면 output 저장이 링크를 따라갈 수 있어 /tmp.
# ══════════════════════════════════════════════
def _labels_path(images):
    """Replace the last images component, leaving earlier components alone."""
    parts = list(images.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    raise ValueError(f"Image directory has no 'images' component: {images}")


def _present(path):
    # Broken symlinks also count as existing entries that must be preserved.
    return os.path.lexists(path)


def _mkdir(path):
    """Make real directories, refusing to traverse an existing directory link."""
    if path.is_symlink():
        raise FileExistsError(f"Mirror directory is a symlink: {path}")
    if not path.exists():
        _mkdir(path.parent)
        path.mkdir(exist_ok=True)
    elif not path.is_dir():
        raise FileExistsError(f"Mirror directory is a file: {path}")


def _mirror_tree(source, target, ancestors=frozenset()):
    resolved = source.resolve()
    if resolved in ancestors:
        raise ValueError(f"Directory symlink cycle: {source}")
    _mkdir(target)
    ancestors = ancestors | {resolved}
    for entry in source.iterdir():
        destination = target / entry.name
        if entry.is_dir():
            _mirror_tree(entry, destination, ancestors)
        elif entry.is_file() and not _present(destination):
            destination.symlink_to(entry.resolve())


def build_mirror(cfg, yaml_dir, dst, seed_cache_dir=None):
    """Mirror configured image/label trees and optionally restore saved caches."""
    yaml_dir = Path(yaml_dir).resolve()
    base = yaml_dir
    if cfg.get("path"):
        candidate = Path(cfg["path"])
        if not candidate.is_absolute():
            candidate = yaml_dir / candidate
        if candidate.exists():
            base = candidate
    dst = Path(dst).resolve()
    for split in ("train", "val", "test"):
        value = cfg.get(split)
        if isinstance(value, list):
            raise ValueError(f"{split} must be a directory string, not a list")
        if not isinstance(value, str):
            continue
        images = base / value
        if not images.is_dir():
            raise FileNotFoundError(images)
        labels = _labels_path(images)
        mirror_images = dst / value
        # A split must stay inside the mirror so writes never reach the source.
        try:
            mirror_images.resolve().relative_to(dst)
        except ValueError:
            raise ValueError(f"Split path escapes the mirror: {value}") from None
        mirror_labels = _labels_path(mirror_images)
        _mirror_tree(images, mirror_images)
        if labels.is_dir():
            _mirror_tree(labels, mirror_labels)
        if seed_cache_dir is not None:
            cache = mirror_labels.with_suffix(".cache")
            seed = Path(seed_cache_dir) / cache.relative_to(dst)
            if seed.is_file() and not _present(cache):
                _mkdir(cache.parent)
                shutil.copy2(seed, cache)
    result = dict(cfg)
    result["path"] = str(dst)
    return result


def export_caches(dst, out_dir):
    """Copy cache contents and return their sorted, POSIX relative paths."""
    dst, out_dir = Path(dst), Path(out_dir)
    caches = sorted(p for p in dst.rglob("*.cache") if p.is_file())
    _mkdir(out_dir)
    exported = []
    for cache in caches:
        relative = cache.relative_to(dst)
        target = out_dir / relative
        _mkdir(target.parent)
        # Replace links so exported caches are independent real files.
        if target.is_symlink():
            target.unlink()
        if cache.resolve() != target.resolve():
            shutil.copy2(cache, target)
        exported.append(relative.as_posix())
    return sorted(exported)


DATA_ROOT = globals().get('DATA_ROOT')                  # 데스크탑 등 Kaggle 밖 (2026-10-23)
MIRROR_DST = '/tmp/isaid_mirror'                       # 경로가 매 실행 같아야 캐시 해시가 맞는다
CACHE_EXPORT = '/kaggle/working/label_cache_export'    # 받아서 Kaggle 데이터셋으로 올릴 것


def _find_cache_seed():
    """이전에 올린 캐시 데이터셋(…/label_cache_export/train/labels.cache) 을 찾는다. 없으면 None."""
    for root in ('/kaggle/input', '/root/.cache/kagglehub'):
        hits = sorted(glob.glob(f'{root}/**/label_cache_export/train/labels.cache', recursive=True))
        if hits:
            return str(Path(hits[0]).parent.parent)
    return None


# ══════════════════════════════════════════════
# data.yaml 탐색 (kd_v7_5 에서 검증된 로직)
# ══════════════════════════════════════════════
LABEL_PREFIX_FIX = bool(globals().get('LABEL_PREFIX_FIX', True))
LABEL_FIX_STATS = {}
SPLIT_CHECK = None


def _dataset_base(cfg, yaml_dir):
    """Return the dataset root using the same path rules as build_mirror."""
    base = Path(yaml_dir).resolve()
    if cfg.get('path'):
        candidate = Path(cfg['path'])
        if not candidate.is_absolute():
            candidate = base / candidate
        if candidate.exists():
            base = candidate.resolve()
    return base


def _validated_split_lists(split_lists):
    if split_lists is None:
        return None
    if not isinstance(split_lists, dict) or set(split_lists) != {'train', 'val'}:
        raise ValueError("SPLIT_LISTS must be None or {'train': path, 'val': path}")
    out = {split: Path(split_lists[split]).expanduser() for split in ('train', 'val')}
    for split, path in out.items():
        if not path.is_file():
            raise FileNotFoundError(f'SPLIT_LISTS[{split!r}] not found: {path}')
    return out


def split_lists_manifest():
    """Hashes are evidence of the exact official split used for this run."""
    lists = _validated_split_lists(SPLIT_LISTS)
    if lists is None:
        return None
    return {split: {'path': str(path), 'sha256': _sha256(path)}
            for split, path in lists.items()}


def build_list_mirror(cfg, yaml_dir, dst, split_lists):
    """Build a train/val-only, file-link mirror from official relative-path lists."""
    lists = _validated_split_lists(split_lists)
    if lists is None:
        raise ValueError('build_list_mirror requires SPLIT_LISTS')
    base, dst = _dataset_base(cfg, yaml_dir), Path(dst).resolve()
    planned, seen_basenames = [], set()
    for split, list_path in lists.items():
        for line_no, raw in enumerate(list_path.read_text(encoding='utf-8').splitlines(), 1):
            rel_text = raw.strip()
            if not rel_text:
                continue
            rel = Path(rel_text)
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError(f'{list_path}:{line_no}: unsafe relative path {rel_text!r}')
            # 공식 재분할 목록은 원본 Kaggle 폴더를 가리킨다 — val 목록에 train/images 출신 줄이 정상으로 들어 있다 (검수 split rev1 codex#1·flash#1)
            if len(rel.parts) != 3 or rel.parts[0] not in ('train', 'val') or rel.parts[1] != 'images':
                raise ValueError(f'{list_path}:{line_no}: expected train|val/images/<basename>, got {rel_text!r}')
            source = base / rel
            label = _labels_path(source.parent) / f'{source.stem}.txt'
            if not source.is_file():
                raise FileNotFoundError(f'{list_path}:{line_no}: image missing: {source}')
            if not label.is_file():
                raise FileNotFoundError(f'{list_path}:{line_no}: label missing: {label}')
            if source.name in seen_basenames:
                raise ValueError(f'duplicate basename across split lists: {source.name}')
            seen_basenames.add(source.name)
            planned.append((split, source, label))

    # A reused /tmp mirror must not retain tiles outside the official lists.
    for split in ('train', 'val'):
        target = dst / split
        if target.is_symlink():
            raise FileExistsError(f'Mirror split directory is a symlink: {target}')
        if target.exists():
            shutil.rmtree(target)
    for split, source, label in planned:
        image_target = dst / split / 'images' / source.name
        label_target = dst / split / 'labels' / label.name
        _mkdir(image_target.parent)
        _mkdir(label_target.parent)
        image_target.symlink_to(source.resolve())
        label_target.symlink_to(label.resolve())

    result = dict(cfg)
    result.update({'path': str(dst), 'train': 'train/images', 'val': 'val/images'})
    return result


def check_split_disjoint(data_yaml):
    """Reject train/val scene overlap before training can consume contaminated data."""
    with open(data_yaml, encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    base = _dataset_base(cfg, Path(data_yaml).parent)

    def images_for(split):
        value = cfg.get(split)
        if not isinstance(value, str):
            raise ValueError(f'{split} in {data_yaml} must be an images directory string')
        images = Path(value)
        if not images.is_absolute():
            images = base / images
        if not images.is_dir():
            raise FileNotFoundError(f'{split} images directory missing: {images}')
        return sorted(p for p in images.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'})  # 검수 split rev1 flash#3

    train, val = images_for('train'), images_for('val')
    scene = lambda p: p.stem.split('_', 1)[0]
    train_ids, val_ids = {scene(p) for p in train}, {scene(p) for p in val}
    overlap = sorted(train_ids & val_ids)
    result = {
        'train_tiles': len(train), 'val_tiles': len(val),
        'train_scenes': len(train_ids), 'val_scenes': len(val_ids),
        'overlap': len(overlap), 'overlap_count': len(overlap),
        'overlap_examples': overlap[:5],
    }
    if overlap:
        message = (f"train/val scene overlap: {len(overlap)} scene(s); "
                   f"examples: {', '.join(overlap[:5])}")
        if not ALLOW_SPLIT_OVERLAP:
            raise RuntimeError(message)
        print(f'[data] WARNING: {message} (ALLOW_SPLIT_OVERLAP=True)')
    return result


def fix_label_prefix(mirror_root):
    """iSAID Kaggle 라벨은 줄마다 'cls cx cy w h x1 y1 …' 이다 (박스 4값 접두 — 2026-10-23 실측 train 908,799 · val 319,153 줄 100%).
    ultralytics 8.4.39 는 cls 뒤 값을 전부 다각형으로 읽고(data/utils.py:221) 박스를 segments2boxes 로 만들어
    가짜 꼭짓점 (cx,cy)·(w,h) 때문에 박스가 늘어난다 (예: 폭 0.0725 → 0.3462). 미러의 라벨 링크를 접두 뺀 실제 파일로 바꾼다.
    접두 판정: 뒤 다각형의 외접 박스와 앞 4값의 차이가 모두 0.005 미만일 때만 뺀다(실측 최대 0.0036). 아니면 줄을 그대로 둔다(개수 기록).
    이미 실제 파일(링크 아님)이면 건너뛴다 → 두 번 돌려도 같다."""
    st = {'files': 0, 'lines': 0, 'fixed': 0, 'kept': 0, 'skipped_files': 0}
    for lb in sorted(Path(mirror_root).rglob('labels/*.txt')):
        if not lb.is_symlink():
            st['skipped_files'] += 1
            continue
        out = []
        for line in lb.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            st['lines'] += 1
            v = np.asarray(parts[1:], dtype=np.float64)
            if len(v) >= 10 and len(v) % 2 == 0:
                poly = v[4:].reshape(-1, 2)
                bb = np.array([(poly[:, 0].min() + poly[:, 0].max()) / 2, (poly[:, 1].min() + poly[:, 1].max()) / 2,
                               np.ptp(poly[:, 0]), np.ptp(poly[:, 1])])
                if np.abs(bb - v[:4]).max() < 0.005:     # 실측 최대 0.0036 (10/23 전수, 검수 labelfix codex#3)
                    out.append(' '.join([parts[0]] + parts[5:]))
                    st['fixed'] += 1
                    continue
            out.append(line)
            st['kept'] += 1
        txt = '\n'.join(out) + '\n'          # 문자열 완성 → 링크 제거 → 쓰기
        lb.unlink()
        lb.write_text(txt)
        st['files'] += 1
    if st['files']:                         # 시드 캐시가 옛 라벨 기준일 수 있다 → 지워서 다시 스캔 (검수 labelfix codex#2)
        for c in Path(mirror_root).rglob('labels.cache'):
            c.unlink()
            st['caches_removed'] = st.get('caches_removed', 0) + 1
    return st


def _split_tag() -> str:
    """미러 정체: 분할 출처(목록 sha 또는 Kaggle 원본) + 라벨 접두 제거 여부."""
    if SPLIT_LISTS is None:
        src = 'kaggle-original'
    else:
        h = hashlib.sha256()
        for k in ('train', 'val'):
            h.update(open(SPLIT_LISTS[k], 'rb').read())
        src = 'official-lists:' + h.hexdigest()[:16]
    return f'{src}|label_prefix_fix={bool(LABEL_PREFIX_FIX)}'


def find_data_yaml() -> str:
    """data.yaml 을 재귀로 찾는다. 경로를 박아두지 않는다.

    마운트 경로는 붙이는 방식에 따라 다르다 — 2026-09-01 실패로 확인:
      웹 UI 'Add Data'         → /kaggle/input/<slug>/...
      API dataset_sources      → /kaggle/input/datasets/<owner>/<slug>/...
      kagglehub 다운로드       → /root/.cache/kagglehub/datasets/<owner>/<slug>/...
    원본 kd_v7_5 는 웹 UI 경로만 박아두고 있어 API push 에서 죽었다.
    """
    # 빠른 경로(같은 컨테이너의 두 번째 프로세스가 미러를 다시 만들지 않게)는 분할 태그가 같을 때만 쓴다.
    # 태그 없이 재사용하면 직전 실행의 다른 분할 미러를 조용히 집는다 [실측 2026-10-23 split_smoke: 옛 분할이 게이트를 통과].
    tag = _split_tag()
    cands = ['/tmp/isaid_fixed.yaml']
    for root in ('/kaggle/input', '/root/.cache/kagglehub', '/content', *([DATA_ROOT] if DATA_ROOT else [])):
        if os.path.isdir(root):
            cands += sorted(glob.glob(f'{root}/**/data.yaml', recursive=True))
    for c in cands:
        if os.path.exists(c):
            with open(c) as f:
                cfg = yaml.safe_load(f)
            if c == '/tmp/isaid_fixed.yaml':
                _b = Path(cfg.get('path') or Path(c).parent)
                _ok = all((_b / str(cfg.get(k, ''))).is_dir() and any((_b / str(cfg.get(k, ''))).iterdir()) for k in ('train', 'val'))
                if cfg.get('nc') == 16 and cfg.get('split_tag') == tag and _ok:   # 미러가 지워졌으면 재생성 (검수 split rev2 codex#2)
                    print(f'[data] 캐시 yaml 재사용 (split_tag {tag})')
                    return c
                continue                     # 다른 분할·옛 형식 캐시 — 원본에서 다시 만든다
            cfg['nc'] = 16
            if cfg.get('path') and not Path(cfg['path']).exists():
                cfg['path'] = str(Path(c).parent)
            seed = _find_cache_seed()
            # 다른 분할로 만든 미러가 남아 있으면 build_mirror 가 그 위에 덧붙인다 [실측 split_smoke3: 24,538/12,753 타일].
            # MIRROR_DST 는 이 함수가 만든 링크·고친 라벨만 담는다 — 원본은 링크 너머라 지워지지 않는다.
            if Path(MIRROR_DST).exists() and not Path(MIRROR_DST).is_symlink():
                shutil.rmtree(MIRROR_DST)
            if SPLIT_LISTS is None:
                cfg = build_mirror(cfg, Path(c).parent, MIRROR_DST, seed_cache_dir=seed)
            else:
                cfg = build_list_mirror(cfg, Path(c).parent, MIRROR_DST, SPLIT_LISTS)
            if LABEL_PREFIX_FIX:
                LABEL_FIX_STATS.update(fix_label_prefix(MIRROR_DST))
                print(f'[data] 라벨 박스 접두 제거: {LABEL_FIX_STATS}')
            mirror_kind = '공식 목록' if SPLIT_LISTS is not None else '라벨 캐시'
            print(f'[data] {mirror_kind} 미러 {MIRROR_DST} (시드 캐시: {seed or "없음 — 이번 실행이 만든다"})')
            cfg['split_tag'] = tag
            blob = yaml.dump(cfg)
            with open('/tmp/isaid_fixed.yaml', 'w') as f:
                f.write(blob)
            print(f'[data] {c} → /tmp/isaid_fixed.yaml (nc=16)')
            return '/tmp/isaid_fixed.yaml'
    raise FileNotFoundError('data.yaml 없음 — Kaggle 데이터셋을 노트북에 추가할 것')


# ══════════════════════════════════════════════
# Teacher / RKD  (N1_MIXED 일 때만 쓰인다)
# ══════════════════════════════════════════════
def load_teacher():
    t = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)
    t.eval().cuda()
    for p in t.parameters():
        p.requires_grad_(False)
    return t


def teacher_cls(teacher, imgs_clean, block=11):
    feat = {}
    h = teacher.blocks[block].register_forward_hook(
        lambda m, i, o: feat.update({block: o[:, 0].detach()}))
    try:                       # teacher 는 저장 대상이 아니지만, 예외 시 hook 이 쌓이면 안 된다
        with torch.no_grad():
            teacher(F.interpolate(imgs_clean, size=(DINOV2_SIZE, DINOV2_SIZE),
                                  mode='bilinear', align_corners=False))
    finally:
        h.remove()
    return feat[block]


# ── SPPF hook ─────────────────────────────────────────────────
#  ⚠️ lambda 나 지역 함수로 hook 을 걸면 모듈에 붙어서 torch.save 때 pickle 이 터진다:
#       AttributeError: Can't get local object '....<locals>.<lambda>'
#     (2026-09-02 v5 실패. 1에폭을 다 돌고 체크포인트 저장에서 사망)
#  그래서 ①모듈 레벨 함수로 만들고 ②저장 직전에는 아예 떼었다가 다시 붙인다.
_FEAT: dict = {}

# KD 가 실제로 손실에 더해졌는지 세는 카운터.
#   self.model.loss 를 덮어쓰는 방식은 ultralytics 가 다른 경로로 손실을 부르면
#   **에러 없이 그냥 호출되지 않는다.** 그러면 N1_MIXED 가 A_NOISY 와 같아지고
#   실험 전체가 무의미해지는데 로그만 보면 정상으로 보인다.
#   (사전검수 kimi/grok 이 'silent failure' 로 지목한 지점)
_KD: dict = {'calls': 0, 'applied': 0, 'sum': 0.0}


def _strip_hooks(model):
    """모델 전체에서 forward hook 을 떼고 (module, hooks) 목록을 돌려준다.

    저장 대상은 self.model 이 아니라 **EMA 사본**이다:
        _setup_train:350  _build_train_pipeline()      ← 여기서 hook 등록
        _setup_train:352  self.ema = ModelEMA(self.model)  ← hook 붙은 채로 deepcopy
    그래서 self.model 에서만 떼면 EMA 에 남은 hook 이 그대로 pickle 된다.
    """
    saved = []
    for m in model.modules():
        if m._forward_hooks:
            saved.append((m, dict(m._forward_hooks)))
            m._forward_hooks.clear()
    return saved


def _sppf_hook(module, inp, out):
    _FEAT['s'] = out.mean(dim=[-1, -2])          # GAP → (B, C)


def rkd_loss(s, t):
    s, t = F.normalize(s, dim=-1), F.normalize(t, dim=-1)
    sd, td = torch.cdist(s, s), torch.cdist(t, t)
    return F.smooth_l1_loss(sd / (sd.mean() + 1e-8), td / (td.mean() + 1e-8))


# ══════════════════════════════════════════════
# 트레이너 — A_NOISY 와 N1_MIXED 가 공유한다
# ══════════════════════════════════════════════
class SharedNoiseTrainer(SegmentationTrainer):
    """노이즈 주입은 항상. KD 손실은 USE_KD 일 때만.

    두 조건의 차이가 KD 항 하나뿐이어야 (N1_MIXED − A_NOISY) 가 KD 기여가 된다.
    """

    def build_optimizer(self, *a, **kw):
        """adapter 파라미터를 **scheduler 가 만들어지기 전에** optimizer 에 넣는다.

        ⚠️ 2026-09-02 실패로 배운 것:
           _build_train_pipeline():284  self.optimizer = self.build_optimizer(...)
           _build_train_pipeline():292  self._setup_scheduler()   ← 여기서 그룹 수가 고정
           _setup_train():350           self._build_train_pipeline()
           _setup_train() 이 끝난 뒤에 add_param_group 을 하면 scheduler 는 이미
           3개 그룹 기준으로 만들어져 있어, 첫 epoch 끝 scheduler.step() 에서
           `zip(param_groups, values, strict=True)` 가 4 vs 3 으로 터진다.
             ValueError: zip() argument 2 is shorter than argument 1
           그래서 add_param_group 은 반드시 이 메서드 안에서 해야 한다.
           (Gemini Flash 검수가 'build_optimizer 를 오버라이드하라'고 정확히 지적했는데,
            근거로 댄 'optimizer 가 아직 없다' 가 틀렸다는 이유로 처방까지 기각했다 —
            이유가 틀려도 처방은 맞을 수 있다.)
        """
        opt = super().build_optimizer(*a, **kw)

        sppf = self.model.model[9]
        assert type(sppf).__name__ == 'SPPF', f'model[9] 가 SPPF 가 아니다: {type(sppf).__name__}'
        ch = sppf.cv2.conv.out_channels
        self._sppf = sppf
        self._hook = sppf.register_forward_hook(_sppf_hook)   # 모듈 레벨 함수 (pickle 가능)

        # adapter 는 두 조건 모두 만들고 등록한다 — param_group 구성을 대칭으로 (검수 4/4)
        self._adapter = nn.Sequential(
            nn.Linear(ch, 256, bias=False), nn.LayerNorm(256),
            nn.ReLU(inplace=True), nn.Linear(256, 768, bias=False),
        ).cuda()
        opt.add_param_group({'params': list(self._adapter.parameters()), 'lr': self.args.lr0})
        self._sppf_ch = ch
        print(f'[trainer] adapter 등록 완료 (scheduler 생성 전). sppf_ch={ch}, groups={len(opt.param_groups)}')
        return opt

    def _setup_train(self, *a, **kw):
        # teacher 는 super() 뒤에 붙인다. student feature 는 모듈 전역 _FEAT 로 받는다.
        self._t_feat = None
        self._teacher = None
        _FEAT.clear()          # 같은 프로세스에서 조건을 연달아 돌리므로 매번 비운다
        _KD.update({'calls': 0, 'applied': 0, 'sum': 0.0})

        super()._setup_train(*a, **kw)   # ← 이 안에서 build_optimizer + _setup_scheduler 가 돈다

        assert len(self.optimizer.param_groups) == len(self.scheduler.base_lrs), (
            f'그룹 {len(self.optimizer.param_groups)} vs scheduler {len(self.scheduler.base_lrs)} '
            '— scheduler 가 adapter 그룹을 모른다')

        if not USE_KD:
            print(f'[trainer] {EXP} — KD 항 없음 '
                  f'(노이즈={"on" if USE_NOISE else "off"}, sppf_ch={self._sppf_ch})')
            return

        self._teacher = load_teacher()
        orig = self.model.loss
        tr = self

        def kd_wrapped(batch, preds=None):
            loss, items = orig(batch, preds)
            _KD['calls'] += 1
            s = _FEAT.get('s')
            if s is not None and tr._t_feat is not None:
                kd = rkd_loss(tr._adapter(s), tr._t_feat)
                loss = loss + LAM_KD * kd
                _KD['applied'] += 1
                _KD['sum'] += float(kd.detach())
            return loss, items

        self.model.loss = kd_wrapped
        print(f'[trainer] N1_MIXED — KD on (sppf_ch={self._sppf_ch}, lam={LAM_KD})')

    def save_model(self):
        """체크포인트에 forward hook 이 딸려가지 않게 self.model 과 EMA 양쪽에서 뗀다.

        hook 이 붙은 채로 torch.save 하면 pickle 이 hook 함수를 직렬화하려 들고,
        지역 함수/lambda 면 즉시 터진다. hook 을 모듈 레벨 함수로 만든 것이 1차 방어,
        저장 동안 아예 빼는 것이 2차 방어다.
        best.pt 를 나중에 YOLO() 로 로드해 평가하므로 체크포인트는 깨끗해야 한다.
        """
        targets = [self.model]
        ema = getattr(getattr(self, 'ema', None), 'ema', None)
        if ema is not None:
            targets.append(ema)
        saved = []
        for mdl in targets:
            saved += _strip_hooks(mdl)
        try:
            return super().save_model()
        finally:
            for m, hooks in saved:
                m._forward_hooks.update(hooks)

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)

        # ── 노이즈: 배치 '안'에서 MIXED_RATIO 만큼의 이미지에만 ──
        #    ⚠️ 이 블록은 USE_KD 와 무관하게 항상 돈다. 원본 kd_v7_5 는
        #       teacher 유무로 감싸고 있어 A_NOISY 에서 노이즈가 꺼졌다.
        clean = batch['img'].clone()
        if USE_NOISE:
            B = clean.shape[0]
            n_noisy = max(1, int(B * MIXED_RATIO))
            idx = torch.randperm(B, device=clean.device)[:n_noisy]
            noise = torch.randn(n_noisy, *clean.shape[1:], device=clean.device) * NOISE_SIGMA
            img = batch['img'].clone()
            img[idx] = torch.clamp(img[idx] + noise, 0, 1)
            batch['img'] = img

        # Teacher 는 언제나 clean 을 본다
        if self._teacher is not None:
            with torch.no_grad():
                self._t_feat = teacher_cls(self._teacher, clean, block=11)
        return batch



# ══════════════════════════════════════════════
# Spatial KD v2.1 — isolated from the four legacy execution paths
# ══════════════════════════════════════════════
_SPATIAL_CACHE = {}
_SPATIAL_RUN = {}


def normalize_spatial(x):
    """Fixed, per-position LN -> L2, in fp32 (no learnable teacher parameters)."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = x.float().permute(0, 2, 3, 1)
        x = F.layer_norm(x, (x.shape[-1],), eps=1e-6)
        return F.normalize(x, dim=-1, eps=1e-6).permute(0, 3, 1, 2)


class SpatialAdapter(nn.Module):
    def __init__(self, interp_mode=INTERP_MODE):
        super().__init__()
        if interp_mode not in ('bilinear_antialias', 'area'):
            raise ValueError(f'unsupported interpolation: {interp_mode}')
        self.interp_mode = interp_mode
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.gn = nn.GroupNorm(32, 256)
        self.act = nn.GELU()
        self.conv1 = nn.Conv2d(256, 768, 1, bias=True)
        assert sum(p.numel() for p in self.parameters()) == 787712

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (256, 40, 40):
            raise RuntimeError(f'P4 must be Bx256x40x40, got {tuple(x.shape)}')
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = self.conv1(self.act(self.gn(self.conv3(x.float()))))
            if self.interp_mode == 'area':
                x = F.interpolate(x, size=(32, 32), mode='area')
            else:
                # No per-run fallback: choose area explicitly for ALL runs after smoke.
                x = F.interpolate(x, size=(32, 32), mode='bilinear',
                                  align_corners=False, antialias=True)
            return normalize_spatial(x)


def prepare_teacher_input(clean, teacher_norm):
    with torch.autocast(device_type=clean.device.type, enabled=False):
        x = F.interpolate(clean.float(), size=(448, 448), mode='bilinear',
                          align_corners=False)
        if teacher_norm == 'imagenet':
            mean = x.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
            std = x.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
            x = (x - mean) / std
        elif teacher_norm != 'none':
            raise ValueError(f'unknown teacher normalization: {teacher_norm}')
        return x


def tokens_to_grid(tokens):
    if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (1025, 768):
        raise RuntimeError(f'block 11 must output Bx1025x768, got {tuple(tokens.shape)}')
    return tokens[:, 1:].transpose(1, 2).reshape(-1, 768, 32, 32).contiguous()


def check_grid_order():
    # Row-major coordinate ramp, including an asymmetric landmark, independent of ViT weights.
    tokens = torch.zeros(1, 1025, 768)
    rows, cols = torch.meshgrid(torch.arange(32), torch.arange(32), indexing='ij')
    tokens[0, 1:, 0] = rows.flatten()
    tokens[0, 1:, 1] = cols.flatten()
    tokens[0, 1 + 3 * 32 + 19, 2] = 123
    grid = tokens_to_grid(tokens)
    if not (torch.equal(grid[0, 0], rows) and torch.equal(grid[0, 1], cols)
            and grid[0, 2, 3, 19] == 123):
        raise RuntimeError('teacher patch grid order failed')


def check_teacher_patch_order(teacher, device):
    """Check the actual cached ViT patch embed, as well as tokens_to_grid."""
    check_grid_order()
    embed = teacher.patch_embed
    rows, cols = torch.meshgrid(torch.arange(448, device=device),
                                torch.arange(448, device=device), indexing='ij')
    ramp = torch.stack((rows / 448, cols / 448, torch.zeros_like(rows))).float()[None]
    ramp[:, 2, 3*14:4*14, 19*14:20*14] = 1
    captured = []
    handle = embed.proj.register_forward_hook(lambda m, i, o: captured.append(o))
    try:
        with torch.no_grad(), torch.autocast(device_type=ramp.device.type, enabled=False):
            actual = embed(ramp)
            if len(captured) != 1 or tuple(captured[0].shape) != (1, 768, 32, 32):
                raise RuntimeError('teacher patch projection grid must be 32x32')
            expected = embed.norm(captured[0].flatten(2).transpose(1, 2))
            if actual.shape != (1, 1024, 768) or not torch.equal(actual, expected):
                raise RuntimeError('teacher patch embedding is not row-major')
    finally:
        handle.remove()


def raw_teacher_patches(teacher, clean, teacher_norm='imagenet', micro_batch=2):
    if len(teacher.blocks) != 12:
        raise RuntimeError('dinov2_vitb14 must have exactly 12 blocks, no registers')
    if micro_batch < 1 or clean.shape[0] < 1:
        raise ValueError('empty batch or invalid micro-batch')
    check_grid_order()
    captured = []

    def capture(module, inputs, output):
        captured.append(tokens_to_grid(output.detach().float()))

    handle = teacher.blocks[11].register_forward_hook(capture)
    chunks = []
    try:
        teacher.eval().requires_grad_(False)
        with torch.no_grad(), torch.autocast(device_type=clean.device.type, enabled=False):
            for imgs in clean.split(micro_batch):
                captured.clear()
                teacher(prepare_teacher_input(imgs, teacher_norm))
                if len(captured) != 1 or captured[0].shape[0] != imgs.shape[0]:
                    raise RuntimeError('block 11 did not produce the current micro-batch exactly once')
                chunks.append(captured[0])
    finally:
        handle.remove()
    return torch.cat(chunks, dim=0)


def anomaly_mask(raw, mask_top=MASK_TOP):
    if tuple(raw.shape[1:]) != (768, 32, 32) or mask_top != 0.01:
        raise ValueError('v2.1 requires 1024 positions and MASK_TOP=0.01')
    # floor(1024 * .01) == 10, not ceil == 11. Stable ties are deterministic.
    norms = torch.linalg.vector_norm(raw.float(), dim=1).flatten(1)
    excluded = torch.argsort(norms, dim=1, descending=True, stable=True)[:, :10]
    mask = torch.ones_like(norms, dtype=torch.bool)
    mask.scatter_(1, excluded, False)
    return mask.reshape(-1, 32, 32)


def teacher_patches(teacher, clean, teacher_norm=TEACHER_NORM, micro_batch=2):
    raw = raw_teacher_patches(teacher, clean, teacher_norm, micro_batch)
    # S1r still uses unnormalized teacher INPUT for the target. Use a canonical
    # ImageNet forward for mask only, so S1/S2/S1r supervise identical positions.
    mask_raw = (raw if teacher_norm == 'imagenet' else
                raw_teacher_patches(teacher, clean, 'imagenet', micro_batch))
    return raw, anomaly_mask(mask_raw)


def load_spatial_teacher(device):
    # Offline-first: never ask torch.hub to contact GitHub when cache is present.
    hub = Path(torch.hub.get_dir())
    repos = sorted(hub.glob('facebookresearch_dinov2_*'))
    checkpoint = hub / 'checkpoints' / 'dinov2_vitb14_pretrain.pth'
    if not repos or not checkpoint.is_file():
        # Kaggle 새 세션: 캐시가 없다. 인터넷으로 1회 받아 캐시를 채운 뒤 아래 로컬 경로로 다시 로드한다.
        print('[spatial] DINOv2 cache missing -> torch.hub download (needs internet)')
        torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)
        repos = sorted(hub.glob('facebookresearch_dinov2_*'))
        if not repos or not checkpoint.is_file():
            raise FileNotFoundError('DINOv2 download did not populate the torch hub cache')
    teacher = torch.hub.load(str(repos[0]), 'dinov2_vitb14', source='local', pretrained=False)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    teacher.load_state_dict(state, strict=True)
    teacher = teacher.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    check_teacher_patch_order(teacher, device)
    return teacher


def shuffle_teacher_tokens(target, mask, generator):
    if tuple(mask.shape) != (target.shape[0], *target.shape[2:]):
        raise ValueError('shuffle mask shape mismatch')
    flat = target.flatten(2)
    shuffled = flat.clone()
    for b in range(target.shape[0]):
        valid = torch.nonzero(mask[b].flatten(), as_tuple=False).flatten()
        # Dedicated CPU generator: consumption is independent of data/noise RNG,
        # and the same permutation sequence is portable between CPU/CUDA.
        perm = torch.randperm(valid.numel(), generator=generator, device='cpu').to(target.device)
        shuffled[b, :, valid] = flat[b, :, valid[perm]]
    return shuffled.reshape_as(target)


def spatial_kd_loss(student, teacher, mask, noisy_mask=None, return_stats=False):
    if student.shape != teacher.shape or tuple(mask.shape) != (student.shape[0], *student.shape[2:]):
        raise ValueError('spatial loss shape mismatch')
    if not bool(mask.any()) or not bool(mask.flatten(1).any(1).all()):
        raise ValueError('each image needs supervised positions')
    with torch.autocast(device_type=student.device.type, enabled=False):
        s, t = normalize_spatial(student), normalize_spatial(teacher.detach())
        distance = 1 - (s * t).sum(dim=1)
        loss = distance.masked_select(mask.bool()).mean()
        if not return_stats:
            return loss
        if noisy_mask is None:
            raise ValueError('noisy_mask is required for group logs')
        stats = {}
        for name, select in (('clean', ~noisy_mask.bool()), ('noisy', noisy_mask.bool())):
            selected = mask.bool() & select[:, None, None]
            stats[name] = float(distance.detach()[selected].mean()) if bool(selected.any()) else None
        stats['student_spatial_variance'] = float(s.detach().flatten(2).var(dim=2, unbiased=False).mean())
        stats['teacher_spatial_variance'] = float(t.detach().flatten(2).var(dim=2, unbiased=False).mean())
        return loss, stats


# ── 붕괴 진단·손실 변형 (2026-10-23) — 작업대 T-0124 workbench/kd_collapse_diag.py 를 그대로 옮긴 것 (커밋 7848098) ──
def _validate(x, mask, other=None):
    if not isinstance(x, torch.Tensor) or x.ndim != 4:
        raise ValueError("features must have shape B x C x H x W")
    if not x.is_floating_point() or any(size == 0 for size in x.shape):
        raise ValueError("features must be nonempty floating point tensors")
    if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
            or mask.shape != (x.shape[0], *x.shape[2:])
            or mask.device != x.device):
        raise ValueError("mask must be a B x H x W boolean tensor on the feature device")
    if not mask.flatten(1).any(dim=1).all().item():
        raise ValueError("each image must contain a valid cell")
    if other is not None:
        if (not isinstance(other, torch.Tensor) or other.shape != x.shape
                or other.device != x.device or not other.is_floating_point()):
            raise ValueError("student and teacher must have matching feature shapes and devices")


def _clean(x, mask):
    return torch.where(mask.unsqueeze(1), x, 0)


def masked_mean(x, mask):
    """Return each image's mean feature over its valid spatial cells."""
    _validate(x, mask)
    counts = mask.sum(dim=(1, 2)).unsqueeze(1)
    return _clean(x, mask).sum(dim=(2, 3)) / counts


def _center(x, mask):
    mean = masked_mean(x, mask)
    return _clean(_clean(x, mask) - mean[:, :, None, None], mask)


def spatial_energy(x, mask):
    """Return the mean squared distance from each image's masked mean."""
    centered = _center(x, mask)
    return centered.square().sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2))


def constant_floor(t, mask):
    """Return the cell-weighted constant-direction loss floor as a float."""
    means = masked_mean(t, mask)
    counts = mask.sum(dim=(1, 2))
    return (1 - (counts * means.norm(dim=1)).sum() / counts.sum()).item()


def centered_spatial_loss(s, t, mask, eps=1e-6):
    """Compare normalized, independently centered spatial features."""
    _validate(s, mask, t)
    s_center = F.normalize(_center(s, mask), dim=1, eps=eps)
    t_center = F.normalize(_center(t, mask), dim=1, eps=eps)
    cosine = (s_center * t_center).sum(dim=1)
    return (1 - cosine[mask]).mean()


def variance_floor_loss(s, t, mask, gamma=0.5):
    """Penalize student energy below a fraction of detached teacher energy."""
    _validate(s, mask, t)
    return F.relu(gamma * spatial_energy(t, mask).detach()
                  - spatial_energy(s, mask)).mean()


def _sampled_centered(s, t, mask, k, generator):
    """Return per-image centered/L2 token samples using the run-local RNG."""
    _validate(s, mask, t)
    if k < 2:
        raise ValueError('sample size must be at least two')
    s = F.normalize(_center(s.float(), mask), dim=1, eps=1e-6)
    t = F.normalize(_center(t.float(), mask), dim=1, eps=1e-6)
    pairs = []
    for sb, tb, valid in zip(s, t, mask):
        positions = torch.nonzero(valid.flatten(), as_tuple=False).flatten()
        count = min(k, positions.numel())
        chosen = positions if count == positions.numel() else positions[
            torch.randperm(positions.numel(), generator=generator, device='cpu')[:count].to(positions.device)]
        pairs.append((sb.flatten(1)[:, chosen], tb.flatten(1)[:, chosen]))
    return pairs


def relation_spatial_loss(s, t, mask, generator, k=RELATION_K):
    """Align off-diagonal spatial self-similarities after per-image centering."""
    with torch.autocast(device_type=s.device.type, enabled=False):
        losses = []
        for sc, tc in _sampled_centered(s, t, mask, k, generator):
            diff = (sc.T @ sc - tc.T @ tc).square()
            losses.append(diff[~torch.eye(diff.shape[0], dtype=torch.bool, device=diff.device)].mean())
        return torch.stack(losses).mean()


def infonce_spatial_loss(s, t, mask, generator, k=INFONCE_K, tau=INFONCE_TAU):
    """Make each centered student token select its corresponding teacher token."""
    if tau <= 0:
        raise ValueError('InfoNCE temperature must be positive')
    with torch.autocast(device_type=s.device.type, enabled=False):
        losses = []
        for sc, tc in _sampled_centered(s, t, mask, k, generator):
            labels = torch.arange(sc.shape[1], device=sc.device)
            losses.append(F.cross_entropy(sc.T @ tc / tau, labels))
        return torch.stack(losses).mean()


def foreground_weights(batch, batch_size, device, height=32, width=32, alpha=FG_ALPHA):
    """Map normalized xywh GT boxes onto grid cells they overlap, including tiny boxes."""
    weights = torch.ones(batch_size, height, width, device=device)
    boxes, batch_idx = batch.get('bboxes'), batch.get('batch_idx')
    if boxes is None or batch_idx is None or boxes.numel() == 0:
        return weights
    boxes, batch_idx = boxes.to(device=device), batch_idx.to(device=device).long().flatten()
    for box, image in zip(boxes, batch_idx):
        if image < 0 or image >= batch_size:
            continue
        cx, cy, bw, bh = box
        left = int(torch.floor((cx - bw / 2) * width).clamp(0, width - 1).item())
        right = int(torch.ceil((cx + bw / 2) * width).clamp(1, width).item())
        top = int(torch.floor((cy - bh / 2) * height).clamp(0, height - 1).item())
        bottom = int(torch.ceil((cy + bh / 2) * height).clamp(1, height).item())
        weights[image, top:bottom, left:right] = alpha
    return weights


def fgcenter_spatial_loss(s, t, mask, weights, eps=1e-6):
    """Apply foreground weights to the centered cosine distance."""
    _validate(s, mask, t)
    if weights.shape != mask.shape or weights.device != s.device:
        raise ValueError('foreground weights must match the spatial mask')
    with torch.autocast(device_type=s.device.type, enabled=False):
        sc = F.normalize(_center(s.float(), mask), dim=1, eps=eps)
        tc = F.normalize(_center(t.float(), mask), dim=1, eps=eps)
        distance = 1 - (sc * tc).sum(dim=1)
        valid_weight = weights.float() * mask
        return (valid_weight * distance).sum() / valid_weight.sum()


@torch.no_grad()
def correspondence_stats(s, t, mask):
    """Return cell-pooled correspondence and image-averaged energy metrics."""
    _validate(s, mask, t)
    correct = 0
    total = int(mask.sum().item())
    plain_sum = 0.0
    for sb, tb, valid in zip(s, t, mask):
        # Boolean indexing preserves row-major order, including for sparse masks.
        sv = F.normalize(sb[:, valid].T, dim=1, eps=1e-6)
        tv = F.normalize(tb[:, valid].T, dim=1, eps=1e-6)
        # Permit distinct floating dtypes without changing differentiable APIs.
        dtype = torch.promote_types(sv.dtype, tv.dtype)
        sv, tv = sv.to(dtype), tv.to(dtype)
        n = sv.shape[0]
        plain_sum += (sv * tv).sum(dim=1).sum().item()
        chunk = max(1, min(n, 1_000_000 // n))
        for start in range(0, n, chunk):
            scores = sv[start:start + chunk] @ tv.T
            nearest = scores.argmax(dim=1)  # First maximum breaks ties.
            expected = torch.arange(start, start + len(nearest), device=s.device)
            correct += (nearest == expected).sum().item()
    return {
        "p_at_1": float(correct / total),
        "p_at_1_chance": float(s.shape[0] / total),
        "plain_cos": float(plain_sum / total),
        "centered_cos": float(1 - centered_spatial_loss(s, t, mask).item()),
        "E_s": float(spatial_energy(s, mask).mean().item()),
        "E_t": float(spatial_energy(t, mask).mean().item()),
        "L_const": constant_floor(t, mask),
    }
# ── T-0124 끝 ──


def add_spatial_loss(detection_loss, kd, batch_size, lam=LAM_SP, reduction=DET_REDUCTION):
    """Collapse components FIRST: KD is added once to the backpropagated scalar."""
    if batch_size < 1 or reduction not in ('batch_sum', 'image_mean'):
        raise ValueError('invalid loss reduction contract')
    det = detection_loss.sum()
    scale = batch_size if reduction == 'batch_sum' else 1
    d_img = det / scale
    # Exact equality for S0, including bitwise scalar value and detection gradient.
    total = det if lam == 0 else det + lam * kd * scale
    return total, d_img


def smoke_lambda_recommendation(ratios, lam):
    if not ratios or not all(np.isfinite(x) and x >= 0 for x in ratios):
        raise ValueError('finite first-epoch ratios required')
    mean_r = float(np.mean(ratios))
    return {'mean_R': mean_r, 'lam_sp_used': lam,
            'lam_sp_recommended': lam * 0.2 / mean_r if mean_r > 0.5 else lam,
            'recalibration_required': mean_r > 0.5,
            'rule': 'S1 seed 0 smoke epoch 0 only; lock before main runs; no mAP selection'}


def lam_schedule_factor(step, total_steps):
    """Return the effective-lambda multiplier for a one-based training step."""
    if total_steps <= 0:
        raise ValueError('total training steps must be positive')
    if LAM_SCHEDULE == 'const':
        return 1.0
    warmup = LAM_WARMUP_FRAC * total_steps
    if warmup and step <= warmup:
        return max(0.0, min(1.0, step / warmup))
    if LAM_SCHEDULE == 'warmup':
        return 1.0
    # No warmup leaves the whole run for the cosine; a full-length warmup has
    # already reached its endpoint above.
    span = total_steps - warmup
    progress = 1.0 if span <= 0 else max(0.0, min(1.0, (step - warmup) / span))
    return LAM_END_FRAC + (1 - LAM_END_FRAC) * (1 + np.cos(np.pi * progress)) / 2


def _lam_auto_step(trainer, ratio, step):
    """Collect the fixed-lambda warm-up ratios, then recalibrate once."""
    if LAM_AUTO_R is None or LAM_SP == 0:
        return None
    if step <= LAM_AUTO_STEPS:
        if ratio is not None:
            trainer._lam_auto_ratios.append(ratio)
        return None
    if step != LAM_AUTO_STEPS + 1 or trainer._lam_auto_done:
        return None
    mean_r = float(np.mean(trainer._lam_auto_ratios))
    if not np.isfinite(mean_r) or mean_r <= 0:
        raise RuntimeError('LAM_AUTO_R requires finite positive warm-up R')
    before = trainer._lam_sp
    trainer._lam_sp = LAM_SP * LAM_AUTO_R / mean_r
    trainer._lam_auto_done = True
    report = {'target': LAM_AUTO_R, 'measured_R': mean_r, 'lam_before': before,
              'lam_after': trainer._lam_sp, 'step': step}
    _SPATIAL_RUN['lam_auto'] = report
    return report


def _p4_hook(module, inputs, output):
    # EMA/validation are eval-mode; never overwrite a training graph with their output.
    if not module.training or not _SPATIAL_CACHE.get('active', False):
        return
    if module is not _SPATIAL_CACHE.get('p4_module'):
        return  # even a mistakenly train-mode EMA cannot overwrite the student cache
    if output.ndim != 4 or tuple(output.shape[1:]) != (256, 40, 40):
        raise RuntimeError(f'model[18] must output Bx256x40x40, got {tuple(output.shape)}')
    if output.shape[0] != _SPATIAL_CACHE['batch_size']:
        raise RuntimeError('P4 batch size mismatch')
    _SPATIAL_CACHE['student'] = output
    _SPATIAL_CACHE['feature_batch_id'] = _SPATIAL_CACHE['batch_id']


def _gradient_norm(grads):
    values = [g.detach().float().square().sum() for g in grads if g is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def _strip_training_state(targets):
    hooks, losses = [], []
    for model in targets:
        hooks.extend(_strip_hooks(model))
        if 'loss' in model.__dict__:
            losses.append((model, model.__dict__.pop('loss')))
    return hooks, losses


def _restore_training_state(state):
    hooks, losses = state
    for module, saved in hooks:
        module._forward_hooks.update(saved)
    for model, loss in losses:
        model.loss = loss


def _install_spatial_loss(trainer):
    original = trainer.model.loss

    def kd_wrapped(batch, preds=None):
        if not trainer.model.training:
            return original(batch, preds)
        _KD['calls'] += 1
        cache = _SPATIAL_CACHE
        if (not cache.get('active') or batch.get('_spatial_batch_id') != cache.get('batch_id')
                or batch['img'] is not cache.get('image')):
            raise RuntimeError('missing/stale spatial batch cache')
        if preds is None:
            cache.pop('student', None)
            cache.pop('feature_batch_id', None)
        detection, items = original(batch, preds)
        if not _SPATIAL_RUN.get('reduction_verified'):
            B0 = batch['img'].shape[0]
            want = items.sum() * (B0 if DET_REDUCTION == 'batch_sum' else 1)
            if not torch.allclose(detection.detach().sum().float(), want.float(), rtol=1e-4, atol=1e-6):
                raise RuntimeError(f'DET_REDUCTION={DET_REDUCTION} contradicts ultralytics loss: '
                                   f'{float(detection.sum())} vs {float(want)}')
            _SPATIAL_RUN['reduction_verified'] = True
        if cache.get('feature_batch_id') != cache['batch_id'] or 'student' not in cache:
            raise RuntimeError('current training P4 feature missing (validation/EMA cache forbidden)')
        with torch.autocast(device_type=cache['student'].device.type, enabled=False):
            prealign = cache['batch_id'] <= ADAPTER_PREALIGN_STEPS
            student = trainer._adapter(cache['student'].detach() if prealign else cache['student'])
            target = cache['teacher']
            if SHUFFLE:
                target = shuffle_teacher_tokens(target, cache['mask'], trainer._shuffle_generator)
            kd, stats = spatial_kd_loss(student, target, cache['mask'], cache['noisy'], True)
            stats['L_plain'] = float(kd.detach())
            if KD_LOSS_VARIANT == 'center':
                kd = centered_spatial_loss(student, target, cache['mask'])
            elif KD_LOSS_VARIANT == 'varfloor':
                kd = kd + VAR_FLOOR_W * variance_floor_loss(student, target, cache['mask'], VAR_FLOOR_GAMMA)
            elif KD_LOSS_VARIANT == 'relation':
                kd = relation_spatial_loss(student, target, cache['mask'], trainer._relation_generator)
            elif KD_LOSS_VARIANT == 'infonce':
                kd = infonce_spatial_loss(student, target, cache['mask'], trainer._relation_generator)
            elif KD_LOSS_VARIANT == 'fgcenter':
                kd = fgcenter_spatial_loss(student, target, cache['mask'], cache['fg_weight'])
            # 대응 지표는 섞기 전 교사(cache['teacher']) 기준 — S2 도 원래 위치와 비교한다
            # 스모크(FRACTION<1)는 매 스텝, 본실행은 DIAG_EVERY 스텝마다만 (GPU 동기화 비용 — 검수 impl_r1 codex#2)
            if FRACTION < 1 or cache['batch_id'] == 1 or cache['batch_id'] % DIAG_EVERY == 0:
                stats.update(correspondence_stats(student.detach(), cache['teacher'].detach(), cache['mask']))
            stats['loss_variant'] = KD_LOSS_VARIANT
            lam_factor = lam_schedule_factor(cache['batch_id'], getattr(trainer, '_lam_total_steps', 1))
            if LAM_AUTO_R is None and LAM_SCHEDULE == 'const':
                # Keep the preregistered default computation byte-for-byte unchanged.
                lam, lam_auto = LAM_SP, None
                total, d_img = add_spatial_loss(detection, kd, batch['img'].shape[0])
            else:
                scale = batch['img'].shape[0] if DET_REDUCTION == 'batch_sum' else 1
                d_img = detection.sum() / scale
                lam_base = trainer._lam_sp if LAM_AUTO_R is not None else LAM_SP
                if LAM_AUTO_R is not None:
                    # 계수 0 스텝은 R 측정에서만 빼고, 보정 시점(51번째)은 놓치지 않는다 (검수 rev2 codex)
                    base_r = (float((lam_base * kd.detach() / d_img.detach()).item())
                              if lam_factor > 0 else None)
                    lam_auto = _lam_auto_step(trainer, base_r, cache['batch_id'])
                    lam_base = trainer._lam_sp
                else:
                    lam_auto = None
                lam = lam_base * lam_factor
                total, d_img = add_spatial_loss(detection, kd, batch['img'].shape[0], lam)
        # autograd.grad leaves optimizer .grad buffers untouched: KD-only diagnostics.
        # 매 스텝 autograd.grad 는 백본 역전파를 한 번 더 해서 학습이 느려진다 → 첫 스텝과 DIAG_EVERY 마다만.
        diag = (cache['batch_id'] == 1) or (cache['batch_id'] % DIAG_EVERY == 0)
        if diag:
            adapter_params = tuple(trainer._adapter.parameters())
            upstream = tuple(p for p in trainer.model.model[:KD_LAYER].parameters() if p.requires_grad)
            grads = torch.autograd.grad(kd, adapter_params + upstream, retain_graph=True, allow_unused=True)
            g_ad, g_up = _gradient_norm(grads[:len(adapter_params)]), _gradient_norm(grads[len(adapter_params):])
        else:
            g_ad = g_up = None
        stats.update({
            'adapter_kd_grad_norm': g_ad,
            'upstream_kd_grad_norm': g_up,
            'batch_id': cache['batch_id'], 'epoch': int(trainer.epoch),
            'batch_size': batch['img'].shape[0], 'noisy_count': int(cache['noisy'].sum()),
            'L_sp': float(kd.detach()), 'L_det': float(detection.detach().sum()),
            'D_img': float(d_img.detach()), 'lam_sp': lam, 'lam_eff': lam,
            'lam_factor': lam_factor, 'prealign': prealign,
            'adapter_initial_lr': _SPATIAL_RUN.get('adapter_initial_lr'),
            'adapter_weight_decay': _SPATIAL_RUN.get('adapter_weight_decay'),
            'R': float((lam * kd.detach() / d_img.detach()).item()),
            'det_return_shape': list(detection.shape), 'det_reduction': DET_REDUCTION,
            'optimizer_group_lrs': [float(g['lr']) for g in trainer.optimizer.param_groups],
        })
        if lam_auto is not None:
            stats['lam_auto'] = lam_auto
        keys = ('L_sp', 'L_det', 'D_img', 'R') + (('adapter_kd_grad_norm', 'upstream_kd_grad_norm') if diag else ())
        keys += tuple(k for k in ('L_plain', 'p_at_1', 'plain_cos', 'centered_cos', 'E_s', 'E_t', 'L_const') if k in stats)  # impl_r1 codex#1
        if not all(np.isfinite(stats[k]) for k in keys):
            raise RuntimeError('nonfinite spatial loss/gradient diagnostics')
        if diag and (stats['adapter_kd_grad_norm'] <= 0
                     or (not prealign and stats['upstream_kd_grad_norm'] <= 0)):
            raise RuntimeError('KD does not reach adapter and upstream student')
        log_path = Path(trainer.save_dir) / 'spatial_kd.jsonl'
        with log_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(stats, allow_nan=False) + '\n')
        if EXP == 'S1' and SEED == 0 and FRACTION < 1 and trainer.epoch == 0 and LAM_AUTO_R is None:
            trainer._smoke_ratios.append(stats['R'])
        _SPATIAL_RUN['optimizer_group_lrs'] = stats['optimizer_group_lrs']
        _KD['applied'] += 1
        _KD['sum'] += stats['L_sp']
        cache.clear()  # no second application, stale graph, or validation reuse
        return total, items

    trainer.model.loss = kd_wrapped


def _spatial_epoch_end(trainer):
    # 자동 보정 실행은 R 이 보정 전후로 섞이므로 스모크 λ 추천을 내지 않는다 (검수 rev1 agy LOW-1)
    if EXP == 'S1' and SEED == 0 and FRACTION < 1 and trainer.epoch == 0 and LAM_AUTO_R is None:
        report = smoke_lambda_recommendation(trainer._smoke_ratios, LAM_SP)
        _SPATIAL_RUN['lambda_smoke'] = report
        (Path(trainer.save_dir) / 'lambda_smoke.json').write_text(
            json.dumps(report, indent=2), encoding='utf-8')
        print(f'[spatial lambda smoke] {report}')


class SpatialNoiseTrainer(SegmentationTrainer):
    """All S conditions share preprocessing/teacher/adapter; legacy trainer is untouched."""
    def build_optimizer(self, *args, **kwargs):
        if getattr(self, '_spatial_optimizer_built', False):
            raise RuntimeError('optimizer rebuild/resume/OOM batch reduction forbidden for spatial runs')
        p4 = self.model.model[KD_LAYER]
        if type(p4).__name__ != 'C2f' or p4.cv2.conv.out_channels != 256:
            raise RuntimeError('model[18] must be C2f with 256 output channels; no layer auto-selection')
        optimizer = super().build_optimizer(*args, **kwargs)
        self._adapter = SpatialAdapter().to(self.device)
        self._hook = p4.register_forward_hook(_p4_hook)
        adapter_group = {'params': list(self._adapter.parameters())}
        # Use the optimizer's resolved value: optimizer=auto may have chosen a
        # different initial LR from args.lr0 before this hook is reached.
        if ADAPTER_LR_MULT is not None:
            # optimizer=auto 가 그룹마다 다른 lr 을 주면(예: MuSGD 특수 그룹 3배) 기준이 모호하다 → 멈춘다 (검수 rev2 codex·agy)
            base_lrs = {g['lr'] for g in optimizer.param_groups}
            if len(base_lrs) != 1:
                raise RuntimeError(f'ADAPTER_LR_MULT needs one body lr, got {sorted(base_lrs)}')
        adapter_group['lr'] = (optimizer.param_groups[0]['lr'] * ADAPTER_LR_MULT
                               if ADAPTER_LR_MULT is not None else self.args.lr0)
        if ADAPTER_WD is not None:
            adapter_group['weight_decay'] = ADAPTER_WD
        optimizer.add_param_group(adapter_group)
        self._spatial_optimizer_built = True
        _SPATIAL_RUN.update({
            'optimizer_actual': type(optimizer).__name__,
            'optimizer_initial_group_lrs': [float(g['lr']) for g in optimizer.param_groups],
            'adapter_initial_lr': float(optimizer.param_groups[-1]['lr']),
            'adapter_weight_decay': float(optimizer.param_groups[-1]['weight_decay']),
            'adapter_params': sum(p.numel() for p in self._adapter.parameters()),
        })
        return optimizer

    def _setup_train(self, *args, **kwargs):
        if self.args.resume:
            raise RuntimeError('resume is forbidden for spatial conditions')
        if getattr(self.args, 'compile', False) or int(os.environ.get('WORLD_SIZE', '1')) != 1:
            raise RuntimeError('spatial KD requires eager single-device training')
        self._teacher = None
        self._batch_id = 0
        self._smoke_ratios = []
        # Initialised once per run, including seed zero; never reseeded per step.
        self._shuffle_generator = torch.Generator(device='cpu').manual_seed(SEED)
        if KD_LOSS_VARIANT in ('relation', 'infonce'):
            self._relation_generator = torch.Generator(device='cpu').manual_seed(SEED)
        if LAM_AUTO_R is not None:
            self._lam_sp = LAM_SP
            self._lam_auto_ratios = []
            self._lam_auto_done = False
        _SPATIAL_CACHE.clear()
        _SPATIAL_RUN.clear()
        _SPATIAL_RUN['entry_sha256'] = _entry_source()[1]
        _KD.update({'calls': 0, 'applied': 0, 'sum': 0.0})
        super()._setup_train(*args, **kwargs)
        self._lam_total_steps = (getattr(self, 'epochs', getattr(self.args, 'epochs', 1))
                                 * len(getattr(self, 'train_loader', (None,))))
        if self.start_epoch != 0:
            raise RuntimeError('spatial runs must start at epoch zero')
        if len(self.optimizer.param_groups) != len(self.scheduler.base_lrs):
            raise RuntimeError('adapter optimizer group missing from scheduler')
        _SPATIAL_RUN['optimizer_group_count'] = len(self.optimizer.param_groups)
        _SPATIAL_RUN['scheduler_group_count'] = len(self.scheduler.base_lrs)
        self._teacher = load_spatial_teacher(self.device)
        _SPATIAL_RUN['resolved_train_args'] = dict(vars(self.args))
        _install_spatial_loss(self)
        self.add_callback('on_train_epoch_end', _spatial_epoch_end)
        # Required for fp32 external-adapter gradients: model AMP remains unchanged.
        print(f'[spatial] {EXP}: norm={TEACHER_NORM}, shuffle={SHUFFLE}, lambda={LAM_SP}, '
              f'interpolation={INTERP_MODE}, reduction={DET_REDUCTION}')

    def preprocess_batch(self, batch):
        _SPATIAL_CACHE.clear()
        batch = super().preprocess_batch(batch)
        clean = batch['img'].clone()
        B = clean.shape[0]
        # Exact same randperm/randn order and noise hyperparameters as legacy.
        n_noisy = max(1, int(B * MIXED_RATIO))
        idx = torch.randperm(B, device=clean.device)[:n_noisy]
        noise = torch.randn(n_noisy, *clean.shape[1:], device=clean.device) * NOISE_SIGMA
        img = batch['img'].clone()
        img[idx] = torch.clamp(img[idx] + noise, 0, 1)
        batch['img'] = img
        noisy = torch.zeros(B, dtype=torch.bool, device=clean.device)
        noisy[idx] = True
        raw, mask = teacher_patches(self._teacher, clean)
        self._batch_id += 1
        batch['_spatial_batch_id'] = self._batch_id
        _SPATIAL_CACHE.update({'active': True, 'batch_id': self._batch_id,
                               'batch_size': B, 'image': batch['img'],
                               'p4_module': self.model.model[KD_LAYER],
                               'teacher': normalize_spatial(raw), 'mask': mask, 'noisy': noisy})
        if KD_LOSS_VARIANT == 'fgcenter':
            _SPATIAL_CACHE['fg_weight'] = foreground_weights(batch, B, clean.device)
        _SPATIAL_RUN['training_batches'] = _SPATIAL_RUN.get('training_batches', 0) + 1
        return batch

    def save_model(self):
        targets = [self.model]
        ema = getattr(getattr(self, 'ema', None), 'ema', None)
        if ema is not None:
            targets.append(ema)
        state = _strip_training_state(targets)
        try:
            return super().save_model()
        finally:
            _restore_training_state(state)


def require_spatial_kd_applied():
    if (_KD['calls'] == 0 or _KD['applied'] != _KD['calls']
            or _KD['applied'] != _SPATIAL_RUN.get('training_batches')):
        raise AssertionError(f'spatial KD application count mismatch: {_KD}')


def _entry_source():
    value = globals().get('__SRC_PATH__')
    if not value or not Path(value).is_file():
        raise RuntimeError('__SRC_PATH__ must name the executed seed_sweep_v2.py')
    path = Path(value).resolve()
    # Compare with the code that was compiled, not merely two copies of an arbitrary path.
    filename = run.__code__.co_filename
    executed = Path(filename).resolve() if not filename.startswith('<') else path
    if executed.is_file() and _sha256(executed) != _sha256(path):
        raise RuntimeError('__SRC_PATH__ hash differs from executed source')
    digest = _sha256(path)
    if not digest:
        raise RuntimeError('entry source hash unavailable')
    if globals().get('_LOADED_SOURCE_SHA256') not in (None, digest):
        raise RuntimeError('entry source changed since code was loaded')
    if filename.startswith('<'):
        # Notebook exec(open(...).read()) has no usable __file__. Check its run
        # code against the supplied source too, so passing the v1 path cannot pass.
        import types
        compiled = compile(path.read_bytes(), filename, 'exec')
        candidate = next((c for c in compiled.co_consts
                          if isinstance(c, types.CodeType) and c.co_name == 'run'), None)
        if candidate != run.__code__:
            raise RuntimeError('__SRC_PATH__ does not match executed run code')
    return path, digest


def verify_student_checkpoint(best_pt):
    """Fresh interpreter imports ONLY ultralytics, never this script/teacher/adapter."""
    import subprocess
    import sys
    program = (
        'import sys, torch; from ultralytics import YOLO; '
        'torch.set_num_threads(2); m=YOLO(sys.argv[1]); '
        'assert all(not x._forward_hooks for x in m.model.modules()); '
        'assert "loss" not in m.model.__dict__; '
        'assert not any("adapter" in n or "teacher" in n for n,_ in m.model.named_modules()); '
        'm.model.eval(); '
        'out=m.model(torch.zeros(1,3,640,640)); print("student-only checkpoint OK")'
    )
    result = subprocess.run([sys.executable, '-B', '-c', program, str(best_pt)],
                            check=True, capture_output=True, text=True)
    return result.stdout.strip()


# ══════════════════════════════════════════════
# provenance — 이 수치가 어디서 나왔는지 남긴다
# ══════════════════════════════════════════════
#  왜 있나: 3개월 전 실험에서 헤드라인 BL0=0.1243 이 어느 run 인지 끝내 못 찾았고,
#  imgsz 800 로그를 640 으로 오인했고, BL_MIXED 가 A_NOISY 의 코드 복제인 것을
#  3개월간 몰랐다. 전부 'run 과 수치를 잇는 기록' 이 없어서 생긴 일이다.
#  ⛔ manifest 에는 사실만 담는다. 해석·청구항·결론은 담지 않는다.

def _sha256(path) -> str:
    h = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ''


# ══════════════════════════════════════════════
# 실측 — 선언값이 아니라 체크포인트가 실제로 기록한 값
# 2026-09-06 신설. HARD-002 검수 3/3 만장일치 지적으로 들어왔다:
#   manifest 에 'hparam_source' 같은 라벨을 적는 것은 감지기가 아니다.
#   6월 사고(BL0 가 실제로는 epochs=2·batch=16 인데 보고서 표에는 10 으로 적힘)는
#   스크립트 상수를 아무리 잘 적어도 안 보인다 — 무엇이 실제로 돌았는지는
#   체크포인트만 안다. 그 사고를 잡은 방법을 그대로 코드로 굳힌 것이다.
# ══════════════════════════════════════════════
_MEASURE_KEYS = ('epochs', 'batch', 'imgsz', 'seed', 'lr0', 'lrf', 'optimizer',
                 'warmup_epochs', 'warmup_bias_lr', 'workers', 'deterministic',
                 'amp', 'fraction')
_SPATIAL_MEASURE_KEYS = _MEASURE_KEYS + ('weight_decay', 'mosaic', 'resume', 'compile')


def _ckpt_train_args(best_pt) -> dict:
    """best.pt 의 train_args 와 실제 기록된 에폭 수를 읽는다."""
    try:
        ck = torch.load(best_pt, map_location='cpu', weights_only=False)
    except Exception as e:
        return {'_error': f'{type(e).__name__}: {e}'}
    ta = ck.get('train_args') or {}
    keys = _SPATIAL_MEASURE_KEYS if IS_SPATIAL else _MEASURE_KEYS
    out = {k: ta.get(k) for k in keys}
    tr = ck.get('train_results') or {}
    curve = tr.get('metrics/mAP50(B)') if isinstance(tr, dict) else None
    out['epochs_actually_recorded'] = len(curve) if hasattr(curve, '__len__') else None
    tm = ck.get('train_metrics') or {}
    out['final_map50_box'] = tm.get('metrics/mAP50(B)')
    return out


def _declared_vs_measured(meas: dict) -> list:
    """선언한 조건과 실측이 어긋난 항목을 나열한다. 빈 리스트면 일치."""
    if not meas or '_error' in meas:
        return [f"체크포인트를 못 읽었다: {meas.get('_error', 'unknown')}"]
    bad = []
    if IS_SPATIAL:
        for key in _SPATIAL_MEASURE_KEYS:
            if meas.get(key) is None:
                bad.append(f'required train_args missing: {key}')
            elif key in _SPATIAL_RUN.get('resolved_train_args', {}):
                expected = _SPATIAL_RUN['resolved_train_args'][key]
                if meas[key] != expected:
                    bad.append(f'{key}: resolved {expected} vs measured {meas[key]}')
        for key, expected in (('lr0', LR0), ('lrf', 0.1), ('optimizer', 'auto'),
                              ('warmup_epochs', 0.1), ('warmup_bias_lr', 0.0),
                              ('weight_decay', 0.0005), ('mosaic', 1.0),
                              ('deterministic', True), ('resume', False), ('compile', False)):
            if meas.get(key) is not None and meas[key] != expected:
                bad.append(f'{key}: declared {expected} vs measured {meas[key]}')
    for k, declared in (('epochs', EPOCHS), ('batch', BATCH), ('imgsz', IMGSZ),
                        ('seed', SEED), ('workers', WORKERS), ('fraction', FRACTION)):
        got = meas.get(k)
        if got is not None and got != declared:
            bad.append(f'{k}: 선언 {declared} vs 실측 {got}')
    n = meas.get('epochs_actually_recorded')
    if IS_SPATIAL and n is None:
        bad.append('required measurement missing: epochs_actually_recorded')
    if n is not None and n != EPOCHS:
        bad.append(f'실제 기록된 에폭 수: 선언 {EPOCHS} vs 실측 {n}')
    return bad


def _spatial_manifest_errors(condition):
    expected = {
        'kd_mode': KD_MODE, 'shuffle': SHUFFLE, 'teacher_norm': TEACHER_NORM,
        'teacher_input_normalized': TEACHER_NORM == 'imagenet',
        'kd_layer': KD_LAYER, 'lam_sp': LAM_SP, 'mask_top': MASK_TOP,
        'interp_mode': INTERP_MODE, 'adapter_params': 787712, 'kd_loss_variant': KD_LOSS_VARIANT,
        'relation_k': RELATION_K, 'infonce_k': INFONCE_K, 'infonce_tau': INFONCE_TAU,
        'fg_alpha': FG_ALPHA, 'lam_auto_r': LAM_AUTO_R, 'lam_auto_steps': LAM_AUTO_STEPS, 'label_prefix_fix': LABEL_PREFIX_FIX, 'skip_val_loss': SKIP_VAL_LOSS,
        'mask_reference_norm': 'imagenet', 'det_reduction': DET_REDUCTION,
        'adapter_lr_mult': ADAPTER_LR_MULT, 'adapter_wd': ADAPTER_WD,
        'adapter_prealign_steps': ADAPTER_PREALIGN_STEPS, 'lam_schedule': LAM_SCHEDULE,
        'lam_warmup_frac': LAM_WARMUP_FRAC, 'lam_end_frac': LAM_END_FRAC,
    }
    return [f'{key}: expected {value!r}, got {condition.get(key)!r}'
            for key, value in expected.items() if condition.get(key) != value]


def write_manifest(run_dir: str, metrics: dict, data_yaml: str) -> str:
    """run 디렉터리에 manifest.json 을 쓰고 경로를 돌려준다."""
    import platform as _plat
    rd = pathlib.Path(run_dir)
    rd.mkdir(parents=True, exist_ok=True)

    # 실제로 돌아간 소스를 run 옆에 복사한다 (해시만 있고 원본이 없으면 재현 불가)
    src_here = pathlib.Path(globals().get('__SRC_PATH__', '/kaggle/working/seed_sweep.py'))
    src_copy = rd / 'seed_sweep.used.py'
    if IS_SPATIAL:
        src_here, digest = _entry_source()
        if _SPATIAL_RUN.get('entry_sha256', digest) != digest:
            raise RuntimeError('entry source changed during training; run invalid')
        shutil.copy(src_here, src_copy)  # failure is fatal, never silently ignored
        if _sha256(src_copy) != digest:
            raise RuntimeError('entry source copy hash mismatch; run invalid')
    else:
        try:
            shutil.copy(src_here, src_copy)
        except Exception:
            src_copy = None

    csv_p = next(iter(sorted(rd.rglob('results.csv'))), None)
    eval_p = rd / 'weights' / f'{EVAL_CKPT}.pt'

    try:
        import ultralytics as _ul; ul_ver = _ul.__version__
    except Exception:
        ul_ver = ''

    # 선언 ↔ 실측. dict 를 만들기 전에 계산해 manifest 안에 함께 남긴다.
    _measured = _ckpt_train_args(eval_p) if eval_p.exists() else {'_error': f'{EVAL_CKPT}.pt 없음'}
    _mismatch = _declared_vs_measured(_measured)

    m = {
        'schema_ver': '1.0',
        'run_id': RUN_NAME,
        'eval_ckpt': EVAL_CKPT,
        'split_lists': split_lists_manifest(),
        'split_check': SPLIT_CHECK,
        'created_utc': datetime.datetime.now(datetime.timezone.utc)
                        .isoformat(timespec='seconds').replace('+00:00', 'Z'),
        'status': 'completed',

        # ── 조건 — 사고 ②③⑤⑥ 을 막는 자리 ──
        'condition': {
            'exp': EXP,
            'kd_enabled': USE_KD,
            'noise_enabled': USE_NOISE,            # BL0 는 False — 2026-09-06 신설
            'workers': WORKERS,                    # 6월 원본 = 8
            'hparam_source': 'june_ckpt_train_args',  # 이 값들의 출처를 못박는다
            'noise_mode': 'per_image_randperm',   # ⚠️ batch_wide 구현과 구분 (사고 ⑥)
            'noise_sigma': NOISE_SIGMA,
            'mixed_ratio': MIXED_RATIO,
            'lam_kd': LAM_KD,
            'eval_sigma': EVAL_SIGMA,
            'imgsz': IMGSZ, 'fraction': FRACTION,                        # 사고 ② (800 vs 640)
            'epochs': EPOCHS,                      # 사고 ⑤ (50ep ÷ 10ep 비대칭)
            'batch': BATCH,
            'seed': SEED,
            'lr0': LR0,
            'base_model': 'yolov8s-seg.pt',
            'teacher': 'dinov2_vitb14' if USE_KD else None,
            'teacher_input_normalized': TEACHER_NORM == 'imagenet',
        },

        # ── 코드 — 사고 ③⑥ ──
        'code': {
            'entry_script': str(src_here),
            'entry_script_sha256': _sha256(src_here),
            'copy_beside_run': str(src_copy) if src_copy else None,
        },

        # ── 환경 ──
        'env': {
            'platform': 'kaggle' if os.path.isdir('/kaggle') else _plat.system().lower(),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'compute_capability': '.'.join(map(str, torch.cuda.get_device_capability(0)))
                                   if torch.cuda.is_available() else None,
            'torch': torch.__version__,
            'ultralytics': ul_ver,
            'python': _plat.python_version(),
            # Kaggle 커밋 URL = 서버 타임스탬프. 특허 실시 시점 증명에 가장 강한 증거.
            'kaggle_kernel_url': globals().get('KAGGLE_KERNEL_URL', ''),
        },

        # ── 데이터 ──
        'data': {
            'data_yaml': data_yaml,
            'data_yaml_sha256': _sha256(data_yaml),
            # Keep these under data too for consumers that group data provenance.
            'split_lists': split_lists_manifest(),
            'split_check': SPLIT_CHECK,
        },

        # ── 결과 — 사고 ① (수치와 run 을 잇는다) ──
        'metrics': metrics,

        # ── 산출물 — 사고 ④ ──
        'artifacts': {
            'results_csv': str(csv_p) if csv_p else None,
            'results_csv_sha256': _sha256(csv_p) if csv_p else '',
            'weights_best': str(rd / 'weights' / 'best.pt') if (rd / 'weights' / 'best.pt').exists() else None,
            'weights_best_sha256': _sha256(rd / 'weights' / 'best.pt') if (rd / 'weights' / 'best.pt').exists() else '',
            'eval_ckpt': EVAL_CKPT,
            'weights_eval': str(eval_p) if eval_p.exists() else None,
            'weights_eval_sha256': _sha256(eval_p) if eval_p.exists() else '',
        },

        # 대조실시예 — 특허 효과 입증에 필수. 여기선 조건명만, 판정은 분석 단계에서.
        'baseline_cond': {'N1_MIXED': 'A_NOISY',      # 노이즈 조건에서의 KD 순기여
                          'C1_CLEAN_RKD': 'BL0',      # 클린 조건에서의 KD 순기여
                          'A_NOISY': 'BL0'}.get(EXP),

        # ── 실측 — 위 condition 은 '선언', 여기는 체크포인트가 '기록'한 값 ──
        'measured': _measured,
        'declared_vs_measured_mismatch': _mismatch,
    }

    if IS_SPATIAL:
        m['schema_ver'] = '2.1'
        m['condition'].update({
            'kd_mode': KD_MODE, 'shuffle': SHUFFLE, 'teacher_norm': TEACHER_NORM,
            'kd_layer': KD_LAYER, 'lam_sp': LAM_SP, 'mask_top': MASK_TOP,
            'interp_mode': INTERP_MODE, 'adapter_params': _SPATIAL_RUN.get('adapter_params'), 'kd_loss_variant': KD_LOSS_VARIANT,
            'relation_k': RELATION_K, 'infonce_k': INFONCE_K, 'infonce_tau': INFONCE_TAU,
            'fg_alpha': FG_ALPHA, 'lam_auto_r': LAM_AUTO_R, 'lam_auto_steps': LAM_AUTO_STEPS, 'label_prefix_fix': LABEL_PREFIX_FIX, 'skip_val_loss': SKIP_VAL_LOSS,
            'mask_reference_norm': 'imagenet', 'det_reduction': DET_REDUCTION,
            'adapter_lr_mult': ADAPTER_LR_MULT, 'adapter_wd': ADAPTER_WD,
            'adapter_prealign_steps': ADAPTER_PREALIGN_STEPS, 'lam_schedule': LAM_SCHEDULE,
            'lam_warmup_frac': LAM_WARMUP_FRAC, 'lam_end_frac': LAM_END_FRAC,
            'lam_sp_shared': LAM_SP_SHARED, 'lam_kd': None, 'cls_rkd_enabled': False,
        })
        m['spatial_runtime'] = dict(_SPATIAL_RUN)
        m['lam_auto'] = _SPATIAL_RUN.get('lam_auto')
        m['baseline_cond'] = {'S0': 'A_NOISY', 'S1': 'S0', 'S2': 'S1', 'S1r': 'S1'}[EXP]
        log_path = rd / 'spatial_kd.jsonl'
        m['artifacts']['spatial_kd_log'] = str(log_path)
        m['artifacts']['spatial_kd_log_sha256'] = _sha256(log_path)
        if not log_path.is_file():
            _mismatch.append('spatial_kd.jsonl missing')
        _mismatch.extend(_spatial_manifest_errors(m['condition']))
        if (_KD['calls'] == 0 or _KD['calls'] != _KD['applied']
                or _KD['applied'] != _SPATIAL_RUN.get('training_batches')):
            _mismatch.append(f'spatial KD count mismatch: {_KD}')
        if LAM_AUTO_R is not None and LAM_SP > 0 and not m.get('lam_auto'):
            _mismatch.append(f'LAM_AUTO_R set but recalibration never fired (steps <= {LAM_AUTO_STEPS}?)')
        for key in ('optimizer_actual', 'optimizer_group_lrs', 'checkpoint_reload'):
            if not _SPATIAL_RUN.get(key):
                _mismatch.append(f'required runtime evidence missing: {key}')
        m['status'] = 'invalid' if _mismatch else 'completed'
    blob = json.dumps(m, ensure_ascii=False, indent=2, sort_keys=True)   # 문자열 먼저 완성
    (rd / 'manifest.json').write_text(blob, encoding='utf-8')
    print(f'[manifest] {rd / "manifest.json"} ({len(blob)}B)')
    if not m['artifacts']['results_csv']:
        print('[🚨] results.csv 를 못 찾았다 — 학습곡선이 남지 않는다')

    # manifest 를 먼저 쓴 뒤에 판정한다 — 증거를 남기고 큰소리로 죽기 위해서다.
    if _mismatch:
        raise AssertionError(
            '선언한 조건과 체크포인트 실측이 다르다. 이 run 의 수치는 쓰면 안 된다:\n  '
            + '\n  '.join(_mismatch)
            + '\n(manifest 는 기록됐다 — declared_vs_measured_mismatch 참조)')
    print(f'[manifest] 선언↔실측 일치 확인 '
          f'(ep={_measured.get("epochs")} batch={_measured.get("batch")} '
          f'workers={_measured.get("workers")} opt={_measured.get("optimizer")})')
    return str(rd / 'manifest.json')


# ══════════════════════════════════════════════
# 평가 — clean / noisy(σ=0.3)
# ══════════════════════════════════════════════
def evaluate(best_pt: str, data_yaml: str):
    from ultralytics.models.yolo.segment.val import SegmentationValidator

    class NoisyVal(SegmentationValidator):
        def preprocess(self, b):
            b = super().preprocess(b)
            b['img'] = (b['img'] + EVAL_SIGMA * torch.randn_like(b['img'])).clamp(0, 1)
            return b

    # ⚠️ m.val(validator=...) 는 실재하는 인자다 (ultralytics 8.4.39 engine/model.py:584
    #    `def val(self, validator=None, **kwargs)`). 검수 6개 중 4개가 '그런 인자 없다,
    #    TypeError' 라고 CRITICAL 로 지적했으나 소스 확인 결과 오답이었다. 지우지 말 것.
    m = YOLO(best_pt)
    set_all_seeds(SEED)          # 평가 노이즈도 재현 가능하게
    clean = m.val(data=data_yaml, imgsz=IMGSZ, batch=BATCH, workers=0, verbose=False)
    set_all_seeds(SEED)
    noisy = m.val(data=data_yaml, imgsz=IMGSZ, batch=BATCH, workers=0,
                  verbose=False, validator=NoisyVal)
    # 2026-09-03: 주 지표는 box.map50 이다. 원본 eval_noisy.py:38 과 ablation.py:132 가
    # .box.map50 을 썼고 보고서의 모든 수치(BL0 0.1243, +0.1606, x11.4)가 그 값이다.
    # 여기서 seg.map50 을 쓰면 이름만 같고 비교 불가능한 값이 나온다.
    return (float(clean.box.map50), float(noisy.box.map50),
            float(clean.seg.map50), float(noisy.seg.map50))


SKIP_VAL_LOSS = bool(globals().get('SKIP_VAL_LOSS', True))


def _skip_val_loss(trainer):
    """학습 중 val 의 손실 계산만 건너뛴다 (2026-10-23).
    ultralytics 8.4.39 validator.py:227 은 학습 중 val 마다 EMA 모델의 model.loss 를 부르고, 객체가 아주 많은 val 배치(219/558)에서
    TaskAlignedAssigner 가 GPU OOM → CPU 로 넘어가 호스트 RAM 이 터졌다 [실측: 데스크탑 WSL OOM-kill, Kaggle dual 두 프로세스 동시 사망].
    best.pt 선택(fitness)은 seg+box mAP50-95 만 쓰므로(metrics.py:1266) val 손실은 기록용이다 → eval 모드에서만 0 을 돌려준다.
    학습 모드(self.training=True)의 손실·기울기는 그대로다. results.csv 의 val/*_loss 는 0 으로 남는다."""
    if not SKIP_VAL_LOSS:
        return
    m = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
    cls = type(m)
    cls._skip_val_trainer = trainer          # 실행마다 갱신 — 같은 프로세스 두 번째 실행이 옛 trainer 를 잡지 않게 (검수 valloss codex#1)
    if getattr(cls, '_val_loss_skipped', False):
        return
    orig = cls.loss

    def loss(self, batch, preds=None, *a, **k):
        if self.training:
            return orig(self, batch, preds, *a, **k)
        z = torch.zeros_like(type(self)._skip_val_trainer.loss_items)
        return z.sum(), z
    cls.loss = loss
    cls._val_loss_skipped = True
    print(f'[val] 학습 중 val 손실 계산 생략 ({cls.__name__}.loss, eval 모드만)')


def run():
    if IS_SPATIAL:
        _entry_source()  # fail BEFORE loading data/model or starting an expensive run
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    set_all_seeds(SEED)
    data_yaml = find_data_yaml()
    global SPLIT_CHECK
    SPLIT_CHECK = check_split_disjoint(data_yaml)
    print(f'[data] split gate: train={SPLIT_CHECK["train_tiles"]} tiles / '
          f'val={SPLIT_CHECK["val_tiles"]} tiles / overlap={SPLIT_CHECK["overlap_count"]}')
    print(f'=== {RUN_NAME} | KD={USE_KD} | imgsz={IMGSZ} batch={BATCH} ep={EPOCHS} ===')

    model = YOLO('yolov8s-seg.pt')
    model.add_callback('on_train_start', _skip_val_loss)
    model.train(
        data=data_yaml, epochs=EPOCHS, batch=BATCH, imgsz=IMGSZ,
        device=0, seed=SEED, deterministic=True,   # seed 비교 실험 — 검수 5/5 만장일치
        # 2026-09-06: 6월 원본 3조건(C1/A_NOISY/N1_MIXED)의 체크포인트 train_args 를
        #   직접 읽어 맞춘 값이다. 세 조건은 하이퍼파라미터가 완전히 동일했다(차이 0건).
        #   여기가 어긋나면 이번 런이 6월 +0.0071 과 비교 불가능해진다.
        optimizer='auto', lr0=LR0, lrf=0.1,
        warmup_epochs=0.1, warmup_bias_lr=0.0,
        weight_decay=0.0005, mosaic=1.0,          # ultralytics 기본값과 동일 — 명시만
        project=PROJECT, name=RUN_NAME, exist_ok=True,
        workers=WORKERS, verbose=True, plots=True,  # plots=True → results.png 남는다
        fraction=FRACTION,
        trainer=SpatialNoiseTrainer if IS_SPATIAL else SharedNoiseTrainer,
    )
    try:  # 캐시는 학습 시작 때 이미 생겼다. 실패해도 실행은 계속
        print(f'[data] 라벨 캐시 내보냄 → {CACHE_EXPORT}: {export_caches(MIRROR_DST, CACHE_EXPORT)}')
    except Exception as e:
        print(f'[data] 라벨 캐시 내보내기 실패 (실행에는 영향 없음): {e!r}')

    # ── KD 가 정말 걸렸는지 확인 ── 침묵 실패라면 여기서 멈춘다
    if USE_KD:
        if IS_SPATIAL:
            require_spatial_kd_applied()
        print(f'[KD] loss 호출 {_KD["calls"]}회 / KD 적용 {_KD["applied"]}회 '
              f'/ 평균 kd={_KD["sum"]/max(_KD["applied"],1):.4f}')
        assert _KD['applied'] > 0, (
            'KD 가 한 번도 적용되지 않았다. model.loss 덮어쓰기가 안 먹은 것이다 — '
            'N1_MIXED 가 A_NOISY 와 동일해지므로 이 결과는 쓰면 안 된다.')
    else:
        print(f'[KD] 미사용 조건 (loss 호출 {_KD["calls"]}회, KD 적용 {_KD["applied"]}회 — 0 이어야 정상)')
        assert _KD['applied'] == 0, 'A_NOISY 인데 KD 가 적용됐다'

    # ── 학습 자원 해제 후 평가 ── 조건을 연달아 돌리므로 VRAM 을 비운다 (검수 3/4 지적)
    import gc
    del model              # trainer·EMA·teacher 참조를 끊는다
    gc.collect()
    torch.cuda.empty_cache()
    print(f'[VRAM] 학습 후 정리: {torch.cuda.memory_allocated()/1e9:.2f}GB 사용 중')

    eval_pt = f'{PROJECT}/{RUN_NAME}/weights/{EVAL_CKPT}.pt'
    if IS_SPATIAL:
        _SPATIAL_RUN['checkpoint_reload'] = verify_student_checkpoint(eval_pt)
        _SPATIAL_RUN['peak_memory_bytes'] = torch.cuda.max_memory_allocated()
        # 13 GiB 는 명세 v2.1 의 [추정] 작업 예산이다 — 실측 근거 없는 무효 판정은 하지 않고 기록만 한다
        # (2026-10-23: Kaggle S0 s0 가 14.12 GiB 로 10에폭을 정상 완료했는데 이 줄 때문에 평가가 빠졌다).
        _SPATIAL_RUN['t4_budget_exceeded'] = bool('T4' in torch.cuda.get_device_name(0) and _SPATIAL_RUN['peak_memory_bytes'] > 13 * 2**30)
        if _SPATIAL_RUN['t4_budget_exceeded']:
            print(f"[경고] T4 작업 예산 13 GiB 초과: peak {_SPATIAL_RUN['peak_memory_bytes'] / 2**30:.2f} GiB — 평가는 계속한다")
    mc, mn, sc, sn = evaluate(eval_pt, data_yaml)
    print(f'\n=== 결과 {RUN_NAME} ===')
    print(f'  [주지표] box  Clean mAP50 : {mc:.4f}')
    print(f'  [주지표] box  Noisy mAP50 : {mn:.4f}  (σ={EVAL_SIGMA})')
    print(f'  [참고]   mask Clean/Noisy : {sc:.4f} / {sn:.4f}')

    gc.collect(); torch.cuda.empty_cache()

    metrics = {'map50_clean': mc, 'map50_noisy': mn,          # box = 주지표, 보고서와 동일 계열
               'seg_map50_clean': sc, 'seg_map50_noisy': sn,   # mask = 참고용
               'metric_family': 'box',
               'eval_ckpt': EVAL_CKPT,
               'kd_calls': _KD['calls'], 'kd_applied': _KD['applied'],
               'kd_mean': _KD['sum']/max(_KD['applied'], 1)}
    run_dir = f'{PROJECT}/{RUN_NAME}'
    manifest_path = write_manifest(run_dir, metrics, data_yaml)

    return {'exp': EXP, 'seed': SEED, 'epochs': EPOCHS, 'run_id': RUN_NAME, 'eval_ckpt': EVAL_CKPT,
            'run_dir': run_dir, 'manifest': manifest_path, **metrics}


# Snapshot at import/exec time, so editing the entry file during or after loading
# cannot make two hashes of the same changed path look like proof of provenance.
if IS_SPATIAL:
    _source_filename = run.__code__.co_filename
    _source_loaded = (globals().get('__SRC_PATH__') if _source_filename.startswith('<')
                      else _source_filename)
    _LOADED_SOURCE_SHA256 = _sha256(_source_loaded) if _source_loaded else None
