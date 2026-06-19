"""
kd_v7_5_n1mixed_fulltime.py — N1_MIXED 50ep FULLTIME (Kaggle A100)

사용법 (Kaggle Notebook Cell):
    exec(open('/kaggle/working/kd_v7_5_n1mixed_fulltime.py').read())
    run_n1mixed()

핵심 설계:
  - N1_MIXED: 50% 배치에 Gaussian 노이즈 (σ=0.3) / Teacher는 항상 clean 입력
  - resume: optimizer state 자동 strip → param group 불일치 방지
  - adapter 가중치: last.pt 옆에 last_kd_adapter.pt 사이드카로 별도 저장/로드
  - Ultralytics 8.4.62+ API 변경 (_setup_train world_size 인자 제거) 대응
"""

import os, gc, glob, shutil, yaml, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.models.yolo.segment import SegmentationTrainer

# ══════════════════════════════════════════════
# 상수 — 필요 시 수정
# ══════════════════════════════════════════════
LAST_PT       = '/content/drive/MyDrive/fedv9_v3/checkpoints/N1_MIXED_50ep/last.pt'
PROJECT       = '/content/hy_N1_MIXED'
RUN_NAME      = 'N1_MIXED_50ep'
FULL_EPOCHS   = 50
BATCH         = 8
LR0           = 5e-5          # resume 시 낮은 lr 권장
IMGSZ         = 640
NOISE_SIGMA   = 50 / 255  # ≈ 0.196 (학습용, 평가 σ=0.3보다 약한 조건)
MIXED_RATIO   = 0.5           # 배치 내 noisy 이미지 비율
DINOV2_SIZE   = 448           # 14의 배수 (14×32)
LAM_KD        = 0.5           # RKD loss weight

# ══════════════════════════════════════════════
# data.yaml 자동 탐색
# ══════════════════════════════════════════════
def _find_data_yaml() -> str:
    candidates = [
        '/tmp/isaid_fixed.yaml',
        '/kaggle/input/isaid-dataset-yolo11-seg-format/iSAIDYolo11Seg/data.yaml',
    ]
    # kagglehub 캐시
    for root in glob.glob('/root/.cache/kagglehub/datasets/redzapdos123/**', recursive=False):
        candidates += glob.glob(f'{root}/**/data.yaml', recursive=True)

    for c in candidates:
        if os.path.exists(c):
            # nc 확인 후 isaid_fixed.yaml로 복사 (nc=16 보장)
            return _ensure_nc16(c)
    raise FileNotFoundError('data.yaml을 찾을 수 없습니다. Kaggle 데이터셋을 추가하세요.')


def _ensure_nc16(yaml_path: str) -> str:
    """nc=16이 아닌 경우 /tmp/isaid_fixed.yaml로 복사해 수정"""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    if cfg.get('nc') == 16 and yaml_path == '/tmp/isaid_fixed.yaml':
        return yaml_path
    # 경로 검증
    base = Path(yaml_path).parent
    cfg['nc'] = 16
    if cfg.get('path') and not Path(cfg['path']).exists():
        cfg['path'] = str(base)
    fixed = '/tmp/isaid_fixed.yaml'
    with open(fixed, 'w') as f:
        yaml.dump(cfg, f)
    print(f'[DataYAML] {yaml_path} → {fixed} (nc=16)')
    return fixed


# ══════════════════════════════════════════════
# Teacher (DINOv2 ViT-B/14)
# ══════════════════════════════════════════════
_teacher_cache = None

def get_teacher():
    global _teacher_cache
    if _teacher_cache is None:
        print('[Teacher] DINOv2 ViT-B/14 로드 중...')
        _teacher_cache = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)
        _teacher_cache.eval().cuda()
        for p in _teacher_cache.parameters():
            p.requires_grad_(False)
        print('[Teacher] 완료')
    return _teacher_cache


def extract_teacher_cls(teacher, imgs_clean, block=11):
    """clean 이미지 → DINOv2 block11 CLS token (B, 768)"""
    feat = {}
    hook = teacher.blocks[block].register_forward_hook(
        lambda m, i, o: feat.update({block: o[:, 0].detach()}))
    with torch.no_grad():
        resized = F.interpolate(imgs_clean, size=(DINOV2_SIZE, DINOV2_SIZE),
                                mode='bilinear', align_corners=False)
        teacher(resized)
    hook.remove()
    return feat[block]   # (B, 768)


# ══════════════════════════════════════════════
# RKD Loss
# ══════════════════════════════════════════════
def rkd_loss(s, t):
    s = F.normalize(s, dim=-1)
    t = F.normalize(t, dim=-1)
    sd = torch.cdist(s, s)
    td = torch.cdist(t, t)
    td = td / (td.mean() + 1e-8)
    sd = sd / (sd.mean() + 1e-8)
    return F.smooth_l1_loss(sd, td)


# ══════════════════════════════════════════════
# N1_MIXED KD Trainer
# ══════════════════════════════════════════════
class N1MixedKDTrainer(SegmentationTrainer):
    """
    N1_MIXED 전용 트레이너.
    - 배치의 MIXED_RATIO 비율에 Gaussian 노이즈 적용 (Student 입력)
    - Teacher는 항상 clean 이미지 입력
    - RKD loss: student adapter output ↔ teacher CLS token
    - resume: _setup_train에서 optimizer state 자동 strip
    - adapter 가중치: ADAPTER_SIDECAR 경로에서 로드/저장
    """

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self._teacher        = None
        self._adapter        = None
        self._s_feat         = None    # SPPF hook이 채움
        self._t_feat         = None    # preprocess_batch가 채움
        self._sppf_ch        = 512
        self._orig_loss      = None

    # ── SPPF 동적 탐색 ──
    def _find_sppf(self):
        for m in self.model.model.model:
            if m.__class__.__name__ == 'SPPF':
                return m
        for idx in [9, 10]:
            try:
                m = self.model.model.model[idx]
                if hasattr(m, 'cv2'):
                    return m
            except (IndexError, AttributeError):
                pass
        raise RuntimeError('[N1Mixed] SPPF layer를 찾을 수 없습니다.')

    # ── 핵심: resume 체크포인트에서 optimizer 자동 strip ──
    def _setup_train(self, world_size=0):
        if self.args.resume:
            _path = str(self.args.model)
            if os.path.exists(_path):
                _c = torch.load(_path, map_location='cpu', weights_only=False)
                if _c.get('optimizer') is not None:
                    _c.pop('optimizer', None)
                    _c.pop('opt', None)
                    torch.save(_c, _path)
                    print('[N1Mixed] ✅ optimizer stripped from checkpoint')

        # Ultralytics 8.4.62+ API 변경 (world_size 인자 제거) 대응
        try:
            super()._setup_train(world_size)
        except TypeError:
            super()._setup_train()

        # Teacher 로드
        self._teacher = get_teacher()

        # SPPF 탐색 + 채널 수
        sppf = self._find_sppf()
        try:
            self._sppf_ch = sppf.cv2.conv.out_channels
        except AttributeError:
            self._sppf_ch = 512

        # SPPF hook 등록
        def _sppf_hook(m, inp, out):
            self._s_feat = out.mean(dim=[-1, -2])  # GAP → (B, C)
        sppf.register_forward_hook(_sppf_hook)

        # Adapter: sppf_ch → 256 → 768
        self._adapter = nn.Sequential(
            nn.Linear(self._sppf_ch, 256, bias=False),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 768, bias=False),
        ).cuda()

        # resume 시 adapter 사이드카 로드 — checkpoint 경로 기반으로 동적 계산
        resume_sidecar = str(self.args.model).replace('.pt', '_kd_adapter.pt')
        if self.args.resume and os.path.exists(resume_sidecar):
            try:
                state = torch.load(resume_sidecar, map_location='cpu', weights_only=False)
                self._adapter.load_state_dict(state, strict=True)
                print(f'[N1Mixed] ✅ adapter loaded from {resume_sidecar}')
            except Exception as e:
                print(f'[N1Mixed] ⚠️ adapter sidecar load 실패 (random init): {e}')
        else:
            print('[N1Mixed] adapter: random init')

        # optimizer에 adapter 파라미터 추가
        kd_params = list(self._adapter.parameters())
        self.optimizer.add_param_group({'params': kd_params, 'lr': self.args.lr0})
        print(f'[N1Mixed] sppf_ch={self._sppf_ch} | kd_params={len(kd_params)} | lam={LAM_KD}')

        # model.loss 래핑 (task_loss + LAM_KD * rkd_loss)
        self._orig_loss = self.model.loss
        _trainer = self

        def _kd_loss_wrapper(batch, preds=None):
            loss, loss_items = _trainer._orig_loss(batch, preds)
            s = _trainer._s_feat
            t = _trainer._t_feat
            if s is not None and t is not None:
                kd = rkd_loss(_trainer._adapter(s), t)
                loss = loss + LAM_KD * kd
            return loss, loss_items

        self.model.loss = _kd_loss_wrapper

    # ── 배치 전처리: 노이즈 적용 + Teacher feature 캐싱 ──
    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)  # imgs: float [0,1] on CUDA

        if self._teacher is not None:
            clean_imgs = batch['img'].clone()

            # N1_MIXED: MIXED_RATIO 비율 배치에 노이즈 주입 (Student용)
            B = clean_imgs.shape[0]
            n_noisy = max(1, int(B * MIXED_RATIO))
            noisy_imgs = batch['img'].clone()
            idx = torch.randperm(B, device=noisy_imgs.device)[:n_noisy]
            noise = torch.randn(n_noisy, *clean_imgs.shape[1:],
                                device=noisy_imgs.device) * NOISE_SIGMA
            noisy_imgs[idx] = torch.clamp(noisy_imgs[idx] + noise, 0, 1)
            batch['img'] = noisy_imgs

            # Teacher는 clean 이미지만 처리
            with torch.no_grad():
                self._t_feat = extract_teacher_cls(self._teacher, clean_imgs, block=11)

        return batch

    # ── Epoch 종료 시 adapter 사이드카 저장 ──
    def save_model(self):
        super().save_model()
        # adapter 사이드카: last.pt 경로 기반으로 저장
        try:
            save_dir = self.save_dir
            for ckpt_name in ['last.pt', 'best.pt']:
                ckpt_path = save_dir / 'weights' / ckpt_name
                if ckpt_path.exists() and self._adapter is not None:
                    sidecar = str(ckpt_path).replace('.pt', '_kd_adapter.pt')
                    torch.save(self._adapter.state_dict(), sidecar)
            print(f'[N1Mixed] adapter sidecar saved')
        except Exception as e:
            print(f'[N1Mixed] ⚠️ adapter sidecar save 실패: {e}')


# ══════════════════════════════════════════════
# 실행 함수
# ══════════════════════════════════════════════
def run_n1mixed():
    data_yaml = _find_data_yaml()
    print(f'[N1Mixed] DATA_YAML: {data_yaml}')
    print(f'[N1Mixed] RESUME: {LAST_PT}')
    print(f'[N1Mixed] ep28 → {FULL_EPOCHS}ep | batch={BATCH} | lam={LAM_KD}')

    if not os.path.exists(LAST_PT):
        raise FileNotFoundError(
            f'체크포인트 없음: {LAST_PT}\n'
            f'Drive에서 복사하거나 경로를 수정하세요.')

    # Ultralytics 공식 resume 패턴: model=checkpoint 경로 + resume=True
    model = YOLO(LAST_PT)
    results = model.train(
        data=data_yaml,
        epochs=FULL_EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        lr0=LR0,
        lrf=0.1,
        momentum=0.937,
        optimizer='auto',
        project=PROJECT,
        name=RUN_NAME,
        exist_ok=True,
        resume=True,
        trainer=N1MixedKDTrainer,
        verbose=True,
        plots=False,
    )

    best_pt = str(results.save_dir / 'weights' / 'best.pt')
    print(f'\n[완료] best.pt → {best_pt}')

    # Drive 백업
    drive_dir = Path('/kaggle/working/drive_backup')
    drive_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_pt, drive_dir / 'N1_MIXED_50ep_best.pt')
    print(f'[백업] → {drive_dir}/N1_MIXED_50ep_best.pt')

    return best_pt


if __name__ == '__main__':
    run_n1mixed()
