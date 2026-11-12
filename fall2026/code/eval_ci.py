"""v23 체크포인트의 영상 단위 평가 기록 + paired bootstrap 신뢰구간 (탐색, 사전등록 44번 교체안, 2026-10-27).

논문 수치와 같은 경로로 평가한다: seed_sweep_v2.evaluate 의 NoisyVal(σ=0.3, set_all_seeds(SEED) 직후 val)과
같은 검증기·배치·시드. 검증기를 상속해 영상마다 box 매칭 결과(IoU 0.5)와 상자 넓이를 저장하므로,
전체 표본에서 다시 계산한 mAP50 이 검증기 값과 같아야 한다(P0, 다르면 중단).

쓰기 (데스크탑):
  python eval_ci.py extract NAME CKPT OUT_DIR        # GPU, clean·noisy 두 번 val → OUT_DIR/NAME_{clean,noisy}.npy
  python eval_ci.py boot OUT_DIR --pairs S1:S0,S2:S0,X1:S0,X1:S2,S1:S2,S0:BL0 [--B 1000]
                                                     # CPU → OUT_DIR/ci.json

작은 객체 AP: 640 입력 좌표에서 GT 상자 넓이 < 32² 를 작은 객체로 둔다(COCO 기준을 입력 해상도에 적용).
COCO 처럼, 작은 GT 와 매칭되지 않았는데 같은 클래스의 큰 GT 와 IoU≥0.5 인 예측과, 매칭되지 않은 큰 예측
(넓이 ≥ 32²)은 계산에서 뺀다 — 근사이며 pycocotools 와 정확히 같지는 않다.
"""
import argparse, json, multiprocessing as mp, pathlib, time
import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
SMALL_AREA = 32 ** 2
IOU50 = 0  # ultralytics iouv 의 첫 칸 = 0.5


def load_train_globals(src):
    """seed_sweep_v2.py 를 run() 없이 실행해 SEED·BATCH·IMGSZ·EVAL_SIGMA·set_all_seeds 를 그대로 쓴다."""
    g = {'EXP': 'BL0', 'SEED': 0, 'DATA_ROOT': None, '__name__': '__evalci__', '__SRC_PATH__': str(src)}
    exec(compile(pathlib.Path(src).read_text(), str(src), 'exec'), g)
    return g


def make_recording_validator(base, sigma, noisy):
    from ultralytics.utils.metrics import box_iou

    class Rec(base):
        records = []

        def preprocess(self, b):
            b = super().preprocess(b)
            if noisy:  # seed_sweep_v2.evaluate 의 NoisyVal 과 같은 식
                b['img'] = (b['img'] + sigma * torch.randn_like(b['img'])).clamp(0, 1)
            return b

        def update_metrics(self, preds, batch):
            # DetectionValidator.update_metrics(8.4.39) 의 통계 부분과 같게 하고, 영상 기록을 덧붙인다.
            for si, pred in enumerate(preds):
                self.seen += 1
                pbatch = self._prepare_batch(si, batch)
                predn = self._prepare_pred(pred)
                cls = pbatch['cls'].cpu().numpy()
                no_pred = predn['cls'].shape[0] == 0
                stat = self._process_batch(predn, pbatch)
                conf = np.zeros(0) if no_pred else predn['conf'].cpu().numpy()
                pcls = np.zeros(0) if no_pred else predn['cls'].cpu().numpy()
                self.metrics.update_stats({**stat, 'target_cls': cls, 'target_img': np.unique(cls), 'conf': conf, 'pred_cls': pcls})
                tb, pb = pbatch['bboxes'], predn['bboxes']
                t_area = ((tb[:, 2] - tb[:, 0]) * (tb[:, 3] - tb[:, 1])).cpu().numpy() if len(tb) else np.zeros(0)
                p_area = ((pb[:, 2] - pb[:, 0]) * (pb[:, 3] - pb[:, 1])).cpu().numpy() if not no_pred else np.zeros(0)
                small = t_area < SMALL_AREA
                if no_pred:
                    tp_s = np.zeros(0, bool); ign = np.zeros(0, bool)
                else:
                    sm = torch.as_tensor(small, device=tb.device)
                    if small.any():  # 작은 GT 만 남기고 다시 매칭
                        tp_s = self.match_predictions(predn['cls'], pbatch['cls'][sm], box_iou(tb[sm], pb)).cpu().numpy()[:, IOU50]
                    else:
                        tp_s = np.zeros(len(pcls), bool)
                    hit_big = np.zeros(len(pcls), bool)
                    if (~small).any():
                        iou_b = box_iou(tb[~sm], pb).cpu().numpy()  # (n_big, n_pred)
                        hit_big = ((iou_b >= 0.5) & (cls[~small][:, None] == pcls[None, :])).any(0)
                    ign = (~tp_s) & (hit_big | (p_area >= SMALL_AREA))
                self.records.append({'im': pbatch['im_file'], 'tp': stat['tp'][:, IOU50].astype(bool), 'conf': conf.astype(np.float32),
                                     'pcls': pcls.astype(np.int16), 'tcls': cls.astype(np.int16), 'tsmall': small,
                                     'tp_s': tp_s, 'ign_s': ign})
    return Rec


def ap50(recs, idx, small=False):
    """ultralytics ap_per_class 로 mAP50 (주어진 영상 색인, 중복 허용)."""
    from ultralytics.utils.metrics import ap_per_class
    tp, conf, pc, tc = [], [], [], []
    for i in idx:
        r = recs[i]
        if small:
            keep = ~r['ign_s']
            tp.append(r['tp_s'][keep]); conf.append(r['conf'][keep]); pc.append(r['pcls'][keep]); tc.append(r['tcls'][r['tsmall']])
        else:
            tp.append(r['tp']); conf.append(r['conf']); pc.append(r['pcls']); tc.append(r['tcls'])
    tp, conf, pc, tc = (np.concatenate(x) for x in (tp, conf, pc, tc))
    if len(tc) == 0:
        return float('nan')
    ap = ap_per_class(tp[:, None], conf, pc, tc, plot=False)[5]  # (클래스, IoU 1칸)
    return float(ap[:, 0].mean())


def extract(a):
    g = load_train_globals(HERE / 'seed_sweep_v2.py')
    from ultralytics import YOLO
    from ultralytics.models.yolo.segment.val import SegmentationValidator
    out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for cond in ('clean', 'noisy'):
        Rec = make_recording_validator(SegmentationValidator, g['EVAL_SIGMA'], cond == 'noisy')
        Rec.records = []
        m = YOLO(a.ckpt)
        g['set_all_seeds'](g['SEED'])
        res = m.val(data=a.data, imgsz=g['IMGSZ'], batch=g['BATCH'], workers=0, verbose=False, plots=False, validator=Rec)
        recs = Rec.records
        full = ap50(recs, range(len(recs)))
        summary[cond] = {'validator_map50': float(res.box.map50), 'recomputed_map50': full, 'n_images': len(recs),
                         'small_ap50': ap50(recs, range(len(recs)), small=True),
                         'n_small_gt': int(sum(r['tsmall'].sum() for r in recs)), 'n_gt': int(sum(len(r['tcls']) for r in recs))}
        if abs(full - float(res.box.map50)) > 1e-6:
            raise SystemExit(f'P0 실패: 재계산 {full} ≠ 검증기 {res.box.map50} ({cond})')
        np.save(out / f'{a.name}_{cond}.npy', np.array(recs, dtype=object), allow_pickle=True)
    blob = json.dumps({'name': a.name, 'ckpt': a.ckpt, 'data': a.data, 'eval_sigma': g['EVAL_SIGMA'], 'seed': g['SEED'],
                       'batch': g['BATCH'], 'imgsz': g['IMGSZ'], 'summary': summary}, ensure_ascii=False, indent=2)
    (out / f'{a.name}_summary.json').write_text(blob)
    print(blob)


_G = {}  # fork 된 작업자가 공유하는 기록 (boot 전용)


def _boot_one(args):
    nm, c, small, ix = args
    return ap50(_G[(nm, c)], ix, small)


def boot(a):
    out = pathlib.Path(a.out_dir); rng = np.random.default_rng(a.seed)
    pairs = [p.split(':') for p in a.pairs.split(',')]
    names = sorted({n for p in pairs for n in p})
    recs = {(n, c): list(np.load(out / f'{n}_{c}.npy', allow_pickle=True)) for n in names for c in ('clean', 'noisy')}
    order = [r['im'] for r in recs[(names[0], 'clean')]]
    for k, v in recs.items():  # 짝 비교는 같은 영상 순서여야 한다
        if [r['im'] for r in v] != order:
            raise SystemExit(f'영상 순서 불일치: {k}')
    _G.update(recs)
    n = len(order); idxs = [rng.integers(0, n, n) for _ in range(a.B)]
    res = {'B': a.B, 'seed': a.seed, 'n_images': n, 'point': {}, 'diff': {}}
    t0 = time.time()
    for c in ('clean', 'noisy'):
        for small in (False, True):
            key = f'{c}_{"small" if small else "all"}'
            with mp.get_context('fork').Pool(a.procs) as pool:
                dist = {nm: np.array(pool.map(_boot_one, [(nm, c, small, ix) for ix in idxs], chunksize=8)) for nm in names}
            res['point'][key] = {nm: ap50(recs[(nm, c)], range(n), small) for nm in names}
            for x, y in pairs:
                d = dist[x] - dist[y]; d = d[~np.isnan(d)]; pt = res['point'][key][x] - res['point'][key][y]
                lo, hi = np.nanpercentile(d, [2.5, 97.5])
                res['diff'].setdefault(key, {})[f'{x}-{y}'] = {'point': pt, 'ci95': [float(lo), float(hi)], 'p_le0': float(np.mean(d <= 0)), 'n_valid': int(d.size)}
            print(f'{key} done {time.time() - t0:.0f}s', flush=True)
    blob = json.dumps(res, ensure_ascii=False, indent=2)
    (out / 'ci.json').write_text(blob); print(blob)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest='cmd', required=True)
    e = sp.add_parser('extract'); e.add_argument('name'); e.add_argument('ckpt'); e.add_argument('out_dir'); e.add_argument('--data', default='/tmp/isaid_fixed.yaml')
    b = sp.add_parser('boot'); b.add_argument('out_dir'); b.add_argument('--pairs', required=True); b.add_argument('--B', type=int, default=1000); b.add_argument('--seed', type=int, default=0); b.add_argument('--procs', type=int, default=10)
    a = ap.parse_args(); {'extract': extract, 'boot': boot}[a.cmd](a)
