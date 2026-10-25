"""정정 4 계산: (a) 옛 val 로 정정 1 경계 재현 (b) 새 val(공식 458 영상) 경계·객체 수 (c) R2 표본 재추출. r4_run.load_gt 와 같은 박스 규칙."""
import os, sys, numpy as np
from PIL import Image
sys.path.insert(0, os.path.expanduser('~/kd_r2'))
from kdr_r2_core import letterbox_params, sample_names
from kdr_r4_core import edges_from_areas
R = os.path.expanduser('~/datasets/isaid_kaggle/iSAIDYolo11Seg')
def areas(paths):
    out, bad = [], 0
    for ip in paths:
        w, h = Image.open(ip).size; r, _, _ = letterbox_params(w, h)
        lp = ip.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
        for line in open(lp):
            v = np.asarray(line.split(), dtype=np.float64); q = v[5:].reshape(-1, 2)
            bb = np.array([(q[:, 0].min() + q[:, 0].max()) / 2, (q[:, 1].min() + q[:, 1].max()) / 2, np.ptp(q[:, 0]), np.ptp(q[:, 1])])
            if len(v) < 11 or len(v) % 2 == 0 or np.abs(bb - v[1:5]).max() >= 0.005: bad += 1; continue
            out.append(np.ptp(q[:, 0]) * w * r * np.ptp(q[:, 1]) * h * r)
    return np.array(out), bad
old = sorted(os.path.join(R, 'val/images', n) for n in os.listdir(os.path.join(R, 'val/images')))
a, bad = areas(old); e = edges_from_areas(a)
print('OLD n', len(old), 'obj', len(a), 'bad', bad, 'edges', [round(x, 4) for x in e])
new = [os.path.join(R, l.strip()) for l in open('/tmp/newsplit_val.txt') if l.strip()]
a, bad = areas(new); e = edges_from_areas(a)
cnt = [int(((a <= e[0])).sum()), int(((a > e[0]) & (a <= e[1])).sum()), int((a > e[1]).sum())]
print('NEW n', len(new), 'obj', len(a), 'bad', bad, 'edges', [round(x, 4) for x in e], 'bins', cnt, 'side', [round(x ** .5, 2) for x in e])
names = [os.path.basename(p) for p in new]; assert len(set(names)) == len(names)
s = sample_names(names, 500, 20261022)
open('/tmp/R_sample_500_v2.txt', 'w').write('\n'.join(s) + '\n'); print('sample first', s[0], 'n', len(s))
