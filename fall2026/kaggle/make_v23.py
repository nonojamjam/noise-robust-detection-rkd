"""v2.3 노트북 생성기 (make_baselines.py 복제) — 공식 iSAID 분할(사전등록 42 정정 4).
차이: 분할 목록(data_split/newsplit_*.txt)을 gzip+base64 로 내장·sha 검증, G 에 SPLIT_LISTS·EVAL_CKPT=last, 슬러그 cap-v23-<exp>-s<seed>.
사용: python make_v23.py LAM INTERP CONDS SEEDS   예) python make_v23.py 0.5 bilinear_antialias S0,S1 0
"""
import base64, gzip, hashlib, json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
import os
KAGGLE_USER = os.environ.get('KAGGLE_USERNAME', 'your-kaggle-username')  # notebook owner on Kaggle
SRC = HERE.parent / 'code' / 'seed_sweep_v2.py'
LAM = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
INTERP = sys.argv[2] if len(sys.argv) > 2 else 'bilinear_antialias'
b = SRC.read_bytes(); sha = hashlib.sha256(b).hexdigest(); B64 = base64.b64encode(b).decode()
LISTS = {k: (HERE.parent / 'data_split' / f'newsplit_{k}.txt').read_bytes() for k in ('train', 'val')}
LSHA = {k: hashlib.sha256(v).hexdigest() for k, v in LISTS.items()}
LB64 = {k: base64.b64encode(gzip.compress(v, mtime=0)).decode() for k, v in LISTS.items()}
assert [len(v.splitlines()) for v in LISTS.values()] == [18732, 6048]

def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}

import sys as _s
CONDS = _s.argv[3].split(',') if len(_s.argv) > 3 else ['A_NOISY']
SEEDS = [int(x) for x in _s.argv[4].split(',')] if len(_s.argv) > 4 else [0, 1, 2]
for exp in CONDS:
    for seed in SEEDS:
        slug = f"cap-v23-{exp.lower().replace('_', '')}-s{seed}"
        cfg = {'EXP': exp, 'SEED': seed, 'EPOCHS': 10, 'FRACTION': 1.0, 'WORKERS': 8,
               'LAM_SP_SHARED': LAM, 'INTERP_MODE': INTERP,
               'PROJECT': '/kaggle/working/seed_sweep', '__name__': '__run__',
               '__SRC_PATH__': '/kaggle/working/seed_sweep_v2.py',
               'SPLIT_LISTS': {'train': '/kaggle/working/split/newsplit_train.txt', 'val': '/kaggle/working/split/newsplit_val.txt'},
               'EVAL_CKPT': 'last'}
        cells = [
            {"cell_type": "markdown", "metadata": {}, "source": f"# {exp} seed {seed} — 10에폭, 공식 iSAID 분할 (사전등록 42 정정 4)\n코드 sha256 {sha}\n분할 목록 sha256 train {LSHA['train']} / val {LSHA['val']}"},
            code("!pip install -q ultralytics==8.4.39 2>&1 | tail -2\nprint('install done')"),
            code("import subprocess, torch\nprint(subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv'],capture_output=True,text=True).stdout)"),
            code(f"import base64, hashlib, pathlib\nB64 = \"{B64}\"\np = pathlib.Path('/kaggle/working/seed_sweep_v2.py'); p.write_bytes(base64.b64decode(B64))\nh = hashlib.sha256(p.read_bytes()).hexdigest(); print(h)\nassert h == '{sha}', '코드가 다르게 실렸다'"),
            code("import base64, gzip, hashlib, pathlib\nd = pathlib.Path('/kaggle/working/split'); d.mkdir(exist_ok=True)\n" + ''.join(f"b = gzip.decompress(base64.b64decode('{LB64[k]}')); (d / 'newsplit_{k}.txt').write_bytes(b)\nassert hashlib.sha256(b).hexdigest() == '{LSHA[k]}', '분할 목록 {k} 이 다르게 실렸다'\n" for k in ('train', 'val')) + "print('split lists ok')"),
            code(f"G = {json.dumps(cfg)}\nexec(compile(open(G['__SRC_PATH__']).read(), G['__SRC_PATH__'], 'exec'), G)\nprint('EXP', G['EXP'], 'SEED', G['SEED'], 'LAM_SP', G['LAM_SP'], 'NORM', G['TEACHER_NORM'], 'SHUFFLE', G['SHUFFLE'], 'INTERP', G['INTERP_MODE'])"),
            code("import time, json, traceback, torch, pathlib\ntorch.cuda.reset_peak_memory_stats()\nt0 = time.time()\ntry:\n    res = G['run']()\nexcept Exception as e:\n    res = {'error': f'{type(e).__name__}: {e}', 'tb': traceback.format_exc()}; print(res['tb'])\nres['minutes'] = round((time.time()-t0)/60, 1)\nres['peak_reserved_gib'] = round(torch.cuda.max_memory_reserved()/2**30, 2)\n" + f"pathlib.Path('/kaggle/working/{exp.lower()}_s{seed}_result.json').write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str))\n" + "print(json.dumps({k:v for k,v in res.items() if k!='tb'}, ensure_ascii=False, indent=2, default=str))"),
        ]
        nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}, "nbformat": 4, "nbformat_minor": 5}
        d = HERE / slug; d.mkdir(exist_ok=True)
        (d / f"{slug}.ipynb").write_text(json.dumps(nb, ensure_ascii=False))
        meta = {"id": f"{KAGGLE_USER}/{slug}", "title": slug, "code_file": f"{slug}.ipynb", "language": "python",
                "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
                "enable_internet": True, "keywords": ["gpu"],
                "dataset_sources": ["redzapdos123/isaid-dataset-yolo11-seg-format"],
                "kernel_sources": [], "competition_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4"}
        (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print('sha', sha, 'lam', LAM, 'interp', INTERP, 'lists', LSHA)
