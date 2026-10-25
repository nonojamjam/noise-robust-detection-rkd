"""Kaggle iSAID train∪val 타일을 공식 iSAID 분할(영상 ID)로 재배정하기 위한 감사. 쓰기 없음(목록 파일만)."""
import hashlib, os, sys, collections
R = os.path.expanduser('~/datasets/isaid_kaggle/iSAIDYolo11Seg')
offval = set(open('/tmp/official_val_ids.txt').read().split()); offtr = set(open('/tmp/official_train_ids.txt').read().split())
pid = lambda n: n.split('_')[0].split('.')[0]
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
tiles = collections.defaultdict(list)
for s in ('train', 'val'):
    for n in os.listdir(f'{R}/{s}/images'):
        tiles[n].append(s)
dup = [n for n, v in tiles.items() if len(v) == 2]
bad = 0
for n in dup:
    l = os.path.splitext(n)[0] + '.txt'
    if md5(f'{R}/train/images/{n}') != md5(f'{R}/val/images/{n}') or md5(f'{R}/train/labels/{l}') != md5(f'{R}/val/labels/{l}'):
        bad += 1
print('unique tiles', len(tiles), 'dup', len(dup), 'dup not identical', bad)
new = {'train': [], 'val': []}
for n, v in sorted(tiles.items()):
    p = pid(n); src = 'train' if 'train' in v else 'val'
    new['val' if p in offval else 'train' if p in offtr else 'ERR'].append(f'{src}/images/{n}')
for k, v in new.items():
    open(f'/tmp/newsplit_{k}.txt', 'w').write('\n'.join(v) + '\n')
    print(k, 'tiles', len(v), 'images', len({pid(x.split('/')[-1]) for x in v}))
assert not set(pid(x.split('/')[-1]) for x in new['train']) & set(pid(x.split('/')[-1]) for x in new['val'])
print('scene overlap 0')
