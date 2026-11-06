"""Round 13b: compare Test500 predictions from reloaded weights with the end-of-training predictions.

Prints per-coordinate differences in pixels (all 17 keypoints of each image) and the keypoint
confidence difference. AP per subset comes from scripts/bootstrap_ap.py on the same two files.
Usage: python scripts/compare_reload.py --a original.json --b reload.json
"""
import argparse
import json

import numpy as np


def load(path):
    rows = json.load(open(path))
    out = {r['image_id']: r for r in rows}
    assert len(out) == len(rows), f'{path}: more than one prediction per image'
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--a', required=True)
    p.add_argument('--b', required=True)
    args = p.parse_args()
    a, b = load(args.a), load(args.b)
    assert a.keys() == b.keys(), 'image ids differ'
    ka = np.array([a[i]['keypoints'] for i in sorted(a)], float).reshape(len(a), 17, 3)
    kb = np.array([b[i]['keypoints'] for i in sorted(a)], float).reshape(len(a), 17, 3)
    d = np.linalg.norm(ka[..., :2] - kb[..., :2], axis=-1)
    c = np.abs(ka[..., 2] - kb[..., 2])
    print(f'images {len(a)}  keypoints {d.size}')
    print(f'pixel distance: median {np.median(d):.3f} mean {d.mean():.3f} p99 {np.percentile(d, 99):.3f} '
          f'max {d.max():.3f}; exactly equal {np.mean(d == 0):.1%}; over 1 px {np.mean(d > 1):.2%}')
    print(f'confidence abs diff: mean {c.mean():.5f} max {c.max():.5f}')


if __name__ == '__main__':
    main()
