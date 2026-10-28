"""Round 12, E4 follow-up: keypoint-level ensemble of two prediction files (same GT boxes).

For each (image_id) pose, coordinates are averaged with weights w_a, w_b (default 0.5/0.5) and
keypoint scores averaged the same way; the instance score is recomputed as in test_vitpose.py
(mean of keypoint scores above 0.2). Fixed rule, no tuning on the scored set.
Usage: python scripts/ensemble_kpts.py --a a.json --b b.json --out ens.json [--wb 0.5]
"""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--wb', type=float, default=0.5)
    args = ap.parse_args()
    b = {}
    for r in json.load(open(args.b)):
        b.setdefault(r['image_id'], []).append(r)
    out = []
    for r in json.load(open(args.a)):
        cands = b.get(r['image_id'], [])
        ka = np.array(r['keypoints']).reshape(17, 3)
        if not cands:
            out.append(r)
            continue
        # one infant per image; if several, pair with the closest pose
        kb = min((np.array(c['keypoints']).reshape(17, 3) for c in cands),
                 key=lambda k: np.abs(k[:, :2] - ka[:, :2]).mean())
        k = (1 - args.wb) * ka + args.wb * kb
        vis = k[:, 2][k[:, 2] > 0.2]
        out.append({'image_id': r['image_id'], 'category_id': 1, 'keypoints': k.reshape(-1).tolist(),
                    'score': float(vis.mean()) if len(vis) else 0.0})
    json.dump(out, open(args.out, 'w'))
    print(f'{len(out)} poses -> {args.out}')


if __name__ == '__main__':
    main()
