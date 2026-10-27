"""Round 12: score grouped-CV runs on the 200 labelled real images (pre-registered metric).

For each method tag, the five fold runs output/<tag>-f<k>/ hold final-epoch predictions on their
held-out images (results/keypoints_heldout_infant_results_0.json, written by the last validation).
They are pooled into <out>/<tag>.json and scored on data/syrip_cv/real200_validate.json
(area = bbox w*h): pooled out-of-fold AP and per-fold AP. With --ref, the per-fold differences and
the advance rule (pooled +0.5 and better in >= 4 of 5 folds) are reported against that tag.
Usage: python scripts/cv_score.py --tags cv-h20 cv-rs [--ref cv-h20] [--out output/audit12/oof]
"""
import argparse
import contextlib
import glob
import io
import json
import os

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

GT = 'data/syrip_cv/real200_validate.json'


def ap(gt, res, ids):
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(res)
        ev = COCOeval(gt, dt, 'keypoints')
        ev.params.useSegm = None
        ev.params.imgIds = ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return ev.stats[0]


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--tags', nargs='+', required=True)
    ap_.add_argument('--ref')
    ap_.add_argument('--k', type=int, default=5)
    ap_.add_argument('--result', default='keypoints_heldout_infant_results_0.json')
    ap_.add_argument('--out', default='output/audit12/oof')
    args = ap_.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(GT)
    folds = json.load(open('data/syrip_cv/folds.json'))
    fold_ids = [[int(i) for i, k in folds.items() if k == f] for f in range(args.k)]
    scores = {}
    for tag in args.tags:
        pooled, ok = [], True
        for f in range(args.k):
            hits = glob.glob(f'output/{tag}-f{f}/syrip/*/*/results/{args.result}')
            if not hits:
                print(f'{tag}: fold {f} missing')
                ok = False
                continue
            pooled += json.load(open(hits[0]))
        if not ok:
            continue
        path = os.path.join(args.out, f'{tag}.json')
        json.dump(pooled, open(path, 'w'))
        scores[tag] = {'pooled': ap(gt, path, sorted(gt.getImgIds())),
                       'folds': [ap(gt, path, ids) for ids in fold_ids]}
        s = scores[tag]
        print(f"{tag:24s} pooled {s['pooled'] * 100:.2f}  folds " + ' '.join(f'{v * 100:.1f}' for v in s['folds']))
    if args.ref and args.ref in scores:
        r = scores[args.ref]
        for tag, s in scores.items():
            if tag == args.ref:
                continue
            d = [(a - b) * 100 for a, b in zip(s['folds'], r['folds'])]
            dp = (s['pooled'] - r['pooled']) * 100
            wins = sum(v > 0 for v in d)
            print(f"{tag} - {args.ref}: pooled {dp:+.2f}, folds " + ' '.join(f'{v:+.1f}' for v in d) +
                  f"  ({wins}/5 better) -> {'ADVANCE' if dp >= 0.5 and wins >= 4 else 'no advance'}")


if __name__ == '__main__':
    main()
