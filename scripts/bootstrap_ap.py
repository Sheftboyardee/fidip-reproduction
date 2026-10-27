"""Round 12 evaluation contract: AP per subset plus paired bootstrap CIs (PLAN.md round 12, sec. 3.2).

Scores one or two COCO keypoint result jsons on the corrected Test500 annotations and the subsets
  test100, test400, and the audit's source-independent subsets test400_indep / test400_linked
  (test images with no / some direct link to a labelled training image, scripts/audit_overlap.py).
For two result files (A = reference, B = candidate) it adds a paired bootstrap of AP(B) - AP(A):
the same resampled images are scored for both, and AP is recomputed on every resample (no
per-image AP averaging). Resampling is by source group (same YouTube video / near-duplicate /
same scene among test images, from the audit's pairs.csv), falling back to single images.
Matching (OKS, COCOeval.evaluate) is done once; each resample re-runs only the accumulation for
area 'all', maxDets 20, which reproduces COCOeval's AP exactly (checked against stats[0]).
The intervals quantify evaluation-sample uncertainty only; they do not correct for Test400 having
been reused to choose methods (see PLAN.md round 12).
Usage: python scripts/bootstrap_ap.py --a ref.json [--b cand.json] [--n 2000] [--tag-a A --tag-b B]
       [--gt <gt json>] [--subsets output/audit12/overlap/test_subsets.json]
       [--pairs output/audit12/overlap/pairs.csv] [--csv out.csv]
"""
import argparse
import collections
import contextlib
import csv
import io
import json
import os

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

GT500 = 'data/syrip500/annotations/person_keypoints_validate_infant.json'
GT100 = 'data/syrip/annotations/validate100/person_keypoints_validate_infant.json'


def per_image_eval(gt, res_file):
    """COCOeval matching once; returns {img_id: evalImg dict} for area 'all', maxDets 20."""
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(res_file)
        ev = COCOeval(gt, dt, 'keypoints')
        ev.params.useSegm = None
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    I = len(ev.params.imgIds)
    # evalImgs is ordered [cat][areaRng][img]; area 'all' is index 0, one category
    out = {img: e for img, e in zip(ev.params.imgIds, ev.evalImgs[:I])}
    return out, ev


def ap_from(evals, img_ids, iou_thrs, rec_thrs, max_det=20):
    E = [evals[i] for i in img_ids if evals.get(i) is not None]
    if not E:
        return float('nan')
    scores = np.concatenate([e['dtScores'][:max_det] for e in E])
    inds = np.argsort(-scores, kind='mergesort')
    dtm = np.concatenate([e['dtMatches'][:, :max_det] for e in E], axis=1)[:, inds]
    dtig = np.concatenate([e['dtIgnore'][:, :max_det] for e in E], axis=1)[:, inds]
    gtig = np.concatenate([e['gtIgnore'] for e in E])
    npig = np.count_nonzero(gtig == 0)
    if npig == 0:
        return float('nan')
    tps = np.logical_and(dtm, np.logical_not(dtig))
    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtig))
    tp_sum = np.cumsum(tps, axis=1).astype(float)
    fp_sum = np.cumsum(fps, axis=1).astype(float)
    precs = []
    for tp, fp in zip(tp_sum, fp_sum):
        rc = tp / npig
        pr = (tp / (fp + tp + np.spacing(1))).tolist()
        for k in range(len(pr) - 1, 0, -1):
            if pr[k] > pr[k - 1]:
                pr[k - 1] = pr[k]
        q = np.zeros(len(rec_thrs))
        idx = np.searchsorted(rc, rec_thrs, side='left')
        for ri, pi in enumerate(idx):
            if pi < len(pr):
                q[ri] = pr[pi]
        precs.append(q)
    return float(np.mean(precs))


def source_groups(img_ids, pairs_file):
    parent = {i: i for i in img_ids}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    if pairs_file and os.path.exists(pairs_file):
        for r in csv.DictReader(open(pairs_file)):
            if r['set_a'].startswith('test') and r['set_b'].startswith('test'):
                a, b = int(r['a'][4:-4]), int(r['b'][4:-4])
                if a in parent and b in parent:
                    parent[find(a)] = find(b)
    g = collections.defaultdict(list)
    for i in img_ids:
        g[find(i)].append(i)
    return list(g.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True)
    ap.add_argument('--b')
    ap.add_argument('--tag-a', default='A')
    ap.add_argument('--tag-b', default='B')
    ap.add_argument('--gt', default=GT500)
    ap.add_argument('--gt100', default=GT100)
    ap.add_argument('--subsets', default='output/audit12/overlap/test_subsets.json')
    ap.add_argument('--pairs', default='output/audit12/overlap/pairs.csv')
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--csv')
    args = ap.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(args.gt)
    all_ids = sorted(gt.getImgIds())
    subsets = {'all': all_ids}
    if args.gt100 and os.path.exists(args.gt100):
        t100 = {im['id'] for im in json.load(open(args.gt100))['images']}
        subsets['test400'] = [i for i in all_ids if i not in t100]
        subsets['test100'] = [i for i in all_ids if i in t100]
    if args.subsets and os.path.exists(args.subsets):
        for k, v in json.load(open(args.subsets)).items():
            if k.startswith('test400'):
                subsets[k] = v

    ev_a, coco_a = per_image_eval(gt, args.a)
    p = coco_a.params
    assert abs(ap_from(ev_a, all_ids, p.iouThrs, p.recThrs) - coco_a.stats[0]) < 1e-9, 'AP mismatch'
    ev_b = per_image_eval(gt, args.b)[0] if args.b else None

    rng = np.random.default_rng(args.seed)
    rows = []
    for name, ids in subsets.items():
        groups = source_groups(ids, args.pairs)
        a0 = ap_from(ev_a, ids, p.iouThrs, p.recThrs)
        row = {'subset': name, 'n_img': len(ids), 'n_groups': len(groups), args.tag_a: a0}
        if ev_b is not None:
            b0 = ap_from(ev_b, ids, p.iouThrs, p.recThrs)
            diffs = []
            for _ in range(args.n):
                pick = rng.integers(0, len(groups), len(groups))
                s = [i for g in pick for i in groups[g]]
                diffs.append(ap_from(ev_b, s, p.iouThrs, p.recThrs) - ap_from(ev_a, s, p.iouThrs, p.recThrs))
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            row.update({args.tag_b: b0, 'diff': b0 - a0, 'ci_lo': lo, 'ci_hi': hi,
                        'p_le0': float(np.mean(np.array(diffs) <= 0))})
        rows.append(row)
        print(' '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}' for k, v in row.items()))
    if args.csv:
        new = not os.path.exists(args.csv)
        with open(args.csv, 'a', newline='') as f:
            w = csv.writer(f)
            if new:
                w.writerow(['tag_a', 'tag_b', 'subset', 'n_img', 'n_groups', 'ap_a', 'ap_b', 'diff', 'ci_lo', 'ci_hi', 'p_le0'])
            for r in rows:
                w.writerow([args.tag_a, args.tag_b if args.b else '', r['subset'], r['n_img'], r['n_groups'],
                            f"{r[args.tag_a]:.4f}", f"{r.get(args.tag_b, float('nan')):.4f}",
                            f"{r.get('diff', float('nan')):.4f}", f"{r.get('ci_lo', float('nan')):.4f}",
                            f"{r.get('ci_hi', float('nan')):.4f}", f"{r.get('p_le0', float('nan')):.4f}"])


if __name__ == '__main__':
    main()
