"""Failure analysis of Test500 predictions: per-joint error, posture, visibility and left/right swaps.

Scores prediction files (one pose per image, GT boxes) against the Test500 labels with COCO's own
per-joint similarity ks = exp(-d^2 / (2 * area * (2 sigma)^2)), the quantity OKS averages.
  * per joint: median error in % of sqrt(box area), mean ks, share of joints with ks < 0.5 ("miss")
  * visible (v=2) vs occluded (v=1) labelled joints
  * posture (labelled on Test100 only): AP and mean OKS
  * left/right swap: exchanging a pair reduces its total error and the pair is at least 5% of
    sqrt(area) apart in the labels, so that coincident joints do not count
Aggregates go to --out (committed); per-image OKS goes to --per-image (kept private, gitignored).
Usage: python scripts/failure_analysis.py --pred name=file.json [name=file.json ...]
           [--out results/failure_analysis] [--per-image output/audit12/failure/per_image.csv]
"""
import argparse
import contextlib
import csv
import io
import json
import os

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

GT500 = 'data/syrip500/annotations/person_keypoints_validate_infant.json'
GT100 = 'data/syrip/annotations/person_keypoints_validate_infant.json'
JOINTS = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow',
          'l_wrist', 'r_wrist', 'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']
SIGMAS = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10
PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


def load_preds(path):
    return {r['image_id']: np.array(r['keypoints'], dtype=float).reshape(17, 3) for r in json.load(open(path))}


def ap(gt, path, ids):
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(gt, gt.loadRes(path), 'keypoints')
        ev.params.useSegm = None
        ev.params.imgIds = ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return ev.stats[0]


def write(fn, head, rows):
    with open(fn, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(rows)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--pred', nargs='+', required=True, help='name=path')
    ap_.add_argument('--out', default='results/failure_analysis')
    ap_.add_argument('--per-image', default='output/audit12/failure/per_image.csv')
    args = ap_.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.per_image), exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(GT500)
    anns = {a['image_id']: a for a in gt.dataset['annotations']}
    ids = sorted(anns)
    g100 = json.load(open(GT100))
    by_name = {im['file_name']: im['posture'] for im in g100['images']}
    posture = {im['id']: by_name[im['file_name']] for im in gt.dataset['images'] if im['file_name'] in by_name}
    t100 = sorted(posture)
    assert len(t100) == 100
    t400 = sorted(set(ids) - set(t100))
    kp = {i: np.array(anns[i]['keypoints'], dtype=float).reshape(17, 3) for i in ids}
    area = {i: float(anns[i]['area']) for i in ids}
    joint_rows, vis_rows, post_rows, swap_rows, img_rows = [], [], [], [], []
    for spec in args.pred:
        name, path = spec.split('=', 1)
        pr = load_preds(path)
        rows = {}
        for i in ids:
            g, p = kp[i], pr[i]
            lab = g[:, 2] > 0
            d = np.linalg.norm(p[:, :2] - g[:, :2], axis=1)
            ks = np.exp(-d ** 2 / (2 * area[i] * (2 * SIGMAS) ** 2 + np.spacing(1)))
            root = np.sqrt(area[i])
            swaps = 0
            for a, b in PAIRS:
                if lab[a] and lab[b] and np.linalg.norm(g[a, :2] - g[b, :2]) >= 0.05 * root:
                    crossed = np.linalg.norm(p[a, :2] - g[b, :2]) + np.linalg.norm(p[b, :2] - g[a, :2])
                    swaps += int(crossed < d[a] + d[b])
            err = d / root * 100
            rows[i] = dict(lab=lab, vis=g[:, 2], err=err, ks=ks, oks=ks[lab].mean(), swaps=swaps,
                           pairs=sum(int(lab[a] and lab[b]) for a, b in PAIRS))
            img_rows.append([name, i, gt.imgs[i]['file_name'], 'test100' if i in posture else 'test400',
                             f'{ks[lab].mean():.4f}', swaps, f'{np.median(err[lab]):.2f}'])
        for sub, sid in (('test100', t100), ('test400', t400)):
            for j, jn in enumerate(JOINTS):
                e = np.array([rows[i]['err'][j] for i in sid if rows[i]['lab'][j]])
                k = np.array([rows[i]['ks'][j] for i in sid if rows[i]['lab'][j]])
                joint_rows.append([name, sub, jn, len(e), f'{np.median(e):.2f}', f'{k.mean():.4f}',
                                   f'{(k < .5).mean():.4f}'])
            for v, vn in ((2, 'visible'), (1, 'occluded')):
                k = np.concatenate([rows[i]['ks'][rows[i]['vis'] == v] for i in sid])
                e = np.concatenate([rows[i]['err'][rows[i]['vis'] == v] for i in sid])
                if len(k):
                    vis_rows.append([name, sub, vn, len(k), f'{np.median(e):.2f}', f'{k.mean():.4f}',
                                     f'{(k < .5).mean():.4f}'])
            swap_rows.append([name, sub, sum(rows[i]['pairs'] for i in sid), sum(rows[i]['swaps'] for i in sid),
                              sum(rows[i]['swaps'] > 0 for i in sid), len(sid)])
        for po in ('supine', 'prone', 'sitting', 'standing'):
            sid = [i for i in t100 if posture[i] == po]
            post_rows.append([name, po, len(sid), f'{ap(gt, path, sid) * 100:.1f}',
                              f"{np.mean([rows[i]['oks'] for i in sid]):.4f}"])
    write(f'{args.out}/per_joint.csv', 'model subset joint n median_err_pct mean_ks miss_rate'.split(), joint_rows)
    write(f'{args.out}/visibility.csv', 'model subset visibility n median_err_pct mean_ks miss_rate'.split(), vis_rows)
    write(f'{args.out}/posture_test100.csv', 'model posture n ap mean_oks'.split(), post_rows)
    write(f'{args.out}/lr_swaps.csv', 'model subset pairs swapped_pairs images_with_swap images'.split(), swap_rows)
    write(args.per_image, 'model image_id file_name subset oks swaps median_err_pct'.split(), img_rows)


if __name__ == '__main__':
    main()
