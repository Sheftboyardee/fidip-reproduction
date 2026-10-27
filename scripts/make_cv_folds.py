"""Round 12, experiment 0: grouped K-fold splits of the 200 labelled real training images.

Images directly linked by the overlap audit (same YouTube video, near-identical hash, or same
scene by ORB matching; scripts/audit_overlap.py -> pairs.csv) always land in the same fold. Only
train-train links are used: chaining through test/unlabelled images would merge unrelated videos. Each fold
k gets a DATASET.ROOT of its own, used with the unchanged training/eval code:
  <out>/f<k>/annotations/person_keypoints_train_infant.json     other folds' real + all 1000 synthetic
  <out>/f<k>/annotations/person_keypoints_heldout_infant.json   held-out real images, area = bbox w*h
  <out>/f<k>/images/train_infant                               -> link to the SyRIP train images
Train with DATASET.ROOT <out>/f<k>/ DATASET.TEST_SET heldout_infant (lib/dataset/syrip.py reads
heldout_infant images from images/train_infant).
The shipped train json gives every annotation a placeholder area (14233), so OKS on the held-out
images uses area = bbox w*h, as in our corrected Test500 file.
The derived annotations are SyRIP data: they stay under data/ (gitignored), never committed.
Usage: python scripts/make_cv_folds.py [--pairs output/audit12/overlap/pairs.csv] [--k 5]
       [--out data/syrip_cv] [--no-links]
"""
import argparse
import collections
import copy
import csv
import json
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/syrip')
    ap.add_argument('--pairs', default='output/audit12/overlap/pairs.csv')
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='data/syrip_cv')
    ap.add_argument('--no-links', action='store_true', help='write jsons only (e.g. on Windows)')
    args = ap.parse_args()

    full = json.load(open(os.path.join(args.root, 'annotations/person_keypoints_train_infant.json')))
    real = [im for im in full['images'] if not im['is_synthetic']]
    assert len(real) == 200, len(real)

    parent = {im['file_name']: im['file_name'] for im in real}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    for row in csv.DictReader(open(args.pairs)):
        if row['set_a'] == row['set_b'] == 'train200R':
            parent[find(row['a'])] = find(row['b'])
    groups = collections.defaultdict(list)
    for im in real:
        groups[find(im['file_name'])].append(im['id'])
    # largest groups first, each to the currently smallest fold (ties broken by a seeded shuffle)
    keys = list(groups)
    random.Random(args.seed).shuffle(keys)
    keys.sort(key=lambda g: -len(groups[g]))
    folds = [[] for _ in range(args.k)]
    for g in keys:
        min(folds, key=len).extend(groups[g])
    print('fold sizes:', [len(f) for f in folds], 'groups:', len(groups),
          'largest:', sorted((len(v) for v in groups.values()), reverse=True)[:5])

    anns = {a['image_id']: a for a in full['annotations']}
    assign = {}
    for k, ids in enumerate(folds):
        held = set(ids)
        for i in ids:
            assign[i] = k
        tr = copy.deepcopy(full)
        tr['images'] = [im for im in full['images'] if im['id'] not in held]
        tr['annotations'] = [a for a in full['annotations'] if a['image_id'] not in held]
        va = copy.deepcopy(full)
        va['images'] = [im for im in full['images'] if im['id'] in held]
        va['annotations'] = [copy.deepcopy(anns[i]) for i in sorted(held)]
        for a in va['annotations']:
            a['area'] = a['bbox'][2] * a['bbox'][3]
        d = os.path.join(args.out, f'f{k}')
        os.makedirs(os.path.join(d, 'annotations'), exist_ok=True)
        json.dump(tr, open(os.path.join(d, 'annotations/person_keypoints_train_infant.json'), 'w'))
        json.dump(va, open(os.path.join(d, 'annotations/person_keypoints_heldout_infant.json'), 'w'))
        if not args.no_links:
            os.makedirs(os.path.join(d, 'images'), exist_ok=True)
            src = os.path.relpath(os.path.join(args.root, 'images/train_infant'), os.path.join(d, 'images'))
            link = os.path.join(d, 'images', 'train_infant')
            if not os.path.lexists(link):
                os.symlink(src, link)
    # all 200 real images with the corrected area, for pooled out-of-fold scoring
    allv = copy.deepcopy(full)
    allv['images'] = real
    allv['annotations'] = [copy.deepcopy(anns[im['id']]) for im in real]
    for a in allv['annotations']:
        a['area'] = a['bbox'][2] * a['bbox'][3]
    json.dump(allv, open(os.path.join(args.out, 'real200_validate.json'), 'w'))
    json.dump({str(i): k for i, k in assign.items()}, open(os.path.join(args.out, 'folds.json'), 'w'))


if __name__ == '__main__':
    main()
