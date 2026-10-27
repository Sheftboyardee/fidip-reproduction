"""Round 12, experiment 0: provenance / near-duplicate audit of the SyRIP real images.

Compares the 200 labelled real training images (200R), the 300 unannotated real images in
images/train_infant, and the 500 Test500 images, using
  * YouTube video ids from the `image_source` field (labelled sets only),
  * a 64-bit DCT perceptual hash (near-identical frames / re-encodes / crops of the same photo),
  * ORB keypoint matching + RANSAC homography inliers (same scene, e.g. other frames of a video).
Writes <out>/pairs.csv (every flagged cross-image pair), <out>/groups.json (connected components
of the "same source" graph, used for grouped CV folds) and prints a summary.
Usage: python scripts/audit_overlap.py [--root data/syrip] [--out output/audit12/overlap]
"""
import argparse
import collections
import csv
import json
import os
import re
from multiprocessing import Pool

import cv2
import numpy as np

PHASH_THR = 10    # Hamming distance (of 64 bits) for "near-identical image"
ORB_THR = 40      # RANSAC homography inliers for "same scene"


def video_id(url):
    m = re.search(r'(?:v=|youtu\.be/)([\w-]{11})', url or '')
    return m.group(1) if m else None


def phash(img):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (32, 32), interpolation=cv2.INTER_AREA)
    d = cv2.dct(np.float32(g))[:8, :8].ravel()
    return d[1:] > np.median(d[1:])


_ORB = {}


def load(path):
    img = cv2.imread(path)
    s = 480 / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def orb_features(path):
    img = load(path)
    orb = cv2.ORB_create(nfeatures=1000)
    kp, des = orb.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
    pts = np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32)
    return phash(img), pts, des


def _init(feats):
    _ORB['f'] = feats
    _ORB['bf'] = cv2.BFMatcher(cv2.NORM_HAMMING)


def orb_inliers(pair):
    i, j = pair
    _, pi, di = _ORB['f'][i]
    _, pj, dj = _ORB['f'][j]
    if di is None or dj is None or len(di) < 10 or len(dj) < 10:
        return 0
    good = []
    for m in _ORB['bf'].knnMatch(di, dj, k=2):
        if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:
            good.append(m[0])
    if len(good) < 10:
        return 0
    src = pi[[m.queryIdx for m in good]]
    dst = pj[[m.trainIdx for m in good]]
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/syrip')
    ap.add_argument('--test500', default='data/syrip500/annotations/person_keypoints_validate_infant.json')
    ap.add_argument('--out', default='output/audit12/overlap')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    A = os.path.join(args.root, 'annotations')
    t100 = {im['id'] for im in json.load(open(os.path.join(A, 'validate100/person_keypoints_validate_infant.json')))['images']}
    r200 = json.load(open(os.path.join(A, '200R/person_keypoints_train_infant.json')))['images']
    t500 = json.load(open(args.test500))['images']
    lab = {im['file_name'] for im in json.load(open(os.path.join(A, 'person_keypoints_train_infant.json')))['images']}

    items = []  # (name, set, path, video)
    for im in r200:
        items.append((im['file_name'], 'train200R', os.path.join(args.root, 'images/train_infant', im['file_name']),
                      video_id(im['image_source'])))
    for f in sorted(os.listdir(os.path.join(args.root, 'images/train_infant'))):
        if f not in lab:
            items.append((f, 'extra300', os.path.join(args.root, 'images/train_infant', f), None))
    for im in t500:
        items.append((im['file_name'], 'test100' if im['id'] in t100 else 'test400',
                      os.path.join(args.root, 'images/validate_infant', im['file_name']),
                      video_id(im.get('image_source', ''))))
    print('images:', collections.Counter(s for _, s, _, _ in items))

    with Pool(args.workers) as p:
        feats = p.map(orb_features, [it[2] for it in items], chunksize=8)
    hashes = np.stack([f[0] for f in feats])
    ham = (hashes[:, None, :] != hashes[None, :, :]).sum(-1)

    n = len(items)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    with Pool(args.workers, initializer=_init, initargs=(feats,)) as p:
        inl = p.map(orb_inliers, pairs, chunksize=2000)

    flagged = []
    for (i, j), k in zip(pairs, inl):
        same_vid = items[i][3] is not None and items[i][3] == items[j][3]
        if ham[i, j] <= PHASH_THR or k >= ORB_THR or same_vid:
            flagged.append((i, j, int(ham[i, j]), k, same_vid))
    with open(os.path.join(args.out, 'pairs.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['a', 'set_a', 'b', 'set_b', 'phash_dist', 'orb_inliers', 'same_video'])
        for i, j, h, k, v in flagged:
            w.writerow([items[i][0], items[i][1], items[j][0], items[j][1], h, k, int(v)])

    # union-find over "same source" links -> groups for grouped CV and leakage accounting
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, j, h, k, v in flagged:
        parent[find(i)] = find(j)
    groups = collections.defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    group_of = {}
    for g, (root, members) in enumerate(sorted(groups.items(), key=lambda kv: min(kv[1]))):
        for i in members:
            group_of[items[i][1] + '/' + items[i][0]] = g
    json.dump(group_of, open(os.path.join(args.out, 'groups.json'), 'w'), indent=0)

    # summary: which sets share a source with which
    def kind(i, j, h, k, v):
        return 'phash' if h <= PHASH_THR else ('orb' if k >= ORB_THR else 'video')
    cnt = collections.Counter()
    linked = collections.defaultdict(set)
    for i, j, h, k, v in flagged:
        a, b = sorted([items[i][1], items[j][1]])
        cnt[(a, b, kind(i, j, h, k, v))] += 1
        linked[(items[i][1], items[j][1])].add(i)
        linked[(items[j][1], items[i][1])].add(j)
    print('flagged pairs by (set, set, strongest evidence):')
    for key, c in sorted(cnt.items()):
        print('  ', key, c)
    for s in ['extra300', 'test100', 'test400']:
        for t in ['train200R', 'extra300', 'test100', 'test400']:
            if s != t and (s, t) in linked:
                print(f'  {len(linked[(s, t)])} {s} images share a source with {t}')
    multi = collections.Counter(len(m) for m in groups.values())
    print('group sizes:', sorted(multi.items()))

    # test images with / without a *direct* link to a labelled training image (chaining through
    # the unlabelled images would over-merge); scored separately by scripts/bootstrap_ap.py
    trained = {i for i, j, *_ in flagged if items[j][1] == 'train200R'} | \
              {j for i, j, *_ in flagged if items[i][1] == 'train200R'}
    subsets = collections.defaultdict(list)
    for i, (name, s, _, _) in enumerate(items):
        if s.startswith('test'):
            subsets[s + ('_linked' if i in trained else '_indep')].append(int(name[4:-4]))
    json.dump({k: sorted(v) for k, v in subsets.items()}, open(os.path.join(args.out, 'test_subsets.json'), 'w'))
    print('test subsets:', {k: len(v) for k, v in sorted(subsets.items())})


if __name__ == '__main__':
    main()
