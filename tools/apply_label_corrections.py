"""Round 12, experiment 1: apply human-adjudicated label corrections (decisions.json from
output/audit12/labels/review.html) to the training annotations.

Only decisions 'accept' and 'edit' change labels; 'reject', 'ambiguous' and random-sample
verdicts are kept for the record only. Per kind:
  swap    -> exchange the joint's label with its left/right partner (both points and flags)
  far     -> move the joint to the proposed (or typed) x,y; visibility unchanged
  vis     -> set v=2 for 'v=1 but visible', or v=0 and zero coordinates for 'v=0 with coordinates'
  edit    -> move to the typed x,y
Writes the corrected train json (all synthetic images untouched) and a corrections log (csv),
which is the record for AGENT.md. Test annotations are never touched by this tool.
Usage: python tools/apply_label_corrections.py --decisions decisions.json \
          --train data/syrip/annotations/person_keypoints_train_infant.json \
          --out data/syrip_labelfix/annotations/person_keypoints_train_infant.json --log corrections.csv
       (and --train/--out for each CV fold's train json; folds' held-out jsons are not changed)
"""
import argparse
import csv
import json
import os

PARTNER = {1: 2, 2: 1, 3: 4, 4: 3, 5: 6, 6: 5, 7: 8, 8: 7, 9: 10, 10: 9, 11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--decisions', required=True)
    ap.add_argument('--train', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--log')
    args = ap.parse_args()
    dec = [d for d in json.load(open(args.decisions)) if d['decision'] in ('accept', 'edit') and d['joint'] >= 0]
    data = json.load(open(args.train))
    by_img = {a['image_id']: a for a in data['annotations']}
    log = []
    swapped = set()
    for d in dec:
        a = by_img.get(d['image_id'])
        if a is None:
            continue  # image not in this split (e.g. a CV fold's held-out image)
        kp = a['keypoints']
        j = d['joint']
        old = kp[3 * j:3 * j + 3]
        xy = d['xy'] or (d['proposal_xy'] if d['decision'] == 'accept' else '')
        if d['kind'] == 'swap' and d['decision'] == 'accept':
            pair = tuple(sorted((j, PARTNER[j])))
            if (a['image_id'], pair) in swapped:
                continue
            swapped.add((a['image_id'], pair))
            p = PARTNER[j]
            kp[3 * j:3 * j + 3], kp[3 * p:3 * p + 3] = kp[3 * p:3 * p + 3], kp[3 * j:3 * j + 3]
        elif d['kind'] == 'vis' and d['decision'] == 'accept':
            if kp[3 * j + 2] == 1:
                kp[3 * j + 2] = 2
            else:
                kp[3 * j:3 * j + 3] = [0, 0, 0]
        elif xy:
            x, y = (float(v) for v in xy.split(','))
            kp[3 * j], kp[3 * j + 1] = x, y
        else:
            continue
        log.append([a['image_id'], j, d['kind'], d['decision'], old, kp[3 * j:3 * j + 3], d.get('reason', '')])
        a['num_keypoints'] = sum(1 for k in range(17) if kp[3 * k + 2] > 0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(data, open(args.out, 'w'))
    if args.log:
        with open(args.log, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['image_id', 'joint', 'kind', 'decision', 'old', 'new', 'reason'])
            w.writerows(log)
    print(f'{len(log)} corrections applied to {args.train} -> {args.out}')


if __name__ == '__main__':
    main()
