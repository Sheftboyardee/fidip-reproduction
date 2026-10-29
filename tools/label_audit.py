"""Round 12, experiment 1: flag suspicious training labels for human adjudication.

Inputs: the 200 labelled real training images (data/syrip_cv/real200_validate.json, area = bbox w*h)
and one or more prediction files on them. The primary one should be the pooled out-of-fold (OOF)
predictions of the grouped CV, so that no model has seen the label it is judged against;
zero-shot models give independent second opinions.
Per joint it computes the OKS-style normalised error e = d / sqrt(area) / k (k = COCO sigma * 2)
and flags
  * swap:     the left/right partner's prediction matches this label better than its own (both models agree),
  * far:      e > --far for the OOF model and for >= half of the other models (persistent disagreement),
  * vis:      a label inside the image flagged v=0, or a confident visible prediction where v=1,
  * outside:  the label lies outside its own bbox.
It also draws a seeded random sample of images (--random) so the review is not limited to
model-selected cases. It writes <out>/candidates.json, one crop per flagged image and
<out>/review.html, a local page (SyRIP images: never publish it) where a human accepts or rejects each
proposal (or types a corrected x,y) and downloads decisions.json for tools/apply_label_corrections.py.
Usage: python tools/label_audit.py --gt data/syrip_cv/real200_validate.json \
          --pred oof=output/audit12/oof/cv-h20.json --pred zs_h=output/audit12/labels/zs_h.json \
          --images data/syrip/images/train_infant --out output/audit12/labels
"""
import argparse
import base64
import json
import os
import random

import cv2
import numpy as np

NAMES = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow',
         'l_wrist', 'r_wrist', 'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']
PARTNER = {1: 2, 2: 1, 3: 4, 4: 3, 5: 6, 6: 5, 7: 8, 8: 7, 9: 10, 10: 9, 11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15}
SIGMAS = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
SKELETON = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
            [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4]]


def load_preds(path):
    out = {}
    for r in json.load(open(path)):
        k = np.array(r['keypoints']).reshape(17, 3)
        if r['image_id'] not in out or r['score'] > out[r['image_id']][1]:
            out[r['image_id']] = (k, r['score'])
    return {i: v[0] for i, v in out.items()}


def draw(img, kp, color, vis_only=True):
    for a, b in SKELETON:
        if (not vis_only or (kp[a, 2] > 0 and kp[b, 2] > 0)):
            cv2.line(img, tuple(int(v) for v in kp[a, :2]), tuple(int(v) for v in kp[b, :2]), color, 2, cv2.LINE_AA)
    for j in range(17):
        if not vis_only or kp[j, 2] > 0:
            cv2.circle(img, (int(kp[j, 0]), int(kp[j, 1])), 4, color, -1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', required=True)
    ap.add_argument('--pred', action='append', required=True, help='name=path; the first is the primary (OOF)')
    ap.add_argument('--images', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--far', type=float, default=3.0, help='normalised error threshold (3 ~ OKS 0.01)')
    ap.add_argument('--conf', type=float, default=0.5)
    ap.add_argument('--random', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--controls-page', default='review_controls.html',
                    help='separate page for the random sample: plain crop + named, L/R-coloured labels only')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    gt = json.load(open(args.gt))
    preds = []
    for p in args.pred:
        name, path = p.split('=', 1)
        preds.append((name, load_preds(path)))
    prim_name, prim = preds[0]
    others = preds[1:]
    images = {im['id']: im for im in gt['images']}

    cands = []
    for a in gt['annotations']:
        img_id = a['image_id']
        g = np.array(a['keypoints'], dtype=float).reshape(17, 3)
        x, y, w, h = a['bbox']
        im = images[img_id]
        scale = np.sqrt(a['area'])
        flags = []
        P = [prim.get(img_id)] + [o.get(img_id) for _, o in others]
        if P[0] is None:
            continue

        def nerr(pk, j, gj):
            return np.linalg.norm(pk[j, :2] - g[gj, :2]) / scale / (2 * SIGMAS[gj])
        for j in range(17):
            labelled = g[j, 2] > 0
            if labelled:
                e = [nerr(p, j, j) for p in P if p is not None]
                if j in PARTNER:
                    es = [nerr(p, PARTNER[j], j) for p in P if p is not None]
                    if all(s < 1.0 and s < 0.5 * own for s, own in zip(es, e)):
                        flags.append({'joint': j, 'kind': 'swap', 'err': e[0], 'proposal': 'swap with ' + NAMES[PARTNER[j]],
                                      'xy': P[0][PARTNER[j], :2].tolist()})
                        continue
                far_others = sum(v > args.far for v in e[1:])
                if e[0] > args.far and far_others * 2 >= len(e) - 1:
                    cons = np.mean([p[j, :2] for p in P if p is not None], 0)
                    flags.append({'joint': j, 'kind': 'far', 'err': e[0], 'proposal': 'move to model consensus',
                                  'xy': cons.tolist()})
                if not (x - 1 <= g[j, 0] <= x + w + 1 and y - 1 <= g[j, 1] <= y + h + 1):
                    flags.append({'joint': j, 'kind': 'outside', 'err': e[0], 'proposal': 'check bbox / point',
                                  'xy': g[j, :2].tolist()})
                if g[j, 2] == 1 and P[0][j, 2] > 0.85 and e[0] < 1.0:
                    flags.append({'joint': j, 'kind': 'vis', 'err': e[0], 'proposal': 'v=1 (occluded) but confidently visible?',
                                  'xy': g[j, :2].tolist()})
            elif 0 <= g[j, 0] < im['width'] and 0 <= g[j, 1] < im['height'] and (g[j, 0] or g[j, 1]):
                flags.append({'joint': j, 'kind': 'vis', 'err': 0.0, 'proposal': 'v=0 but has coordinates',
                              'xy': g[j, :2].tolist()})
        cands.append({'image_id': img_id, 'file_name': im['file_name'], 'flags': flags,
                      'gt': g.tolist(), 'pred': {n: (p.get(img_id).tolist() if p.get(img_id) is not None else None)
                                                 for n, p in preds}})
    flagged = [c for c in cands if c['flags']]
    rnd = random.Random(args.seed).sample([c for c in cands if not c['flags']],
                                          min(args.random, len([c for c in cands if not c['flags']])))
    for c in rnd:
        c['random_sample'] = True
    review = flagged + rnd
    json.dump(review, open(os.path.join(args.out, 'candidates.json'), 'w'), indent=0)
    kinds = {}
    for c in flagged:
        for f in c['flags']:
            kinds[f['kind']] = kinds.get(f['kind'], 0) + 1
    print(f'{len(flagged)} flagged images, {sum(len(c["flags"]) for c in flagged)} flags {kinds}; '
          f'{len(rnd)} random unflagged images added')

    # crops: GT green, primary prediction red, flagged joints circled yellow
    cards = []
    for c in review:
        img = cv2.imread(os.path.join(args.images, c['file_name']))
        g = np.array(c['gt'])
        p = np.array(c['pred'][prim_name])
        over = img.copy()
        draw(over, g, (0, 200, 0))
        draw(over, p, (0, 0, 230), vis_only=False)
        named = draw_named(img.copy(), g)
        for f in c['flags']:
            jx, jy = g[f['joint'], :2]
            cv2.circle(over, (int(jx), int(jy)), 14, (0, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(over, NAMES[f['joint']], (int(jx) + 12, int(jy) - 8), 0, 0.6, (0, 220, 255), 2)
        vis = g[g[:, 2] > 0]
        x0, y0 = np.maximum(vis[:, :2].min(0) - 60, 0).astype(int)
        x1, y1 = np.minimum(vis[:, :2].max(0) + 60, [img.shape[1] - 1, img.shape[0] - 1]).astype(int)
        cards.append((c, enc(over, x0, y0, x1, y1), enc(img, x0, y0, x1, y1), enc(named, x0, y0, x1, y1)))
    write_html(os.path.join(args.out, 'review.html'), [cd[:2] for cd in cards if cd[0]['flags']], prim_name)
    write_controls(os.path.join(args.out, args.controls_page), [cd for cd in cards if cd[0].get('random_sample')])


def enc(img, x0, y0, x1, y1, size=520):
    crop = img[y0:y1, x0:x1]
    s = size / max(crop.shape[:2])
    ok, buf = cv2.imencode('.jpg', cv2.resize(crop, None, fx=s, fy=s), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf).decode()


SHORT = ['nose', 'L eye', 'R eye', 'L ear', 'R ear', 'L sho', 'R sho', 'L elb', 'R elb',
         'L wri', 'R wri', 'L hip', 'R hip', 'L kne', 'R kne', 'L ank', 'R ank']
SIDE = {'L': (255, 120, 0), 'R': (0, 140, 255), 'n': (255, 255, 255)}  # BGR: left blue, right orange


def draw_named(img, kp):
    """Labels only: each joint named, the infant's left side blue, right side orange; v=1 drawn hollow."""
    t = max(1, round(max(img.shape[:2]) / 700))
    for a, b in SKELETON:
        if kp[a, 2] > 0 and kp[b, 2] > 0:
            col = SIDE[SHORT[a][0]] if SHORT[a][0] == SHORT[b][0] else (200, 200, 200)
            cv2.line(img, tuple(int(v) for v in kp[a, :2]), tuple(int(v) for v in kp[b, :2]), col, t, cv2.LINE_AA)
    for j in range(17):
        if kp[j, 2] > 0:
            col = SIDE[SHORT[j][0]]
            pt = (int(kp[j, 0]), int(kp[j, 1]))
            cv2.circle(img, pt, 4 * t, col, -1 if kp[j, 2] == 2 else t, cv2.LINE_AA)
            for c, w in (((0, 0, 0), 3 * t), (col, t)):
                cv2.putText(img, SHORT[j], (pt[0] + 6 * t, pt[1] - 4 * t), 0, 0.45 * t, c, w, cv2.LINE_AA)
    return img


def write_controls(path, cards):
    rows = []
    for k, (c, _, plain, named) in enumerate(cards):
        i = c['image_id']
        rows.append(f'''<section><h3>#{k + 1} {c['file_name']} (id {i})</h3>
<img src="data:image/jpeg;base64,{plain}"><img src="data:image/jpeg;base64,{named}">
<div class="flag random" data-id="{i}_random" data-img="{i}" data-joint="-1" data-kind="random" data-xy="">
<label><input type="radio" name="{i}_r" value="ok"> labels OK</label>
<label><input type="radio" name="{i}_r" value="problem"> problem</label>
<input class="why" placeholder="which joint / what is wrong" size="40"></div></section>''')
    html = f'''<!doctype html><html><head><meta charset="utf-8"><title>SyRIP label audit: controls</title>
<style>body{{font:14px system-ui;margin:16px;max-width:1120px}}section{{display:flex;gap:12px;border-top:1px solid #ccc;padding:12px 0;flex-wrap:wrap}}
h3{{width:100%;margin:0}}img{{max-width:520px;width:49%}}.flag{{width:100%;padding:6px;background:#f5f5f5;border-radius:6px}}
label{{margin-right:8px}}#bar{{position:sticky;top:0;background:#fff;padding:8px 0;z-index:1;border-bottom:1px solid #ccc}}
.L{{color:#0078ff;font-weight:bold}}.R{{color:#ff8c00;font-weight:bold}}</style></head><body>
<div id="bar"><b>SyRIP label audit: 20 random unflagged images</b> (control sample). Left: plain image. Right: the stored labels only &mdash;
<span class="L">blue = infant's LEFT</span>, <span class="R">orange = infant's RIGHT</span>, filled dot = visible (v=2), hollow = occluded (v=1), no dot = unlabelled.
Check that each named point sits on that joint and on the correct side of the body.
Local file with private SyRIP images: do not upload or share. <button onclick="dl()">Download decisions_controls.json</button> <span id="n"></span></div>
{''.join(rows)}
<script>
function collect(){{const out=[];document.querySelectorAll('.flag').forEach(f=>{{const r=f.querySelector('input[type=radio]:checked');
const why=f.querySelector('.why');if(!r&&!why.value)return;
out.push({{id:f.dataset.id,image_id:+f.dataset.img,joint:-1,kind:'random',proposal_xy:'',decision:r?r.value:'problem',xy:'',reason:why.value}});}});return out;}}
function dl(){{const b=new Blob([JSON.stringify(collect(),null,1)],{{type:'application/json'}});const a=document.createElement('a');
a.href=URL.createObjectURL(b);a.download='decisions_controls.json';a.click();}}
document.addEventListener('change',()=>{{document.getElementById('n').textContent=collect().length+' / {len(cards)} reviewed';}});
</script></body></html>'''
    open(path, 'w', encoding='utf-8').write(html)


def write_html(path, cards, prim_name):
    rows = []
    for k, (c, b64) in enumerate(cards):
        items = []
        for f_i, f in enumerate(c['flags']):
            fid = f'{c["image_id"]}_{f["joint"]}_{f["kind"]}'
            items.append(f'''<div class="flag" data-id="{fid}" data-img="{c['image_id']}" data-joint="{f['joint']}"
 data-kind="{f['kind']}" data-xy="{f['xy'][0]:.1f},{f['xy'][1]:.1f}">
<b>{NAMES[f['joint']]}</b> <span class="kind">{f['kind']}</span> err {f['err']:.1f} &middot; {f['proposal']}
<label><input type="radio" name="{fid}" value="accept"> accept</label>
<label><input type="radio" name="{fid}" value="reject"> reject (keep label)</label>
<label><input type="radio" name="{fid}" value="ambiguous"> ambiguous</label>
<input class="xy" placeholder="or x,y" size="10"> <input class="why" placeholder="reason" size="24"></div>''')
        extra = '<div class="flag random" data-id="%s_random" data-img="%s" data-joint="-1" data-kind="random" data-xy="">' \
                'random sample: <label><input type="radio" name="%s_r" value="ok"> labels OK</label> ' \
                '<label><input type="radio" name="%s_r" value="problem"> problem</label> ' \
                '<input class="why" placeholder="which joint / why" size="30"></div>' % ((c['image_id'],) * 4) \
            if c.get('random_sample') else ''
        rows.append(f'''<section><h3>#{k + 1} {c['file_name']} (id {c['image_id']})</h3>
<img src="data:image/jpeg;base64,{b64}"><div class="flags">{''.join(items)}{extra}</div></section>''')
    html = f'''<!doctype html><html><head><meta charset="utf-8"><title>SyRIP label audit</title>
<style>body{{font:14px system-ui;margin:16px;max-width:1100px}}section{{display:flex;gap:16px;border-top:1px solid #ccc;padding:12px 0;flex-wrap:wrap}}
h3{{width:100%;margin:0}}img{{max-width:520px}}.flags{{flex:1;min-width:300px}}.flag{{padding:6px;margin:4px 0;background:#f5f5f5;border-radius:6px}}
.kind{{background:#333;color:#fff;border-radius:4px;padding:0 5px}}label{{margin-right:8px}}#bar{{position:sticky;top:0;background:#fff;padding:8px 0;z-index:1}}</style></head><body>
<div id="bar"><b>SyRIP training-label audit</b> &mdash; green = label, red = {prim_name} prediction, yellow circle = flagged joint.
Local file with private SyRIP images: do not upload or share. <button onclick="dl()">Download decisions.json</button> <span id="n"></span></div>
{''.join(rows)}
<script>
function collect(){{const out=[];document.querySelectorAll('.flag').forEach(f=>{{const r=f.querySelector('input[type=radio]:checked');
const xy=f.querySelector('.xy'),why=f.querySelector('.why');if(!r&&!(xy&&xy.value))return;
out.push({{id:f.dataset.id,image_id:+f.dataset.img,joint:+f.dataset.joint,kind:f.dataset.kind,proposal_xy:f.dataset.xy,
decision:r?r.value:'edit',xy:xy?xy.value:'',reason:why?why.value:''}});}});return out;}}
function dl(){{const b=new Blob([JSON.stringify(collect(),null,1)],{{type:'application/json'}});const a=document.createElement('a');
a.href=URL.createObjectURL(b);a.download='decisions.json';a.click();}}
document.addEventListener('change',()=>{{document.getElementById('n').textContent=collect().length+' decisions';}});
</script></body></html>'''
    open(path, 'w', encoding='utf-8').write(html)


if __name__ == '__main__':
    main()
