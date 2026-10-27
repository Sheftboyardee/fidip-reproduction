"""Round 12, experiment 0: parity audit of our ViTPose pipeline against HF transformers' reference.

The repo pipeline (lib/dataset + core.inference.get_final_preds, used for training-time validation
and scripts/eval500.sbatch) and HF's VitPoseImageProcessor (tools/test_vitpose.py) differ in
  * crop geometry: box centre x + (w - 1) / 2 and a (w - 1) span (repo) vs x + w / 2 and w (HF);
    cv2.warpAffine vs scipy affine_transform;
  * peak refinement: DARK on a cv2 11x11 Gaussian blur (sigma 2) vs HF's DARK-UDP with sigma 0.8;
  * back-projection to the image (each pipeline inverts its own crop).
1. Coordinate round trip: a known image point is encoded into a heatmap with each pipeline's
   training-target convention, rendered as a sigma-2 Gaussian and decoded; the error should be ~0.
2. AP on Test500 (Test400 / Test100) for the 2 x 2 combinations of {repo, HF} crop/back-projection
   x {repo, HF} peak refinement, from one forward pass per crop (flip test on).
Usage: python tools/parity_vitpose.py --cfg <vitpose cfg> [MODEL.HF_NAME ...] [TEST.MODEL_FILE ckpt]
       --out output/audit12/parity
"""
import argparse
import contextlib
import io
import json
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import _init_paths  # noqa: F401
from config import cfg, update_config
from core.inference import get_final_preds
from utils.transforms import flip_back, get_affine_transform, affine_transform
import dataset  # noqa: F401
import models

GT500 = 'data/syrip500/annotations/person_keypoints_validate_infant.json'
GT100 = 'data/syrip/annotations/validate100/person_keypoints_validate_infant.json'
FLIP_PAIRS = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]


def gaussian_map(h, w, x, y, sigma=2.0):
    xs, ys = np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)[:, None]
    return np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma ** 2))


def round_trip(processor, img_w, img_h, box, hm_size, pts):
    """Encode image points into heatmap coords with each convention, render, decode; return errors (px)."""
    from transformers.models.vitpose.image_processing_vitpose import box_to_center_and_scale, transform_preds
    W, H = hm_size
    # repo convention: training targets (JointsDataset) and decoding (get_final_preds)
    x, y, w, h = box
    ds = dataset.syrip.__new__(dataset.syrip)
    ds.aspect_ratio, ds.pixel_std = cfg.MODEL.IMAGE_SIZE[0] / cfg.MODEL.IMAGE_SIZE[1], 200
    c, s = ds._xywh2cs(x, y, w, h)
    t = get_affine_transform(c, s, 0, np.array(hm_size))
    out = {}
    hms = np.stack([gaussian_map(H, W, *affine_transform(p, t)) for p in pts])[None]
    dec, _ = get_final_preds(cfg, hms.copy(), np.array([c]), np.array([s]))
    out['repo'] = np.abs(dec[0] - pts).max()
    # HF convention: inverse of its transform_preds, decoded by its own post-processing
    ch, sh = box_to_center_and_scale(box, image_width=processor.size['width'], image_height=processor.size['height'])
    sc = sh * 200.0
    hx = (pts[:, 0] - ch[0] + sc[0] * 0.5) * (W - 1.0) / sc[0]
    hy = (pts[:, 1] - ch[1] + sc[1] * 0.5) * (H - 1.0) / sc[1]
    hms = np.stack([gaussian_map(H, W, a, b) for a, b in zip(hx, hy)])[None]
    dec, _ = processor.keypoints_from_heatmaps(hms.copy(), ch[None], sh[None], kernel=11)
    out['hf'] = np.abs(dec[0] - pts).max()
    # repo targets decoded by HF post-processing (the mix a model trained here meets in test_vitpose.py)
    hms = np.stack([gaussian_map(H, W, *affine_transform(p, t)) for p in pts])[None]
    dec, _ = processor.keypoints_from_heatmaps(hms.copy(), ch[None], sh[None], kernel=11)
    out['repo_targets_hf_decode'] = np.abs(dec[0] - pts).max()
    return out


def score(gt, results, ids):
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(results)
        ev = COCOeval(gt, dt, 'keypoints')
        ev.params.useSegm = None
        ev.params.imgIds = ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return ev.stats[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--out', default='output/audit12/parity')
    ap.add_argument('opts', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    args.modelDir = args.logDir = args.dataDir = args.prevModelDir = ''
    update_config(cfg, args)
    os.makedirs(args.out, exist_ok=True)
    from transformers import AutoProcessor
    from transformers.models.vitpose.image_processing_vitpose import box_to_center_and_scale
    processor = AutoProcessor.from_pretrained(cfg.MODEL.HF_NAME)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. coordinate round trip on a few boxes / points
    rng = np.random.default_rng(0)
    rt = []
    for _ in range(20):
        bw, bh = rng.uniform(100, 900), rng.uniform(100, 900)
        box = [rng.uniform(0, 300), rng.uniform(0, 300), bw, bh]
        pts = np.stack([box[0] + rng.uniform(0.2, 0.8, 5) * bw, box[1] + rng.uniform(0.2, 0.8, 5) * bh], 1)
        rt.append(round_trip(processor, 2000, 2000, box, cfg.MODEL.HEATMAP_SIZE, pts))
    rt_max = {k: float(max(r[k] for r in rt)) for k in rt[0]}
    print('round-trip max error (image px):', rt_max)

    # 2. AP of the 2 x 2 combinations
    model, _ = models.adaptive_pose_vitpose.get_adaptive_pose_net(cfg, is_train=False)
    if cfg.TEST.MODEL_FILE:
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE, map_location='cpu'), strict=True)
    model = model.to(dev).eval()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ds = dataset.syrip(cfg, 'data/syrip500/', 'validate_infant', False, transforms.Compose([transforms.ToTensor(), normalize]))
    gt_json = json.load(open(GT500))
    anns = {a['image_id']: a for a in gt_json['annotations']}

    def heatmaps(x):
        with torch.no_grad():
            _, h = model(x.to(dev))
            _, hf = model(x.flip(-1).to(dev))
        hf = flip_back(hf.float().cpu().numpy(), FLIP_PAIRS)
        return (h.float().cpu().numpy() + hf) * 0.5

    res = {k: [] for k in ['repo_crop+repo_refine', 'repo_crop+hf_refine', 'hf_crop+hf_refine', 'hf_crop+repo_refine']}
    in_diff = []
    for k in range(len(ds)):
        x, _, _, meta = ds[k]
        img_id = int(os.path.basename(meta['image'])[4:-4])
        box = anns[img_id]['bbox']
        image = Image.open(meta['image']).convert('RGB')
        xh = processor(image, boxes=[[box]], return_tensors='pt')['pixel_values']
        in_diff.append(float((xh[0] - x).abs().mean()))
        h_r, h_h = heatmaps(x[None]), heatmaps(xh)
        c_r, s_r = meta['center'][None], meta['scale'][None]
        c_h, s_h = box_to_center_and_scale(box, image_width=processor.size['width'], image_height=processor.size['height'])
        for name, hm, crop in [('repo_crop', h_r, 'repo'), ('hf_crop', h_h, 'hf')]:
            for refine in ['repo', 'hf']:
                if crop == 'repo' and refine == 'repo':
                    p, v = get_final_preds(cfg, hm.copy(), c_r, s_r)
                elif crop == 'hf' and refine == 'hf':
                    p, v = processor.keypoints_from_heatmaps(hm.copy(), c_h[None], s_h[None], kernel=11)
                elif crop == 'repo':  # HF refinement in heatmap space, repo back-projection
                    p, v = processor.keypoints_from_heatmaps(hm.copy(), np.array([[0., 0.]]), np.array([[(hm.shape[3] - 1) / 200, (hm.shape[2] - 1) / 200]]), kernel=11)
                    t = get_affine_transform(c_r[0], s_r[0], 0, [hm.shape[3], hm.shape[2]], inv=1)
                    p = np.stack([affine_transform(q + [(hm.shape[3] - 1) / 2, (hm.shape[2] - 1) / 2], t) for q in p[0]])[None]
                else:  # repo refinement in heatmap space, HF back-projection
                    # centre (W-1)/2 and a W-pixel span make the repo's affine the identity
                    p, v = get_final_preds(cfg, hm.copy(), np.array([[(hm.shape[3] - 1) / 2, (hm.shape[2] - 1) / 2]]),
                                           np.array([[hm.shape[3] / 200, hm.shape[2] / 200]]))
                    sc = s_h * 200.0
                    p = p.copy()
                    p[..., 0] = p[..., 0] * sc[0] / (hm.shape[3] - 1) + c_h[0] - sc[0] * 0.5
                    p[..., 1] = p[..., 1] * sc[1] / (hm.shape[2] - 1) + c_h[1] - sc[1] * 0.5
                v = np.asarray(v).reshape(-1)
                vis = v[v > cfg.TEST.IN_VIS_THRE]
                res[f'{name}+{refine}_refine'].append({
                    'image_id': img_id, 'category_id': 1,
                    'keypoints': np.concatenate([p[0][:, :2], v[:, None]], 1).reshape(-1).tolist(),
                    'score': float(vis.mean()) if len(vis) else 0.0})
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(GT500)
    t100 = {im['id'] for im in json.load(open(GT100))['images']}
    ids = sorted(gt.getImgIds())
    report = {'model': cfg.MODEL.HF_NAME, 'ckpt': cfg.TEST.MODEL_FILE, 'round_trip_max_px': rt_max,
              'input_mean_abs_diff': float(np.mean(in_diff)), 'ap': {}}
    for k, r in res.items():
        json.dump(r, open(os.path.join(args.out, k.replace('+', '__') + '.json'), 'w'))
        report['ap'][k] = {'test400': score(gt, r, [i for i in ids if i not in t100]),
                           'test100': score(gt, r, [i for i in ids if i in t100])}
        print(k, {s: round(a * 100, 2) for s, a in report['ap'][k].items()})
    print('mean |input difference| (normalised units):', report['input_mean_abs_diff'])
    json.dump(report, open(os.path.join(args.out, 'report.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
