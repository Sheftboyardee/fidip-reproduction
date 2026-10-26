"""Zero-shot ViTPose (HF transformers) on SyRIP with ground-truth boxes (PLAN.md stage D1).

Writes a COCO keypoint results json that scripts/subset_ap.py scores exactly like our HRNet runs.
Mirrors the HRNet test protocol: GT boxes (annotations whose keypoints are all zero are skipped,
as in the repo's loader), flip test, and the repo's rescoring (mean of keypoint scores above
IN_VIS_THRE=0.2; box score is 1 for GT boxes).

Usage: python tools/test_vitpose.py --model usyd-community/vitpose-plus-large \
           --gt data/syrip500/annotations/person_keypoints_validate_infant.json \
           --images data/syrip500/images/validate_infant --out results.json [--no-flip]
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, VitPoseForPoseEstimation

# COCO left/right keypoint pairs (eyes, ears, shoulders, elbows, wrists, hips, knees, ankles)
FLIP_PAIRS = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
IN_VIS_THRE = 0.2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--images', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--no-flip', action='store_true')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = AutoProcessor.from_pretrained(args.model)
    model = VitPoseForPoseEstimation.from_pretrained(args.model).to(device).eval()
    # ViTPose++ checkpoints are mixture-of-experts; expert 0 is the COCO head
    moe = getattr(model.config.backbone_config, 'num_experts', 1) > 1
    flip_pairs = torch.tensor(FLIP_PAIRS, device=device)

    gt = json.load(open(args.gt))
    anns = {}
    for a in gt['annotations']:
        if max(a['keypoints']) == 0 or a['bbox'][2] <= 0 or a['bbox'][3] <= 0:
            continue
        anns.setdefault(a['image_id'], []).append(a)

    results = []
    with torch.no_grad():
        for im in gt['images']:
            if im['id'] not in anns:
                continue
            image = Image.open(os.path.join(args.images, im['file_name'])).convert('RGB')
            boxes = [a['bbox'] for a in anns[im['id']]]
            inputs = processor(image, boxes=[boxes], return_tensors='pt').to(device)
            kw = {'dataset_index': torch.zeros(len(boxes), dtype=torch.long, device=device)} if moe else {}
            out = model(**inputs, **kw)
            if not args.no_flip:
                out_f = model(inputs['pixel_values'].flip(-1), flip_pairs=flip_pairs, **kw)
                out.heatmaps = (out.heatmaps + out_f.heatmaps) / 2
            poses = processor.post_process_pose_estimation(out, boxes=[boxes])[0]
            for p in poses:
                xy = p['keypoints'].cpu().numpy()
                s = p['scores'].cpu().numpy()
                vis = s[s > IN_VIS_THRE]
                results.append({
                    'image_id': im['id'], 'category_id': 1,
                    'keypoints': np.concatenate([xy, s[:, None]], 1).reshape(-1).tolist(),
                    'score': float(vis.mean()) if len(vis) else 0.0,
                })
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(results, open(args.out, 'w'))
    print(f'{args.model}: {len(results)} poses on {len(anns)} images -> {args.out}')


if __name__ == '__main__':
    main()
