"""Round 12, experiment 4: zero-shot Sapiens2 pose on SyRIP with ground-truth boxes.

Runs in ~/venvs/sapiens2 (transformers 5.x; scripts/sapiens2_setup.sbatch). Same protocol as
tools/test_vitpose.py: GT boxes, flip test, rescoring = mean of keypoint scores above 0.2 (box score 1).
The 308 Goliath keypoints are mapped to COCO-17 by name (sapiens/pose/configs/_base_/keypoints308.py):
  COCO  nose eyes ears shoulders elbows -> Goliath 0-8; l/r wrist -> 62 / 41; hips knees ankles -> 9-14.
--overlays N draws N images with the mapped keypoints (green = GT, red = Sapiens2) to check the mapping.
Usage: python tools/test_sapiens2.py --model facebook/sapiens2-pose-1b --gt <coco json> --images <dir>
           --out results.json [--overlays 8]
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, Sapiens2ForPoseEstimation

COCO_FROM_GOLIATH = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]
IN_VIS_THRE = 0.2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--images', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--no-flip', action='store_true')
    ap.add_argument('--overlays', type=int, default=0)
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = Sapiens2ForPoseEstimation.from_pretrained(args.model, torch_dtype=torch.bfloat16
                                                      if dev == 'cuda' and torch.cuda.is_bf16_supported()
                                                      else torch.float16 if dev == 'cuda' else torch.float32)
    model = model.to(dev).eval()
    flip_pairs = torch.tensor(model.config.flip_pairs, device=dev)
    gt = json.load(open(args.gt))
    anns = {}
    for a in gt['annotations']:
        if max(a['keypoints']) > 0 and a['bbox'][2] > 0 and a['bbox'][3] > 0:
            anns.setdefault(a['image_id'], []).append(a)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    ov_dir = os.path.splitext(args.out)[0] + '_overlays'
    results, n_ov = [], 0
    raw_scores = []
    with torch.no_grad():
        for im in gt['images']:
            if im['id'] not in anns:
                continue
            image = Image.open(os.path.join(args.images, im['file_name'])).convert('RGB')
            boxes = [a['bbox'] for a in anns[im['id']]]
            inputs = processor(image, boxes=[boxes], return_tensors='pt').to(dev)
            pv = inputs['pixel_values'].to(model.dtype)
            out = model(pixel_values=pv)
            out_f = None if args.no_flip else model(pixel_values=pv.flip(-1), flip_pairs=flip_pairs)
            poses = processor.post_process_pose_estimation(out, boxes=[boxes], outputs_flipped=out_f)[0]
            for p, a in zip(poses, anns[im['id']]):
                xy = p['keypoints'].float().cpu().numpy()[COCO_FROM_GOLIATH]
                s = p['scores'].float().cpu().numpy()[COCO_FROM_GOLIATH]
                raw_scores.append(s)
                vis = s[s > IN_VIS_THRE]
                results.append({'image_id': im['id'], 'category_id': 1,
                                'keypoints': np.concatenate([xy, s[:, None]], 1).reshape(-1).tolist(),
                                'score': float(vis.mean()) if len(vis) else 0.0})
                if n_ov < args.overlays:
                    os.makedirs(ov_dir, exist_ok=True)
                    d = ImageDraw.Draw(image)
                    g = np.array(a['keypoints']).reshape(17, 3)
                    for j in range(17):
                        if g[j, 2] > 0:
                            d.ellipse([g[j, 0] - 5, g[j, 1] - 5, g[j, 0] + 5, g[j, 1] + 5], fill=(0, 200, 0))
                        d.ellipse([xy[j, 0] - 4, xy[j, 1] - 4, xy[j, 0] + 4, xy[j, 1] + 4], outline=(230, 0, 0), width=2)
                        d.text((xy[j, 0] + 6, xy[j, 1]), str(j), fill=(230, 0, 0))
                    image.save(os.path.join(ov_dir, im['file_name']))
                    n_ov += 1
    json.dump(results, open(args.out, 'w'))
    sc = np.array(raw_scores)
    print(f'{args.model}: {len(results)} poses -> {args.out}; keypoint score mean {sc.mean():.3f} '
          f'p10 {np.percentile(sc, 10):.3f} p90 {np.percentile(sc, 90):.3f}')


if __name__ == '__main__':
    main()
