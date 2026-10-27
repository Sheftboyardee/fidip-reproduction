"""Round 12, experiment 3: person boxes for unannotated images (no detector in this repo).

Two passes of zero-shot ViTPose (HF pipeline, as tools/test_vitpose.py): start from the whole
image, take the box around keypoints with score > --kpt-thr, enlarge it by --margin per side,
and repeat once from that box. Writes [{"image", "bbox" [x, y, w, h], "score"}] for
lib/dataset/unlabeled.py. A zero-shot model is used so that no fold's labels shape the boxes.
Usage: python tools/estimate_boxes.py --images list.json --out boxes.json \
           [--model usyd-community/vitpose-plus-huge]
"""
import argparse
import json

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, VitPoseForPoseEstimation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True, help='json list of image paths')
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='usyd-community/vitpose-plus-huge')
    ap.add_argument('--kpt-thr', type=float, default=0.3)
    ap.add_argument('--margin', type=float, default=0.1)
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = AutoProcessor.from_pretrained(args.model)
    model = VitPoseForPoseEstimation.from_pretrained(args.model).to(dev).eval()
    moe = getattr(model.config.backbone_config, 'num_experts', 1) > 1
    out = []
    for path in json.load(open(args.images)):
        image = Image.open(path).convert('RGB')
        W, H = image.size
        box = [0.0, 0.0, float(W), float(H)]
        for _ in range(2):
            inputs = processor(image, boxes=[[box]], return_tensors='pt').to(dev)
            kw = {'dataset_index': torch.zeros(1, dtype=torch.long, device=dev)} if moe else {}
            with torch.no_grad():
                pose = processor.post_process_pose_estimation(model(**inputs, **kw), boxes=[[box]])[0][0]
            xy, s = pose['keypoints'].cpu().numpy(), pose['scores'].cpu().numpy()
            keep = xy[s > args.kpt_thr]
            if len(keep) < 3:
                break
            x0, y0 = keep.min(0)
            x1, y1 = keep.max(0)
            mx, my = (x1 - x0) * args.margin, (y1 - y0) * args.margin
            x0, y0 = max(0.0, x0 - mx), max(0.0, y0 - my)
            x1, y1 = min(W - 1.0, x1 + mx), min(H - 1.0, y1 + my)
            box = [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]
        out.append({'image': path, 'bbox': box, 'score': float(s.mean())})
        print(path, [round(v) for v in box], round(float(s.mean()), 3))
    json.dump(out, open(args.out, 'w'), indent=0)


if __name__ == '__main__':
    main()
