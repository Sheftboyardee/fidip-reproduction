"""Round 13: fine-tune Sapiens2 pose (Goliath-308 head) on SyRIP with COCO-17 supervision.

Runs in ~/venvs/sapiens2 (transformers 5.x). The pretrained head is kept: the 17 COCO joints are
written into their Goliath channels (tools/test_sapiens2.COCO_FROM_GOLIATH) and the loss uses only
those channels, so the other 291 are unsupervised and ignored at test time.
  * crops/targets: the HF processor itself (GT box -> 1.25-padded 1024x768 crop, UDP Gaussian
    heatmaps at 256x192, sigma 6), i.e. exactly the convention its decoder inverts
  * augmentation: horizontal flip (p 0.5, L/R joints swapped), box scale x[0.75, 1.25] and shift
    +-0.1 of box size, colour jitter 0.3; no rotation (the processor crop is axis-aligned)
  * loss: MSE over the 17 channels, weighted by visibility; AdamW, bf16 autocast, gradient
    checkpointing, step decay x0.1 at 75% of the epochs (as in the ViTPose H20 recipe)
  * final epoch only: flip-test predictions on --heldout are written to
    <out>/results/keypoints_heldout_infant_results_0.json (the path scripts/cv_score.py reads)
Usage: python tools/train_sapiens2.py --model facebook/sapiens2-pose-1b --train <coco json>
          --images <dir> --heldout <coco json> --out output/cv-s2ft-f0/syrip/sapiens2/1b [--epochs 8]
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_sapiens2 import COCO_FROM_GOLIATH, load_model, predict  # noqa: E402

COCO_FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


class PoseCrops(Dataset):
    def __init__(self, coco, images, processor, augment=True, n_kpt=308):
        self.items = [(next(im for im in coco['images'] if im['id'] == a['image_id'])['file_name'], a)
                      for a in coco['annotations'] if a['num_keypoints'] > 0 and a['bbox'][2] > 0 and a['bbox'][3] > 0]
        self.images, self.processor, self.augment, self.n_kpt = images, processor, augment, n_kpt

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i, flip=None, jitter=None):
        fn, a = self.items[i]
        image = Image.open(os.path.join(self.images, fn)).convert('RGB')
        w_img = image.width
        kp = np.array(a['keypoints'], dtype=np.float32).reshape(17, 3)
        x, y, w, h = a['bbox']
        flip = self.augment and random.random() < 0.5 if flip is None else flip
        jitter = self.augment if jitter is None else jitter
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            kp[:, 0] = np.where(kp[:, 2] > 0, w_img - 1 - kp[:, 0], 0)
            kp = kp[COCO_FLIP]
            x = w_img - x - w
        if jitter:
            s = random.uniform(0.75, 1.25)
            cx, cy = x + w / 2 + random.uniform(-0.1, 0.1) * w, y + h / 2 + random.uniform(-0.1, 0.1) * h
            w, h = w * s, h * s
            x, y = cx - w / 2, cy - h / 2
            for enh in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
                image = enh(image).enhance(random.uniform(0.7, 1.3))
        full = np.zeros((self.n_kpt, 3), dtype=np.float32)
        full[COCO_FROM_GOLIATH] = kp
        out = self.processor(image, boxes=[[[x, y, w, h]]], keypoints=[[full.tolist()]], return_tensors='pt',
                             keypoint_heatmap_downscale_factor=4, keypoint_heatmap_sigma=6.0)
        lab, wt = out['labels'], out['label_weights']
        lab = lab[0] if isinstance(lab, (list, tuple)) else lab
        wt = wt[0] if isinstance(wt, (list, tuple)) else wt
        idx = torch.tensor(COCO_FROM_GOLIATH)
        return out['pixel_values'][0], lab.reshape(-1, *lab.shape[-2:])[idx], wt.reshape(-1)[idx]


def check_targets(ds, processor, n, say):
    """Round trip: GT keypoints -> processor targets -> HF decoder, without jitter, plain and flipped
    (the flipped decode is mapped back to the original image and L/R before comparing)."""
    from types import SimpleNamespace
    for flip in (False, True):
        errs = []
        for i in random.Random(0).sample(range(len(ds)), n):
            fn, a = ds.items[i]
            _, lab, wt = ds.__getitem__(i, flip=flip, jitter=False)
            full = torch.zeros(1, 308, *lab.shape[-2:])
            full[0, COCO_FROM_GOLIATH] = lab
            x, y, w, h = a['bbox']
            w_img = Image.open(os.path.join(ds.images, fn)).width
            box = [[w_img - x - w, y, w, h]] if flip else [a['bbox']]
            p = processor.post_process_pose_estimation(SimpleNamespace(heatmaps=full), boxes=[box])[0][0]
            xy = p['keypoints'].float().numpy()[COCO_FROM_GOLIATH]
            if flip:
                xy = xy[COCO_FLIP]
                xy[:, 0] = w_img - 1 - xy[:, 0]
            g = np.array(a['keypoints'], dtype=np.float32).reshape(17, 3)
            ok = (g[:, 2] > 0) & (wt.numpy() > 0)
            errs += (np.linalg.norm(xy - g[:, :2], axis=1)[ok] / max(w, h) * 100).tolist()
        say(f'target round trip ({"flipped" if flip else "plain"}, {n} crops): error mean {np.mean(errs):.3f}, '
            f'max {np.max(errs):.3f} (% of box size; one heatmap px ~ 0.65%)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='facebook/sapiens2-pose-1b')
    ap.add_argument('--train', required=True)
    ap.add_argument('--images', required=True)
    ap.add_argument('--heldout')
    ap.add_argument('--heldout-images')
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--max-steps', type=int, default=0, help='smoke test: stop after N steps')
    ap.add_argument('--check-targets', type=int, default=0,
                    help='decode N un-augmented targets with the HF decoder and report the error vs GT, then exit')
    ap.add_argument('--save', action='store_true', help='also save the fine-tuned weights (bf16, ~2 GB for 1b)')
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(os.path.join(args.out, 'results'), exist_ok=True)
    log = open(os.path.join(args.out, 'train.log'), 'a')

    def say(m):
        m = time.strftime('%Y-%m-%d %H:%M:%S ') + m
        print(m, flush=True)
        log.write(m + '\n'); log.flush()

    say(f'args {vars(args)}')
    dev = 'cuda'
    processor, model = load_model(args.model, dev)
    model = model.float().train()
    model.gradient_checkpointing_enable()
    ds = PoseCrops(json.load(open(args.train)), args.images, processor)
    if args.check_targets:
        check_targets(ds, processor, args.check_targets, say)
        return
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
                    persistent_workers=True, worker_init_fn=lambda k: random.seed(args.seed * 100 + k))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    step_at = math.ceil(0.75 * args.epochs)
    say(f'{len(ds)} training crops, {len(dl)} steps/epoch, lr x0.1 from epoch {step_at}')
    step = 0
    for ep in range(args.epochs):
        for g in opt.param_groups:
            g['lr'] = args.lr * (0.1 if ep >= step_at else 1.0)
        t0, tot = time.time(), 0.0
        for it, (pv, lab, wt) in enumerate(dl):
            pv, lab, wt = pv.to(dev, non_blocking=True), lab.to(dev), wt.to(dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                hm = model(pixel_values=pv).heatmaps
            hm = hm.float()[:, COCO_FROM_GOLIATH]
            loss = (((hm - lab) ** 2).mean((2, 3)) * wt).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
            step += 1
            if it % 20 == 0:
                say(f'epoch {ep} [{it}/{len(dl)}] loss {loss.item():.6f} ({tot / (it + 1):.6f}) '
                    f'{(time.time() - t0) / (it + 1):.2f} s/it mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB')
            if args.max_steps and step >= args.max_steps:
                break
        say(f'epoch {ep} done: loss {tot / (it + 1):.6f}, {time.time() - t0:.0f} s')
        if args.max_steps and step >= args.max_steps:
            break
    model.eval()
    if args.save:
        torch.save({k: v.to(torch.bfloat16) for k, v in model.state_dict().items()},
                   os.path.join(args.out, 'final_state_bf16.pth'))
    if args.heldout:
        model = model.to(torch.bfloat16)
        res, sc = predict(model, processor, json.load(open(args.heldout)), args.heldout_images or args.images)
        path = os.path.join(args.out, 'results', 'keypoints_heldout_infant_results_0.json')
        json.dump(res, open(path, 'w'))
        say(f'{len(res)} held-out poses -> {path}; keypoint score mean {sc.mean():.3f}')


if __name__ == '__main__':
    main()
