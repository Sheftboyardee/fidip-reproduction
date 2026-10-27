"""Weighted average of pose-net state dicts (model soups / WiSE-FT), with optional BN recalibration.

All inputs must share keys (COCO w48 init, our final_state.pth files and hrnet_fidip.pth do).
Weights are normalised to sum to 1. Integer buffers (num_batches_tracked) are taken from the first
input. With --recal-bn, BN running stats are re-estimated on the training set (train-time
augmentation, cumulative average over the whole epoch) instead of being averaged.

Usage (from repo root, on a GPU node):
  python scripts/weight_average.py --cfg <cfg.yaml> --out soup.pth [--recal-bn] \
      0.5:ckpt_a.pth 0.5:ckpt_b.pth [cfg opts after --opts ...]
"""
import argparse
import os
import sys

import torch
import torchvision.transforms as transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from config import cfg, update_config  # noqa: E402
import dataset  # noqa: E402,F401
import models  # noqa: E402,F401


def average(specs):
    weights = [float(s.split(':', 1)[0]) for s in specs]
    paths = [s.split(':', 1)[1] for s in specs]
    total = sum(weights)
    sds = [torch.load(p, map_location='cpu') for p in paths]
    keys = set(sds[0])
    for p, sd in zip(paths[1:], sds[1:]):
        if set(sd) != keys:
            sys.exit(f'key mismatch between {paths[0]} and {p}')
    out = {}
    for k, v in sds[0].items():
        if v.is_floating_point():
            out[k] = sum(w / total * sd[k].float() for w, sd in zip(weights, sds)).to(v.dtype)
        else:
            out[k] = v.clone()
    return out


@torch.no_grad()
def recalibrate_bn(state):
    model_p, _ = eval('models.' + cfg.MODEL.NAME + '.get_adaptive_pose_net')(cfg, is_train=False)
    model_p.load_state_dict(state)
    bns = [m for m in model_p.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative moving average over every batch seen
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
        transforms.Compose([transforms.ToTensor(), normalize]))
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
                                         shuffle=True, num_workers=cfg.WORKERS, drop_last=True)
    model_p = model_p.cuda().train()
    for batch in loader:
        model_p(batch[0].cuda(non_blocking=True))
    print(f'recalibrated {len(bns)} BN layers on {len(loader)} batches of {cfg.DATASET.TRAIN_SET}')
    return {k: v.cpu() for k, v in model_p.state_dict().items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--recal-bn', action='store_true')
    ap.add_argument('specs', nargs='+', help='weight:checkpoint.pth')
    ap.add_argument('--opts', nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()
    # update_config expects the attribute names used by the tools/ scripts
    args.opts = args.opts or []
    args.modelDir = args.logDir = args.dataDir = args.prevModelDir = ''
    update_config(cfg, args)

    state = average(args.specs)
    if args.recal_bn and cfg.MODEL.NAME == 'adaptive_pose_vitpose':
        # the ViTPose wrapper keeps its BN in eval mode, so recalibration would leave reset stats
        sys.exit('--recal-bn is not supported for adaptive_pose_vitpose (decoder BN is frozen)')
    if args.recal_bn:
        state = recalibrate_bn(state)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(state, args.out)
    print(f'wrote {args.out} from {args.specs}')


if __name__ == '__main__':
    main()
