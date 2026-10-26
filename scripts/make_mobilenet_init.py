"""Build an ImageNet-initialised backbone checkpoint for adaptive_pose_mobile.

No COCO-pretrained Pose-MobileNet is available (the paper's starting point), so the backbone
starts from torchvision's ImageNet MobileNetV2 and the deconv/heatmap head stays randomly
initialised. PoseMobileNet's `features.*` tensors match torchvision's one-to-one in order and
shape but not in name, so they are mapped by position.

Usage: python scripts/make_mobilenet_init.py <out.pth> [--no-download]
  --no-download: use randomly initialised torchvision weights (mapping test only).
"""
import argparse
import os
import sys

import torch
import torchvision

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from config import cfg, update_config  # noqa: E402
import models  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--no-download', action='store_true')
args = ap.parse_args()

weights = None if args.no_download else torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
tv = torchvision.models.mobilenet_v2(weights=weights).state_dict()
src = [(k, v) for k, v in tv.items() if k.startswith('features.')]

update_config(cfg, argparse.Namespace(
    cfg='experiments/coco/mobilenet/mobile_224x224_adam_lr1e-3_infant.yaml', opts=[],
    modelDir='', logDir='', dataDir='', prevModelDir=''))
model_p, _ = models.adaptive_pose_mobile.get_adaptive_pose_net(cfg, is_train=False)
dst = [(k, v) for k, v in model_p.state_dict().items() if k.startswith('features.')]
assert len(src) == len(dst), (len(src), len(dst))
out = {}
for (ks, vs), (kd, vd) in zip(src, dst):
    assert vs.shape == vd.shape, (ks, kd, vs.shape, vd.shape)
    out[kd] = vs.clone()
torch.save(out, args.out)

# Verify through the same path training uses (init_weights with strict=False).
model_p.init_weights(args.out)
sd = model_p.state_dict()
bad = [k for k in out if not torch.equal(sd[k], out[k])]
print(f'mapped {len(out)} backbone tensors -> {args.out}; mismatches after init_weights: {len(bad)}')
assert not bad
