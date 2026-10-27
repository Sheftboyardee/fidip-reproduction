# ------------------------------------------------------------------------------
# Round 12, experiment 3: unlabelled person crops for mean-teacher training (tools/train_vitpose.py).
# Each item is one image + person box (from tools/estimate_boxes.py or a GT box) and returns
#   weak:   the plain test-time crop (teacher input),
#   strong: a scale/rotation/flip/colour-jittered crop of the same person (student input),
#   A:      2x3 affine taking weak-heatmap coordinates to strong-heatmap coordinates,
#   flip:   whether the strong view is mirrored (left/right joints must be swapped).
# Geometry follows JointsDataset (same centre/scale convention, get_affine_transform), so the
# pseudo-targets live in exactly the coordinate frame the supervised targets use.
# ------------------------------------------------------------------------------
import json
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.transforms import get_affine_transform


def _h(m):
    return np.vstack([m, [0, 0, 1]])


class UnlabeledCrops(Dataset):
    def __init__(self, cfg, items_file, transform, strong_color_jitter=0.4):
        # items_file: json list of {"image": path, "bbox": [x, y, w, h]}
        self.items = json.load(open(items_file))
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.aspect_ratio = self.image_size[0] / self.image_size[1]
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.color_rgb = cfg.DATASET.COLOR_RGB
        self.jitter = strong_color_jitter
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def _cs(self, box):
        # same as SYRIPDataset._xywh2cs
        x, y, w, h = box
        c = np.array([x + (w - 1) * 0.5, y + (h - 1) * 0.5], dtype=np.float32)
        if w > self.aspect_ratio * h:
            h = w / self.aspect_ratio
        else:
            w = h * self.aspect_ratio
        return c, np.array([w / 200.0, h / 200.0], dtype=np.float32) * 1.25

    def _jitter(self, img):
        j = self.jitter
        img = img.astype(np.float32) * np.random.uniform(1 - j, 1 + j)
        mean = img.mean()
        img = (img - mean) * np.random.uniform(1 - j, 1 + j) + mean
        gray = img.mean(axis=2, keepdims=True)
        img = (img - gray) * np.random.uniform(1 - j, 1 + j) + gray
        return np.clip(img, 0, 255).astype(np.uint8)

    def __getitem__(self, idx):
        it = self.items[idx]
        img = cv2.imread(it['image'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if self.color_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        c, s = self._cs(it['bbox'])
        size = tuple(int(v) for v in self.image_size)
        weak = cv2.warpAffine(img, get_affine_transform(c, s, 0, self.image_size), size, flags=cv2.INTER_LINEAR)
        hm_weak = get_affine_transform(c, s, 0, self.heatmap_size)

        sf, rf = self.scale_factor, self.rotation_factor
        s2 = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.6 else 0
        c2, src, flip = c.copy(), img, False
        F = np.eye(3)
        if self.flip and random.random() <= 0.5:
            src, flip = img[:, ::-1, :], True
            c2[0] = img.shape[1] - c2[0] - 1
            F = np.array([[-1, 0, img.shape[1] - 1], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        strong = cv2.warpAffine(np.ascontiguousarray(src), get_affine_transform(c2, s2, r, self.image_size), size,
                                flags=cv2.INTER_LINEAR)
        strong = self._jitter(strong)
        hm_strong = get_affine_transform(c2, s2, r, self.heatmap_size)
        # weak heatmap -> image -> (flipped image) -> strong heatmap
        A = (_h(hm_strong) @ F @ np.linalg.inv(_h(hm_weak)))[:2].astype(np.float32)
        return self.transform(weak), self.transform(strong), torch.from_numpy(A), flip


def pseudo_targets(teacher_hm, A, flip, flip_pairs, heatmap_size, sigma, thr):
    """Teacher heatmaps on the weak view -> Gaussian targets + weights in the strong view.
    teacher_hm: (B, J, H, W) tensor; A: (B, 2, 3); flip: (B,) bool. Keeps joints whose peak
    value exceeds thr and whose transformed location falls inside the strong heatmap."""
    B, J, H, W = teacher_hm.shape
    flat = teacher_hm.reshape(B, J, -1)
    conf, idx = flat.max(-1)
    x, y = (idx % W).float(), (idx // W).float()
    # sub-pixel: quarter-pixel shift towards the higher neighbour (as in the original HRNet decoder)
    hm = teacher_hm
    xi, yi = x.long().clamp(1, W - 2), y.long().clamp(1, H - 2)
    b = torch.arange(B, device=hm.device)[:, None].expand(B, J)
    j = torch.arange(J, device=hm.device)[None, :].expand(B, J)
    x = x + 0.25 * torch.sign(hm[b, j, yi, xi + 1] - hm[b, j, yi, xi - 1])
    y = y + 0.25 * torch.sign(hm[b, j, yi + 1, xi] - hm[b, j, yi - 1, xi])
    pts = torch.stack([x, y, torch.ones_like(x)], -1)                    # (B, J, 3)
    st = torch.einsum('bij,bkj->bki', A.to(pts), pts)                    # (B, J, 2)
    w = (conf > thr).float()
    if flip.any():
        perm = torch.arange(J, device=hm.device)
        for a, c in flip_pairs:
            perm[a], perm[c] = c, a
        fl = flip.to(hm.device)
        st = torch.where(fl[:, None, None], st[:, perm], st)
        w = torch.where(fl[:, None], w[:, perm], w)
    gx, gy = st[..., 0], st[..., 1]
    w = w * ((gx >= 0) & (gx <= W - 1) & (gy >= 0) & (gy <= H - 1)).float()
    xs = torch.arange(W, device=hm.device, dtype=torch.float32)
    ys = torch.arange(H, device=hm.device, dtype=torch.float32)
    tgt = torch.exp(-((xs[None, None, None, :] - gx[..., None, None]) ** 2 +
                      (ys[None, None, :, None] - gy[..., None, None]) ** 2) / (2 * sigma ** 2))
    return tgt * w[..., None, None], w.unsqueeze(-1)
