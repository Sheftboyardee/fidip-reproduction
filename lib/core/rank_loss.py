# ------------------------------------------------------------------------------
# Round 12, experiment 2: RSPose-style ranking losses for heatmaps (Keles et al., arXiv 2511.13857).
# No official code was released, so this is our implementation of the paper's equations:
#   L = c_r * L_S-Rank + c_s * L_S-Sort (+ c_i * L_I-Sort)                          (Eq. 14)
# * S-Rank (Eq. 5-6): within each joint heatmap, every positive pixel should outrank every
#   negative pixel (AP-Loss style ranking error FP(i) / rank(i)).
# * S-Sort (Eq. 7-9): positive pixels should be ordered by their Gaussian target value.
# * I-Sort (Eq. 10-13): across the joint instances of a batch, the heatmap peak (the keypoint
#   confidence used by AP's cross-image ranking) should be ordered by keypoint similarity
#   KS = exp(-d^2 / (2 s^2 k^2)) of the decoded peak.
# The step function is smoothed as clamp(x / (2 delta) + 0.5, 0, 1) and the non-differentiable
# errors are optimised with the error-driven ("identity update") gradients of AP-Loss / RS-Loss
# (Appendix A; same update as Oksuz et al.'s official RankSort code), computed here in a
# vectorised, chunked form. Deltas and weights default to the paper's ViTPose-H row (Table 8).
# Our targets are untruncated Gaussians (JointsDataset.generate_target), so "positive" pixels
# are those within the usual 3-sigma window: target > exp(-4.5).
# ------------------------------------------------------------------------------
import math

import torch
import torch.nn as nn

POS_THR = math.exp(-4.5)
# COCO OKS constants (pycocotools: vars = (2 * sigma)^2)
COCO_SIGMAS = torch.tensor([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62,
                            1.07, 1.07, .87, .87, .89, .89]) / 10.0


def _step(x, delta):
    return torch.clamp(x / (2 * delta) + 0.5, 0, 1)


def _sort_terms(s_i, s_j, t_i, t_j, valid_i, valid_j, delta):
    """Sorting error and identity-update gradients for items i against positives j.
    s_*/t_*/valid_*: scores / targets / masks, shapes (..., P, 1) and (..., 1, P).
    Returns (error_i, grad_i, grad_j), zero for padded items."""
    rel = _step(s_j - s_i, delta) * valid_j                     # (.., P, P): j ranked above i
    rank_pos = rel.sum(-1)
    current = (rel * (1 - t_j)).sum(-1) / rank_pos.clamp_min(1e-10)
    ordered = (t_j >= t_i).float()
    tgt = ordered * rel
    target = (tgt * (1 - t_j)).sum(-1) / tgt.sum(-1).clamp_min(1e-10)
    err = (current - target) * valid_i.squeeze(-1)             # (.., P)
    missorted = (1 - ordered) * rel
    denom = missorted.sum(-1)
    has = (denom > 1e-10).float()
    grad_i = -err * has
    grad_j = (missorted * (err * has / denom.clamp_min(1e-10)).unsqueeze(-1)).sum(-2)
    return err, grad_i, grad_j


class _SpatialRS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, targets, valid, d_rank, d_sort, c_rank, c_sort, chunk):
        # scores/targets: (N, M) flattened heatmaps; valid: (N,) bool (joint visible, has positives)
        grads = torch.zeros_like(scores)
        rank_err_sum = scores.new_zeros(())
        sort_err_sum = scores.new_zeros(())
        idx = valid.nonzero().squeeze(1)
        n_valid = max(len(idx), 1)
        for c0 in range(0, len(idx), chunk):
            rows = idx[c0:c0 + chunk]
            s, t = scores[rows], targets[rows]
            pos = t > POS_THR
            P = int(pos.sum(1).max())
            # gather positives into a padded (n, P) block, ordered by pixel index
            order = torch.argsort((~pos).float(), dim=1, stable=True)[:, :P]
            ps, pt = s.gather(1, order), t.gather(1, order)
            pv = pos.gather(1, order).float()
            n_pos = pv.sum(1)
            # --- S-Rank: positives vs negatives ---
            fg_rel = _step(ps.unsqueeze(1) - ps.unsqueeze(2), d_rank) * pv.unsqueeze(1)
            bg_rel = _step(s.unsqueeze(1) - ps.unsqueeze(2), d_rank) * (~pos).float().unsqueeze(1)
            fp = bg_rel.sum(-1)
            rank_err = fp / (fg_rel.sum(-1) + fp).clamp_min(1e-10) * pv
            has = (fp > 1e-10).float() * pv
            g_pos = -rank_err * has
            g_all = (bg_rel * (rank_err * has / fp.clamp_min(1e-10)).unsqueeze(-1)).sum(1)
            del fg_rel, bg_rel
            # --- S-Sort: positives ordered by target value ---
            sort_err, gi, gj = _sort_terms(ps.unsqueeze(2), ps.unsqueeze(1), pt.unsqueeze(2), pt.unsqueeze(1),
                                           pv.unsqueeze(2), pv.unsqueeze(1), d_sort)
            g_sort = gi + gj
            g = c_rank * g_all
            g.scatter_add_(1, order, c_rank * g_pos + c_sort * g_sort)
            grads[rows] = g / n_pos.unsqueeze(1) / n_valid
            rank_err_sum = rank_err_sum + (rank_err.sum(1) / n_pos).sum()
            sort_err_sum = sort_err_sum + (sort_err.sum(1) / n_pos).sum()
        ctx.save_for_backward(grads)
        rank, sort = rank_err_sum / n_valid, sort_err_sum / n_valid
        ctx.mark_non_differentiable(rank, sort)
        return c_rank * rank + c_sort * sort, rank, sort

    @staticmethod
    def backward(ctx, g_total, g_rank, g_sort):
        grads, = ctx.saved_tensors
        return grads * g_total, None, None, None, None, None, None, None


class _InstanceSort(torch.autograd.Function):
    @staticmethod
    def forward(ctx, conf, ks, delta):
        # conf, ks: (N,) peak confidence and keypoint similarity of each visible joint instance
        P = conf.numel()
        one = torch.ones(P, device=conf.device)
        err, gi, gj = _sort_terms(conf.view(P, 1), conf.view(1, P), ks.view(P, 1), ks.view(1, P),
                                  one.view(P, 1), one.view(1, P), delta)
        ctx.save_for_backward((gi + gj) / P)
        return err.mean()

    @staticmethod
    def backward(ctx, g):
        grads, = ctx.saved_tensors
        return grads * g, None, None


class RSPoseLoss(nn.Module):
    """Drop-in replacement for JointsMSELoss: forward(output, target, target_weight) -> loss.
    Joints with target_weight 0 are skipped; the target_weight magnitude is not used (ranking
    errors are normalised per heatmap). self.last holds the individual terms for logging."""

    def __init__(self, d_rank=0.4, d_sort=1.5, d_inst=3.0, c_rank=1.0, c_sort=2.0, c_inst=0.0,
                 crop_padding=1.25, chunk=64):
        super().__init__()
        self.d_rank, self.d_sort, self.d_inst = d_rank, d_sort, d_inst
        self.c_rank, self.c_sort, self.c_inst = c_rank, c_sort, c_inst
        self.crop_padding, self.chunk = crop_padding, chunk
        self.last = {}

    def forward(self, output, target, target_weight):
        B, J, H, W = output.shape
        s = output.float().reshape(B * J, H * W)
        t = target.float().reshape(B * J, H * W)
        valid = (target_weight.reshape(B * J) > 0) & ((t > POS_THR).sum(1) > 0)
        loss, rank, sort = _SpatialRS.apply(s, t, valid, self.d_rank, self.d_sort, self.c_rank, self.c_sort,
                                            self.chunk)
        self.last = {'s_rank': rank.item(), 's_sort': sort.item()}
        if self.c_inst > 0:
            inst = self._instance_sort(s, t, valid, J, H, W)
            loss = loss + self.c_inst * inst
            self.last['i_sort'] = inst.item()
        return loss

    def _instance_sort(self, s, t, valid, J, H, W):
        idx = valid.nonzero().squeeze(1)
        if len(idx) < 2:
            return s.sum() * 0
        sv, tv = s[idx], t[idx]
        peak = sv.argmax(1)
        conf = sv.gather(1, peak.unsqueeze(1)).squeeze(1)
        py, px = (peak // W).float(), (peak % W).float()
        # GT location = centroid of the target's 3-sigma window (exact unless clipped by the border)
        w = tv * (tv > POS_THR)
        ys = torch.arange(H, device=s.device, dtype=torch.float32).repeat_interleave(W)
        xs = torch.arange(W, device=s.device, dtype=torch.float32).repeat(H)
        gy = (w * ys).sum(1) / w.sum(1)
        gx = (w * xs).sum(1) / w.sum(1)
        d2 = (px - gx) ** 2 + (py - gy) ** 2
        # object area in heatmap pixels: the crop is the box enlarged by crop_padding per side
        area = H * W / self.crop_padding ** 2
        k = COCO_SIGMAS.to(s.device)[idx % J] * 2
        ks = torch.exp(-d2 / (2 * area * k ** 2))
        # gradient flows only into the peak pixel of each heatmap
        return _InstanceSort.apply(conf, ks.detach(), self.d_inst)
