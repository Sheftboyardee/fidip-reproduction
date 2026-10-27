# ------------------------------------------------------------------------------
# HF transformers ViTPose / ViTPose++ wrapped in the adaptive_pose_* interface (PLAN.md stage D2),
# so tools/test_adaptive_model.py, core.function.validate_adaptive and scripts/eval500.sbatch work
# unchanged. model_p(x) returns (feature, heatmaps) like PoseHighResolutionNet; the feature is a
# tiny placeholder because the domain classifier is not used for ViTPose fine-tuning.
# Checkpoint: cfg.MODEL.HF_NAME (e.g. usyd-community/vitpose-plus-large); set HF_HOME to the
# /projects cache and HF_HUB_OFFLINE=1 on compute nodes.
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn


class ViTPoseNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # imported here so HRNet runs don't need transformers
        from transformers import VitPoseForPoseEstimation
        self.vitpose = VitPoseForPoseEstimation.from_pretrained(cfg.MODEL.HF_NAME)
        # ViTPose++ is mixture-of-experts; expert 0 is the COCO head (17 keypoints, same order)
        self.moe = getattr(self.vitpose.config.backbone_config, 'num_experts', 1) > 1
        w, h = cfg.MODEL.IMAGE_SIZE
        if [h, w] != list(self.vitpose.config.backbone_config.image_size):
            self._resize_input(h, w)

    def _resize_input(self, h, w):
        """Bicubic-interpolate the patch position embeddings to an h x w input (ViTPose does the same
        for its 384x288 / 576x432 models). HF's patch embedding refuses any other size, so update it too.
        Fine-tuned checkpoints then load with the resized table."""
        emb = self.vitpose.backbone.embeddings
        patch = emb.patch_embeddings
        ph, pw = patch.patch_size
        old_h, old_w = patch.image_size
        # same arithmetic as the stride-16, padding-2 projection conv
        gh_old, gw_old = (old_h + 4 - ph) // ph + 1, (old_w + 4 - pw) // pw + 1
        gh, gw = (h + 4 - ph) // ph + 1, (w + 4 - pw) // pw + 1
        pos = emb.position_embeddings.data
        cls_pos, grid = pos[:, :1], pos[:, 1:]
        grid = grid.reshape(1, gh_old, gw_old, -1).permute(0, 3, 1, 2)
        grid = nn.functional.interpolate(grid.float(), size=(gh, gw), mode='bicubic', align_corners=False)
        grid = grid.permute(0, 2, 3, 1).reshape(1, gh * gw, -1).to(pos.dtype)
        emb.position_embeddings = nn.Parameter(torch.cat([cls_pos, grid], dim=1))
        patch.image_size = (h, w)
        patch.num_patches = gh * gw
        # the head reshapes tokens to a grid using backbone_config.image_size // patch_size
        self.vitpose.config.backbone_config.image_size = [h, w]
        self.vitpose.backbone.config.image_size = [h, w]

    def train(self, mode=True):
        # Keep BatchNorm (the ViTPose++ "classic" decoder has two) on its COCO running stats.
        # In train mode, per-batch stats of our small augmented infant batches roughly doubled the
        # heatmap amplitude (loss x100) and wrecked the model within an epoch (round 8, first try).
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
        return self

    def forward(self, x):
        kw = {}
        if self.moe:
            kw['dataset_index'] = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        heatmaps = self.vitpose(pixel_values=x, **kw).heatmaps
        return heatmaps.new_zeros(x.shape[0], 1), heatmaps


def get_adaptive_pose_net(cfg, is_train, **kwargs):
    return ViTPoseNet(cfg), None
