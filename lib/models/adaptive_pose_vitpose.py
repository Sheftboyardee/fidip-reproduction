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

    def forward(self, x):
        kw = {}
        if self.moe:
            kw['dataset_index'] = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        heatmaps = self.vitpose(pixel_values=x, **kw).heatmaps
        return heatmaps.new_zeros(x.shape[0], 1), heatmaps


def get_adaptive_pose_net(cfg, is_train, **kwargs):
    return ViTPoseNet(cfg), None
