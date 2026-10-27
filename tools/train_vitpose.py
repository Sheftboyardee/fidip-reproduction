# ------------------------------------------------------------------------------
# Fine-tune HF ViTPose / ViTPose++ on SyRIP train_infant (PLAN.md stage D2).
# Same data pipeline, loss (JointsMSELoss), balanced real/synthetic sampler and Test100
# validation as tools/train_adaptive_model_hrnet.py, minus the domain classifier (FiDIP's
# adversarial term did not help in rounds 1-4) and minus that script's hard-coded
# "freeze the first 109 parameters", which is HRNet-specific.
# Differences for ViTs: AdamW (TRAIN.WD), fp16/bf16 autocast, optional layer-wise LR decay
# (TRAIN.LAYER_DECAY), model-only checkpoints (no resume; runs are short).
# Usage: python tools/train_vitpose.py --cfg experiments/syrip/vitpose/vitpose_plus_256x192_infant.yaml \
#            MODEL.HF_NAME usyd-community/vitpose-plus-large TRAIN.LR 5e-5 OUTPUT_DIR output/vit-l
# ------------------------------------------------------------------------------
import argparse
import os
import pprint
import re
import shutil

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import _init_paths  # noqa: F401
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import AverageMeter
from core.function import validate_adaptive
from utils.utils import create_logger
from utils.sampler import BalancedBatchSampler
import dataset  # noqa: F401
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune ViTPose on SyRIP')
    parser.add_argument('--cfg', required=True, type=str)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.modelDir = args.logDir = args.dataDir = args.prevModelDir = ''
    return args


def param_groups(model, lr, wd, decay):
    """AdamW groups: no weight decay on biases/norms/pos-embeddings; optional layer-wise LR decay."""
    n_layers = model.vitpose.config.backbone_config.num_hidden_layers
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        m = re.search(r'encoder\.layer\.(\d+)\.', name)
        if m:
            depth = int(m.group(1)) + 1
        elif 'embeddings' in name:
            depth = 0
        else:  # decoder head
            depth = n_layers + 1
        scale = decay ** (n_layers + 1 - depth)
        no_wd = p.ndim == 1 or 'position_embeddings' in name
        key = (depth, no_wd)
        if key not in groups:
            groups[key] = {'params': [], 'lr': lr * scale, 'weight_decay': 0.0 if no_wd else wd}
        groups[key]['params'].append(p)
    return list(groups.values())


def main():
    args = parse_args()
    update_config(cfg, args)
    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'train')
    logger.info(pprint.pformat(args))
    logger.info(cfg)
    cudnn.benchmark = cfg.CUDNN.BENCHMARK

    model_p, _ = models.adaptive_pose_vitpose.get_adaptive_pose_net(cfg, is_train=True)
    shutil.copy2(os.path.join(os.path.dirname(__file__), '../lib/models/adaptive_pose_vitpose.py'),
                 final_output_dir)
    logger.info('=> %s: %.1fM params', cfg.MODEL.HF_NAME,
                sum(p.numel() for p in model_p.parameters()) / 1e6)
    model_p = torch.nn.DataParallel(model_p, device_ids=cfg.GPUS).cuda()

    criterion_p = JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT).cuda()
    optimizer = torch.optim.AdamW(
        param_groups(model_p.module, cfg.TRAIN.LR, cfg.TRAIN.WD, cfg.TRAIN.LAYER_DECAY), lr=cfg.TRAIN.LR)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype == torch.float16)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tf = transforms.Compose([transforms.ToTensor(), normalize])
    train_dataset = eval('dataset.' + cfg.DATASET.DATASET)(cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True, tf)
    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False, tf)
    syn_labels = train_dataset._load_syrip_syn_annotations()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=BalancedBatchSampler(train_dataset, syn_labels),
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        num_workers=cfg.WORKERS, pin_memory=cfg.PIN_MEMORY)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS), shuffle=False,
        num_workers=cfg.WORKERS, pin_memory=cfg.PIN_MEMORY)

    writer_dict = {'writer': SummaryWriter(log_dir=tb_log_dir),
                   'train_global_steps': 0, 'valid_global_steps': 0}

    # zero-shot Test100 through this repo's pipeline, to compare with tools/test_vitpose.py
    perf = validate_adaptive(cfg, valid_loader, valid_dataset, model_p, criterion_p,
                             final_output_dir, tb_log_dir, writer_dict)
    logger.info('=> zero-shot Test100 AP (repo pipeline): %.4f', perf)

    best_perf, losses_list = 0.0, []
    for epoch in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        model_p.train()
        losses = AverageMeter()
        for i, (input, target, target_weight, meta) in enumerate(train_loader):
            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)
            with torch.autocast('cuda', dtype=amp_dtype):
                _, output = model_p(input.cuda(non_blocking=True))
            loss = criterion_p(output.float(), target, target_weight)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.update(loss.item(), input.size(0))
            losses_list.append(loss.item())
            if i % cfg.PRINT_FREQ == 0:
                logger.info('Epoch: [%d][%d/%d]\tLoss_p %.6f (%.6f)\tlr %.2e', epoch, i,
                            len(train_loader), losses.val, losses.avg, optimizer.param_groups[-1]['lr'])
            writer_dict['writer'].add_scalar('train_loss', loss.item(), writer_dict['train_global_steps'])
            writer_dict['train_global_steps'] += 1
        lr_scheduler.step()

        perf = validate_adaptive(cfg, valid_loader, valid_dataset, model_p, criterion_p,
                                 final_output_dir, tb_log_dir, writer_dict)
        if perf > best_perf:
            best_perf = perf
            torch.save(model_p.module.state_dict(), os.path.join(final_output_dir, 'model_best.pth'))
            logger.info('=> new best Test100 AP %.4f at epoch %d', perf, epoch)

    torch.save(model_p.module.state_dict(), os.path.join(final_output_dir, 'final_state.pth'))
    np.save(os.path.join(final_output_dir, 'losses_P.npy'), np.array(losses_list))
    writer_dict['writer'].close()
    logger.info('=> saved final_state.pth; best Test100 AP %.4f', best_perf)


if __name__ == '__main__':
    main()
