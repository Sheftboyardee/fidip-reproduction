# ------------------------------------------------------------------------------
# Fine-tune HF ViTPose / ViTPose++ on SyRIP train_infant (PLAN.md stage D2).
# Same data pipeline, loss (JointsMSELoss), balanced real/synthetic sampler and Test100
# validation as tools/train_adaptive_model_hrnet.py, minus the domain classifier (FiDIP's
# adversarial term did not help in rounds 1-4) and minus that script's hard-coded
# "freeze the first 109 parameters", which is HRNet-specific.
# Differences for ViTs: AdamW (TRAIN.WD), fp16/bf16 autocast, optional layer-wise LR decay
# (TRAIN.LAYER_DECAY), optional ranking loss (LOSS.TYPE rspose, round 12) and mean teacher
# (TRAIN.MT_*, round 12). AUTO_RESUME True saves
# checkpoint.pth (model + optimiser) every epoch and resumes from it, so long runs (e.g. 384x288
# ViTPose++-H) can be split into chained short jobs.
# Usage: python tools/train_vitpose.py --cfg experiments/syrip/vitpose/vitpose_plus_256x192_infant.yaml \
#            MODEL.HF_NAME usyd-community/vitpose-plus-large TRAIN.LR 5e-5 OUTPUT_DIR output/vit-l
# ------------------------------------------------------------------------------
import argparse
import copy
import itertools
import json
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
from core.rank_loss import RSPoseLoss
from core.function import AverageMeter
from core.function import validate_adaptive
from utils.utils import create_logger
from utils.sampler import BalancedBatchSampler, MixSampler
from utils.transforms import flip_back
from dataset.unlabeled import UnlabeledCrops, pseudo_targets
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
    if cfg.TRAIN.GRAD_CKPT:
        model_p.vitpose.gradient_checkpointing_enable()
    model_p = torch.nn.DataParallel(model_p, device_ids=cfg.GPUS).cuda()

    criterion_p = JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT).cuda()
    # validation always reports the MSE loss; training may use the round 12 ranking loss instead
    train_criterion = RSPoseLoss(c_inst=cfg.LOSS.RS_INST) if cfg.LOSS.TYPE == 'rspose' else criterion_p
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
    sampler = BalancedBatchSampler(train_dataset, syn_labels)
    if cfg.TRAIN.REAL_FRACTION >= 0:  # stage B: same epoch length, chosen real share, shuffled
        sampler = MixSampler(syn_labels, cfg.TRAIN.REAL_FRACTION, len(sampler))
        logger.info('=> MixSampler: %d of %d samples per epoch from %d real images',
                    sampler.n_real, len(sampler), len(sampler.real))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        num_workers=cfg.WORKERS, pin_memory=cfg.PIN_MEMORY)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS), shuffle=False,
        num_workers=cfg.WORKERS, pin_memory=cfg.PIN_MEMORY)

    # round 12 experiment 3: mean teacher on unlabelled crops (TRAIN.MT_ITEMS); the pool can also
    # hold this run's labelled real training images, with their labels unused (TRAIN.MT_WITH_LABELED)
    teacher, unl_iter = None, None
    if cfg.TRAIN.MT_ITEMS or cfg.TRAIN.MT_WITH_LABELED:
        pool = json.load(open(cfg.TRAIN.MT_ITEMS)) if cfg.TRAIN.MT_ITEMS else []
        n_extra = len(pool)
        if cfg.TRAIN.MT_WITH_LABELED:
            coco = train_dataset.syrip
            for img_id in coco.getImgIds():
                if coco.loadImgs(img_id)[0].get('is_synthetic'):
                    continue
                for a in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
                    pool.append({'image': train_dataset.image_path_from_index(img_id), 'bbox': a['bbox']})
        pool_file = os.path.join(final_output_dir, 'mt_pool.json')
        json.dump(pool, open(pool_file, 'w'))
        logger.info('=> mean teacher: %d unlabelled crops (%d extra + %d labelled real, labels unused), '
                    'EMA %.4f, thr %.2f, weight %.2f', len(pool), n_extra, len(pool) - n_extra,
                    cfg.TRAIN.MT_EMA, cfg.TRAIN.MT_THR, cfg.TRAIN.MT_WEIGHT)
        unl_loader = torch.utils.data.DataLoader(
            UnlabeledCrops(cfg, pool_file, tf), batch_size=cfg.TRAIN.MT_BATCH, shuffle=True, drop_last=True,
            num_workers=cfg.WORKERS, pin_memory=cfg.PIN_MEMORY, persistent_workers=cfg.WORKERS > 0)
        unl_iter = itertools.chain.from_iterable(itertools.repeat(unl_loader))
        teacher = copy.deepcopy(model_p.module).eval()
        for q in teacher.parameters():
            q.requires_grad_(False)
        flip_pairs = valid_dataset.flip_pairs
        accepted = AverageMeter()

    writer_dict = {'writer': SummaryWriter(log_dir=tb_log_dir),
                   'train_global_steps': 0, 'valid_global_steps': 0}

    best_perf, losses_list, begin_epoch = 0.0, [], cfg.TRAIN.BEGIN_EPOCH
    ckpt_file = os.path.join(final_output_dir, 'checkpoint.pth')
    if cfg.AUTO_RESUME and not os.path.exists(ckpt_file) and \
            os.path.exists(os.path.join(final_output_dir, 'final_state.pth')):
        # a spare chained job starting after the run finished must not train from scratch
        logger.info('=> %s already has final_state.pth -- nothing to do', final_output_dir)
        return
    if cfg.AUTO_RESUME and os.path.exists(ckpt_file):
        ckpt = torch.load(ckpt_file, map_location='cpu')
        model_p.module.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        best_perf, losses_list, begin_epoch = ckpt['best_perf'], ckpt['losses'], ckpt['epoch']
        if teacher is not None:
            teacher.load_state_dict(ckpt['teacher'])
        writer_dict['train_global_steps'] = len(losses_list)
        logger.info('=> resumed from %s at epoch %d (best Test100 AP %.4f)', ckpt_file, begin_epoch, best_perf)
        del ckpt
    else:
        # zero-shot Test100 through this repo's pipeline, to compare with tools/test_vitpose.py
        perf = validate_adaptive(cfg, valid_loader, valid_dataset, model_p, criterion_p,
                                 final_output_dir, tb_log_dir, writer_dict)
        logger.info('=> zero-shot Test100 AP (repo pipeline): %.4f', perf)

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        model_p.train()
        losses = AverageMeter()
        for i, (input, target, target_weight, meta) in enumerate(train_loader):
            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)
            with torch.autocast('cuda', dtype=amp_dtype):
                _, output = model_p(input.cuda(non_blocking=True))
            loss = train_criterion(output.float(), target, target_weight)
            if teacher is not None:
                weak, strong, A, flipped = next(unl_iter)
                weak = weak.cuda(non_blocking=True)
                with torch.no_grad(), torch.autocast('cuda', dtype=amp_dtype):
                    t_hm = teacher(weak)[1].float()
                    t_hm_f = teacher(weak.flip(-1))[1].float()
                t_hm = (t_hm + torch.from_numpy(flip_back(t_hm_f.cpu().numpy(), flip_pairs).copy()).cuda()) * 0.5
                u_target, u_weight = pseudo_targets(t_hm, A.cuda(), flipped, flip_pairs, t_hm.shape[2:],
                                                    cfg.MODEL.SIGMA, cfg.TRAIN.MT_THR)
                with torch.autocast('cuda', dtype=amp_dtype):
                    _, s_hm = model_p(strong.cuda(non_blocking=True))
                loss_u = criterion_p(s_hm.float(), u_target, u_weight)
                loss = loss + cfg.TRAIN.MT_WEIGHT * loss_u
                accepted.update(u_weight.mean().item(), weak.size(0))
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if teacher is not None:
                with torch.no_grad():
                    for tq, sq in zip(teacher.parameters(), model_p.module.parameters()):
                        tq.mul_(cfg.TRAIN.MT_EMA).add_(sq.detach(), alpha=1 - cfg.TRAIN.MT_EMA)
            losses.update(loss.item(), input.size(0))
            losses_list.append(loss.item())
            if i % cfg.PRINT_FREQ == 0:
                logger.info('Epoch: [%d][%d/%d]\tLoss_p %.6f (%.6f)\tlr %.2e%s', epoch, i,
                            len(train_loader), losses.val, losses.avg, optimizer.param_groups[-1]['lr'],
                            ''.join(f'\t{k} {v:.4f}' for k, v in getattr(train_criterion, 'last', {}).items()) +
                            (f'\tloss_u {loss_u.item():.6f}\taccepted {accepted.avg:.3f}' if teacher is not None else ''))
            writer_dict['writer'].add_scalar('train_loss', loss.item(), writer_dict['train_global_steps'])
            writer_dict['train_global_steps'] += 1
        lr_scheduler.step()

        perf = validate_adaptive(cfg, valid_loader, valid_dataset, model_p, criterion_p,
                                 final_output_dir, tb_log_dir, writer_dict)
        if perf > best_perf:
            best_perf = perf
            torch.save(model_p.module.state_dict(), os.path.join(final_output_dir, 'model_best.pth'))
            logger.info('=> new best Test100 AP %.4f at epoch %d', perf, epoch)
        if cfg.AUTO_RESUME:
            torch.save({'epoch': epoch + 1, 'state_dict': model_p.module.state_dict(),
                        'optimizer': optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict(),
                        'scaler': scaler.state_dict(), 'best_perf': best_perf, 'losses': losses_list,
                        'teacher': teacher.state_dict() if teacher is not None else None},
                       ckpt_file + '.tmp')
            os.replace(ckpt_file + '.tmp', ckpt_file)  # a job killed mid-save keeps the previous one

    torch.save(model_p.module.state_dict(), os.path.join(final_output_dir, 'final_state.pth'))
    if teacher is not None:  # the EMA teacher is a second candidate final model
        torch.save(teacher.state_dict(), os.path.join(final_output_dir, 'teacher_final.pth'))
    np.save(os.path.join(final_output_dir, 'losses_P.npy'), np.array(losses_list))
    writer_dict['writer'].close()
    if os.path.exists(ckpt_file):
        os.remove(ckpt_file)  # ~7.5 GB for ViTPose++-H; final_state.pth + model_best.pth are kept
    logger.info('=> saved final_state.pth; best Test100 AP %.4f', best_perf)


if __name__ == '__main__':
    main()
