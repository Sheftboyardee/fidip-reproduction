# MANIFEST.md -- Project Inventory

Everything needed to reproduce the reported results. Paths are relative to the repo root unless
noted.

## Final system (start here)

Best system: **Sapiens2-1B fine-tuned on SyRIP** (rounds 13 and 13b). The published baseline is DarkPose + FiDIP, Test100 AP 93.6.

| Item | Value |
|---|---|
| Result | Test100 AP 95.4 / 95.3, Test400 AP 94.2 / 94.3 (seeds 1 / 2); released FiDIP 93.6 / 90.7 |
| Conditions | Official SyRIP split, COCO keypoint AP, GT boxes, flip test, final-epoch checkpoint |
| Jobs | 11504452 (seed 1), 11504476 (seed 2) |
| Checkpoints | Explorer `~/Infant-Pose-Estimation/output/full-s2ft-s{1,2}/syrip/sapiens2/1b/final_state_bf16.pth` |
| Environment | `~/venvs/sapiens2` from `scripts/sapiens2_setup.sbatch`; exact versions in `requirements-sapiens2.txt` |
| Scores | `results/eval500.csv`, `results/report_data/` |
| Training logs | `results/logs/` |
| Failure analysis | `results/failure_analysis/` (`scripts/failure_analysis.py`) |
| Report | `report/report.tex`; figures from `scripts/make_report_figures.py` |

To reproduce:
1. Build the environment: `sbatch scripts/sapiens2_setup.sbatch`.
2. Train: `sbatch -J fidip-full-s2ft-s1 --gres=gpu:a100:1 --constraint=a100@80g --time=02:30:00 scripts/sapiens2_ft.sbatch full-s2ft-s1 all --epochs 10 --seed 1 --save`.
   Predictions on Test500 land in `output/full-s2ft-s1/syrip/sapiens2/1b/results/`.
3. Check reloading the saved weights: `sbatch -J fidip-s2eval-s1 scripts/sapiens2_eval.sbatch full-s2ft-s1`.
4. Score: `python scripts/bootstrap_ap.py --a <released.json> --b <predictions.json> --tag-a released --tag-b s2ft-s1`.

Later sections are a dated history. "Final reported numbers (2026-10-23)" and the later updates below were each the best result at that date.

## Code

| Item | Source |
|---|---|
| FiDIP code | `https://github.com/ostadabbas/Infant-Pose-Estimation` @ `a76b4f4`, plus local Windows-compat patches (see AGENT.md 2026-10-08) |

## Environment

| Where | Name | Details |
|---|---|---|
| Laptop (Win11, RTX 4060 8GB) | conda `fidip` | Python 3.12.15, torch 2.3.1+cu121, torchvision 0.18.1, numpy 1.26.4; full pins in `requirements-win.txt` |
| Explorer (`sheft.n`) | venv `~/venvs/fidip` | Python 3.12.15 (uv), torch 2.3.1+cu121, numpy 1.26.4; pins in `requirements-explorer.txt` (on cluster) |
| Explorer (`sheft.n`) | venv `~/venvs/sapiens2` | Python 3.12.15, torch 2.7.1+cu126, transformers 5.21.0; pins in `requirements-sapiens2.txt` |

## Datasets

| Dataset | Source | Local path | Cluster path | Status |
|---|---|---|---|---|
| SyRIP (`SyRIP.zip`, 328 MB) | https://coe.northeastern.edu/Research/AClab/SyRIP/ (password from authors) | `data/syrip/` | `~/Infant-Pose-Estimation/data/syrip/` | extracted 2026-10-08 |
| SyRIP for training (`syrip_for_train.zip`, 637 MB) | same | `data/syrip/` (merged with above) | `~/Infant-Pose-Estimation/data/syrip/` | extracted 2026-10-08 |
| `train_pre_infant` (COCO val2017 + SURREAL, real/synthetic labels) | included in `syrip_for_train.zip` | `data/syrip/images/train_pre_infant/` | `~/Infant-Pose-Estimation/data/syrip/images/train_pre_infant/` | 4057 imgs; 3904 used by loader |

**Data actually used by the configs** (`DATASET.ROOT: data/syrip/`):
- `annotations/person_keypoints_train_infant.json` -- 1200 imgs (200 real + 1000 synthetic), from syrip_for_train
- `annotations/person_keypoints_train_pre_infant.json` -- 4057 imgs (loader keeps 3904)
- `annotations/person_keypoints_validate_infant.json` -- 100 imgs = **Test100** (identical keypoints to `annotations/validate100/`, plus a `posture` field)
- Also present from SyRIP.zip: `annotations/{200R,1000S,200R_1000S,validate100,validate500}/`, 500 validate images, 300 extra unannotated train_infant images.

## Pretrained checkpoints

| File | Source | Path | Status |
|---|---|---|---|
| `w48_384×288.pth` (DarkPose HRNet-W48 COCO; note Unicode `×` in name, matches config) | [TGA_models](https://drive.google.com/drive/folders/14kAA1zXuKODYgrRiQmKnVcipbY7RedVV) | `models/coco/w48_384×288.pth` | downloaded 2026-10-08 (255 MB) |
| `hrnet_fidip.pth` | [FiDIP_models](https://drive.google.com/drive/folders/108P-1SnTqaj3xNtjYZ1o7T8z6UvUYuiC) | `models/hrnet_fidip.pth` | downloaded 2026-10-08 (255 MB) |
| `mobile_fidip.pth` | same | `models/mobile_fidip.pth` | downloaded 2026-10-08 (17 MB); Test100 79.2 / Test400 68.3 |

Checkpoint md5 (identical on laptop and Explorer `~/Infant-Pose-Estimation/models/`):
`hrnet_fidip.pth` c1b4994799f08e6cbc44942f44472790, `mobile_fidip.pth` 049dbf22c3131bb72872bc744a1eefe5,
`coco/w48_384×288.pth` 403962abcb38f9a85891c1a7a029dcdc.

## Where run outputs physically live (Explorer)

Since 2026-10-23, `~/Infant-Pose-Estimation/output` -> `/projects/ML_Small_Data/sheft.n/fidip/output`
and `~/Infant-Pose-Estimation/eval500` -> `/projects/ML_Small_Data/sheft.n/fidip/eval500` (symlinks).
All `output/...` checkpoint paths in this file resolve through them unchanged. Copy verified
(840 + 133 files, byte totals equal; md5 of `ft40-s1/.../final_state.pth`
b2b6dfb1f5a8bb5d646f06cfa0a2bc36 identical) before the home copies were removed.
The unused `eval500/*/feature.npy` dumps (40 files, 26.5 GB) were deleted at the user's request;
the prediction JSONs used for every reported number were kept.

## Job scripts

| Script | Purpose |
|---|---|
| `scripts/eval.sbatch <cfg> <model>` | Test100 eval on Explorer |
| `scripts/train.sbatch <cfg> [overrides]` | FiDIP training on Explorer (HRNet init from COCO DarkPose) |

## Evaluation sets

| Set | GT file | Notes |
|---|---|---|
| Test100 | `data/syrip/annotations/person_keypoints_validate_infant.json` (== `validate100/`) | official benchmark; used for best-epoch selection |
| Test500 (corrected) | `data/syrip500/annotations/person_keypoints_validate_infant.json` | shipped `validate500` json has a constant placeholder `area` -> inflated AP; corrected = Test100 anns verbatim + `area = bbox w*h` for the other 400 |
| Test400 | Test500 minus Test100 image ids (computed in `scripts/subset_ap.py`) | **unbiased held-out set** |

All checkpoint scores on these sets: `results/eval500.csv`; summary `results/eval500_summary.md`.

## Derived starting weights

| File | How | Path (Explorer) |
|---|---|---|
| `mobilenet_v2_pose_init.pth` | torchvision ImageNet MobileNetV2 (`mobilenet_v2-b0353104.pth`, download.pytorch.org) mapped by position onto PoseMobileNet `features.*` via `scripts/make_mobilenet_init.py` (job 11386711) | `~/Infant-Pose-Estimation/models/imagenet/` |

## Configs

| Config | Purpose |
|---|---|
| `experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml` | FiDIP HRNet-W48 train/test |
| `experiments/coco/mobilenet/mobile_224x224_adam_lr1e-3_infant.yaml` | FiDIP MobileNetV2 |

## Runs and results

| Run | Config / overrides | Checkpoint | AP | AP50 | AP75 | AR | Notes |
|---|---|---|---|---|---|---|---|
| Paper: DarkPose + FiDIP | -- | -- | 93.6 | 98.5 | 98.5 | 94.6 | published baseline (SyRIP Test100) |
| 2026-10-08 baseline repro (laptop) | `w48_384x288_adam_lr1e-3_infant.yaml`, `TEST.MODEL_FILE models/hrnet_fidip.pth TEST.USE_GT_BBOX True` | `models/hrnet_fidip.pth` (released) | 93.6 | 98.5 | 98.5 | 94.6 | exact match to paper; log `log/syrip/adaptive_pose_hrnet/w48_384x288_adam_lr1e-3_infant_2026-10-08-17-55/` |
| Paper: Pose-MobileNet + FiDIP | -- | -- | 79.2 | 99.0 | 89.4 | 84.1 | published baseline (SyRIP Test100) |
| 2026-10-08 MobileNet repro (laptop) | `mobile_224x224_adam_lr1e-3_infant.yaml`, `TEST.MODEL_FILE models/mobile_fidip.pth TEST.USE_GT_BBOX True` | `models/mobile_fidip.pth` (released) | 79.2 | 99.0 | 89.4 | 84.1 | exact match; log `log/syrip/adaptive_pose_mobile/mobile_224x224_adam_lr1e-3_infant_2026-10-08-17-58/` |
| 2026-10-08 HRNet repro (Explorer job 10929307, V100-32GB) | `scripts/eval.sbatch` + infant cfg | `models/hrnet_fidip.pth` | 93.6 | 98.5 | 98.5 | 94.6 | identical to laptop; log `/scratch/sheft.n/fidip/logs/fidip-eval_10929307.out` |
| 2026-10-23 train repro A: FiDIP (job 11329332) | infant cfg + `TRAIN.CHECKPOINT models/coco/w48_384×288.pth TRAIN.LAMBDA 0.0005 TRAIN.PRE_EPOCH 10 OUTPUT_DIR output/repro_fidip` | Explorer `~/Infant-Pose-Estimation/output/repro_fidip/syrip/adaptive_pose_hrnet/w48_384x288_adam_lr1e-3_infant/model_best.pth` (epoch 12) | 92.0 | 98.4 | 98.4 | 93.6 | best-of-20 on Test100; final epoch 90.4; paper 93.6 |
| 2026-10-23 train repro B: shipped cfg = finetune (job 11329348) | infant cfg + `TRAIN.CHECKPOINT models/coco/w48_384×288.pth OUTPUT_DIR output/repro_shipped` | Explorer `~/Infant-Pose-Estimation/output/repro_shipped/syrip/adaptive_pose_hrnet/w48_384x288_adam_lr1e-3_infant/model_best.pth` (epoch 20) | 93.0 | -- | -- | -- | best-of-20 on Test100; paper Finetune 92.7 |
| 2026-10-23 sweep rep2 (job 11330470) | = run A | `output/rep2/.../model_best.pth` (ep 9) | 91.3 | -- | -- | -- | repeat of A |
| 2026-10-23 sweep rep3 (job 11330481) | = run A | `output/rep3/.../model_best.pth` (ep 8) | 91.9 | -- | -- | -- | repeat of A |
| 2026-10-23 sweep lrstep (job 11330486) | A + `TRAIN.LR_STEP [10,15]` | `output/lrstep/.../model_best.pth` (ep 11) | 91.2 | -- | -- | -- | final 91.2 |
| 2026-10-23 sweep lam5e-3 (job 11330496) | A with `TRAIN.LAMBDA 0.005` | `output/lam5e-3/.../model_best.pth` (ep 1) | 85.5 | -- | -- | -- | diverged, final 73.9 |
| 2026-10-23 sweep lam5e-2 (job 11330524) | A with `TRAIN.LAMBDA 0.05` | `output/lam5e-2/.../model_best.pth` (ep 1) | 86.4 | -- | -- | -- | collapsed, final 0.8 |
| 2026-10-23 round2 B-rep2 (job 11331011) | shipped cfg (repeat of B) | `output/B-rep2/.../model_best.pth` (ep 19) | 92.6 | -- | -- | -- | final 91.5 |
| 2026-10-23 round2 B-rep3 (job 11331062) | shipped cfg (repeat of B) | `output/B-rep3/.../model_best.pth` (ep 10) | 92.7 | -- | -- | -- | final 92.5 |
| 2026-10-23 round2 adv-only (job 11331069) | `TRAIN.PRE_EPOCH 0 TRAIN.LAMBDA 0.0005` | `output/adv-only/.../model_best.pth` (ep 18) | 93.0 | -- | -- | -- | final 92.7; ties best |
| 2026-10-23 round2 pre-only (job 11331060) | `TRAIN.PRE_EPOCH 10 TRAIN.LAMBDA 0.0` | `output/pre-only/.../model_best.pth` (ep 20) | 92.1 | -- | -- | -- | final 92.1 |
| 2026-10-23 round2 bnfreeze (job 11331067) | A + `TRAIN.PRE_BN_FREEZE True` | `output/bnfreeze/.../model_best.pth` (ep 19) | 92.6 | -- | -- | -- | final 91.6 |
| 2026-10-23 round2 conf5e-4 (job 11331081) | A + `TRAIN.ADV_LOSS confusion` (LAMBDA 5e-4) | `output/conf5e-4/.../model_best.pth` (ep 3) | 91.2 | -- | -- | -- | final 86.4 |
| 2026-10-23 round2 conf5e-3 (job 11331086) | ... confusion, LAMBDA 5e-3 | `output/conf5e-3/.../model_best.pth` (ep 1) | 86.3 | -- | -- | -- | collapsed, final 26.9 |
| 2026-10-23 round2 conf5e-2 (job 11331331) | ... confusion, LAMBDA 5e-2 | `output/conf5e-2/.../model_best.pth` (ep 1) | 85.9 | -- | -- | -- | collapsed, final 0.0 |
| 2026-10-23 round3 adv-only-rep2 (job 11332174) | `TRAIN.PRE_EPOCH 0 TRAIN.LAMBDA 0.0005` | `output/adv-only-rep2/.../model_best.pth` (ep 14) | 92.2 | -- | -- | -- | Test400 in results/eval500.csv |
| 2026-10-23 round3 adv-only-rep3 (job 11332165) | same | `output/adv-only-rep3/.../model_best.pth` (ep 6) | 92.2 | -- | -- | -- | |
| 2026-10-23 round3 bnfreeze-rep2 (job 11332183) | A + `TRAIN.PRE_BN_FREEZE True` | `output/bnfreeze-rep2/.../model_best.pth` (ep 12) | 91.6 | -- | -- | -- | |
| 2026-10-23 round3 bnfreeze-rep3 (job 11332187) | same | `output/bnfreeze-rep3/.../model_best.pth` (ep 13) | 91.6 | -- | -- | -- | |
| 2026-10-23 round4 ft40-s1 (job 11333089) | shipped cfg + `TRAIN.END_EPOCH 40 TRAIN.LR_STEP [30]` | `output/ft40-s1/.../final_state.pth` | 92.4 | -- | -- | -- | **best recipe**: Test400 final 90.3 (best ckpt ep 30: Test100 93.1) |
| 2026-10-23 round4 ft40-s2 (job 11333092) | same | `output/ft40-s2/.../final_state.pth` | 91.7 | -- | -- | -- | Test400 final 90.0 |
| 2026-10-23 round4 adv40-s2 (job 11333110) | + `TRAIN.PRE_EPOCH 0 TRAIN.LAMBDA 0.0005` | `output/adv40-s2/.../final_state.pth` | 91.6 | -- | -- | -- | Test400 final 89.7 |
| 2026-10-23 round4 adv40-s3 (job 11334290) | same, excluded slow nodes | `output/adv40-s3/.../final_state.pth` | 91.8 | -- | -- | -- | Test400 final 89.9 |
| 2026-10-23 round4 adv40-s1 (job 11333095) | same | `output/adv40-s1/.../model_best.pth` | 92.2 | -- | -- | -- | TIMEOUT at epoch 26 (V100-PCIe node), excluded |
| 2026-10-26 round5 aug-cj-s2 (job 11386778) | ft40 + `DATASET.COLOR_JITTER 0.4` | `output/aug-cj-s2/.../final_state.pth` | 92.1 | -- | -- | -- | Test400 final 90.3 |
| 2026-10-26 round5 aug-cj-s3 (job 11387264) | same (replaces aug-cj-s1, cancelled on T4) | `output/aug-cj-s3/.../final_state.pth` | 92.5 | -- | -- | -- | Test400 final 90.3 |
| 2026-10-26 round5 aug-geo-s1 (job 11386746) | ft40 + `DATASET.ROT_FACTOR 60 DATASET.SCALE_FACTOR 0.5` | `output/aug-geo-s1/.../final_state.pth` | 92.8 | -- | -- | -- | Test400 final 90.0 |
| 2026-10-26 round5 aug-geo-s2 (job 11386785) | same | `output/aug-geo-s2/.../final_state.pth` | 93.7 | -- | -- | -- | Test400 final 90.3 (Test100 best 93.9) |
| 2026-10-26 round5 aug-light-s1 (job 11386757) | ft40 + `DATASET.ROT_FACTOR 15 DATASET.SCALE_FACTOR 0.15 DATASET.PROB_HALF_BODY 0.0` | `output/aug-light-s1/.../final_state.pth` | 90.7 | -- | -- | -- | Test400 final 88.3 |
| 2026-10-26 round5 aug-light-s2 (job 11386795) | same | `output/aug-light-s2/.../final_state.pth` | 90.5 | -- | -- | -- | Test400 final 88.5 |
| 2026-10-26 round5 mob-ft (job 11386820) | mobile cfg + `MODEL.PRETRAINED models/imagenet/mobilenet_v2_pose_init.pth TRAIN.END_EPOCH 40 TRAIN.LR_STEP [30]` | `output/mob-ft/syrip/adaptive_pose_mobile/mobile_224x224_adam_lr1e-3_infant/final_state.pth` | 57.3 | -- | -- | -- | Test400 43.5; ImageNet init, not comparable to paper |
| 2026-10-26 round5 mob-fidip (job 11386845) | same + `TRAIN.PRE_EPOCH 10 TRAIN.LAMBDA 0.0005` | `output/mob-fidip/.../final_state.pth` | 9.0 | -- | -- | -- | collapsed (best 17.2 @ep20); Test400 8.7 |
| 2026-10-26 round6 mob-ft120-s1 (job 11389498) | mobile cfg + ImageNet init + `TRAIN.END_EPOCH 120 TRAIN.LR_STEP [90,110]` | `output/mob-ft120-s1/syrip/adaptive_pose_mobile/mobile_224x224_adam_lr1e-3_infant/final_state.pth` | 56.7 | -- | -- | -- | Test400 final 43.9 |
| 2026-10-26 round6 mob-ft120-s2 (job 11389506) | mobile cfg + ImageNet init + `same` | `output/mob-ft120-s2/syrip/adaptive_pose_mobile/mobile_224x224_adam_lr1e-3_infant/final_state.pth` | 55.9 | -- | -- | -- | Test400 final 43.8 |
| 2026-10-26 round6 mob-ft200-s1 (job 11389505) | mobile cfg + ImageNet init + `TRAIN.END_EPOCH 200 TRAIN.LR_STEP [150,180]` | `output/mob-ft200-s1/syrip/adaptive_pose_mobile/mobile_224x224_adam_lr1e-3_infant/final_state.pth` | 58.1 | -- | -- | -- | Test400 final 44.6 |
| 2026-10-26 round6 mob-ft200-s2 (job 11389508) | mobile cfg + ImageNet init + `same` | `output/mob-ft200-s2/syrip/adaptive_pose_mobile/mobile_224x224_adam_lr1e-3_infant/final_state.pth` | 59.0 | -- | -- | -- | Test400 final 46.1 (best MobileNet we trained) |

| 2026-10-26 round7 stage A soups (job 11394628; eval 11394636/11394637) | `scripts/weight_average.py` over `scripts/stageA_soups.txt` (32 soups) | `output/soups/<tag>.pth` (7.7 GB on /projects) | see `results/eval500_summary.md` | -- | -- | -- | **A1-four-bn** (uniform ft40-s1/s2 + aug-cj-s2/s3, BN recal) md5 f0f0e80ffa0416b0c4ca438c23e17cca: Test400 91.1, Test100 92.9 |
| 2026-10-26 round7 stage D1 ViTPose zero-shot (job 11394634) | `tools/test_vitpose.py`, models in `scripts/vitpose_models.txt` | HF cache `/projects/ML_Small_Data/sheft.n/fidip/hf_cache` (6.1 GB); predictions `eval500/vitpose/zs-*.json` | ViTPose++-H 92.7 | -- | -- | -- | Test400: ViTPose++-H 90.1, ++-L 89.4, ++-B 87.5; COCO HRNet-W48 zero-shot 86.7 |

| 2026-10-26 round8 vit-* (BN bug; jobs 11395200/03/05/06) | `experiments/syrip/vitpose/vitpose_plus_256x192_infant.yaml` | `output/vit-*` | -- | -- | -- | -- | CANCELLED: decoder BN in train mode collapsed AP. Outputs are invalid; delete. |
| 2026-10-26 round8b vit2-l-lr1e-5-s1 (job 11395499) | vitpose cfg + `MODEL.HF_NAME usyd-community/vitpose-plus-large TRAIN.LR 1e-5` | `output/vit2-l-lr1e-5-s1/syrip/adaptive_pose_vitpose/vitpose_plus_256x192_infant/final_state.pth` (md5 0db210dd2432e9ebe63eed8205ce54bf) | 94.5 | -- | -- | -- | Test400 final 92.0 |
| 2026-10-26 round8c vit2-l-lr1e-5-s2/s3 (jobs 11398128/11398145) | same | `output/vit2-l-lr1e-5-s{2,3}/.../final_state.pth` | 94.4 / 94.6 | -- | -- | -- | Test400 final 91.5 / 92.1 |
| 2026-10-26 round8b vit2-l-lr5e-5-s1 (job 11395455) | vitpose cfg + plus-large, `TRAIN.LR 5e-5` | `output/vit2-l-lr5e-5-s1/.../final_state.pth` | 93.1 | -- | -- | -- | Test400 final 90.5 (overfits) |
| 2026-10-26 round8b vit2-h-lr{5e-5,1e-5}-s1 (jobs 11395479/11395504) | vitpose cfg + plus-huge, 40 ep | `output/vit2-h-lr*-s1/.../model_best.pth` | best 95.0 / 95.7 | -- | -- | -- | TIMEOUT at ep 24. Test400 of best ckpt 92.6 / 92.7. |
| 2026-10-27 round8d **vit2-h20-lr1e-5-s1** (job 11398580) | vitpose cfg + `MODEL.HF_NAME usyd-community/vitpose-plus-huge TRAIN.LR 1e-5 TRAIN.END_EPOCH 20 TRAIN.LR_STEP [15]` | `output/vit2-h20-lr1e-5-s1/syrip/adaptive_pose_vitpose/vitpose_plus_256x192_infant/final_state.pth` (md5 44481f435c577c67720f393e4ef98bb0, 3.6 GB) | 95.1 | 98.3 | 98.3 | 96.3 | **Test400 final 92.7** |
| 2026-10-27 round8d **vit2-h20-lr1e-5-s2** (job 11398625) | same | `output/vit2-h20-lr1e-5-s2/.../final_state.pth` (md5 cefdd05b06a05d72021d35fee428ddba) | 95.4 | 98.4 | 98.4 | 96.5 | **Test400 final 92.8** |

| 2026-10-27 round10 C2 c2-ld{95,90}-lr1e-5-s1, c2-ld90-lr3e-5-s1 (jobs 11400434/11400442/11400455; eval 11400473) | H20 recipe + `TRAIN.LAYER_DECAY 0.95 / 0.9` (`TRAIN.LR 3e-5` for the last one) | `output/c2-*/syrip/adaptive_pose_vitpose/vitpose_plus_256x192_infant/final_state.pth` | 95.6 / 94.7 / 95.3 | -- | -- | -- | Test400 final 92.7 / 92.7 / 92.9: null |
| 2026-10-27 round10 B b-rf{17,50,75,100}-s1 (jobs 11400506/11400510/11400521/11400537; eval 11401502) | H20 recipe + `TRAIN.REAL_FRACTION 0.1667 / 0.5 / 0.75 / 1.0` | `output/b-rf*-s1/.../final_state.pth` | 95.6 / 95.7 / 94.9 / 94.7 | -- | -- | -- | Test400 final 92.6 / 93.0 / 92.8 / 92.7: null; real-only = baseline |
| 2026-10-27 round11 B2 b-rf100-s2 (job 11410487; eval 11410741) | H20 recipe + `TRAIN.REAL_FRACTION 1.0`, second seed | `output/b-rf100-s2/.../final_state.pth` | 94.4 | -- | -- | -- | Test400 final 92.9; real-only 2-seed 92.8 ± 0.1 = baseline |
| 2026-10-27 round11 D3 d3-h384-lr1e-5-s1 (jobs 11410716/11410725/11410732 chained; eval 11410743) | `vitpose_plus_384x288_infant.yaml`, H20 recipe + `TRAIN.GRAD_CKPT True` | `output/d3-h384-lr1e-5-s1/syrip/adaptive_pose_vitpose/vitpose_plus_384x288_infant/final_state.pth` | 95.0 | -- | -- | -- | Test400 final 92.7 (best 92.8): null |

**Reported numbers at 2026-10-23 (superseded; see "Final system" at the top):**
- Reproduction of the paper: released `models/hrnet_fidip.pth` -> Test100 AP 93.6 (exact match; laptop + Explorer job 10929307).
- Best model we trained: `ft40-s1` (job 11333089), `output/ft40-s1/syrip/adaptive_pose_hrnet/w48_384x288_adam_lr1e-3_infant/final_state.pth` on Explorer: Test400 90.3, Test100 92.4 (final epoch). Recipe mean over 2 seeds: Test400 90.2 ± 0.2.
- All checkpoint scores: `results/eval500.csv`; summary table: `results/eval500_summary.md`.

**Update 2026-10-26 (round 7):** best HRNet-W48 model is now the pre-registered uniform soup
`output/soups/A1-four-bn.pth` (md5 f0f0e80ffa0416b0c4ca438c23e17cca; built by job 11394628 from
ft40-s1, ft40-s2, aug-cj-s2, aug-cj-s3 with BN recalibration): **Test400 91.1, Test100 92.9**
(released hrnet_fidip: 90.7 / 93.6).

**Update 2026-10-27 (round 8): final best configuration.**

ViTPose++-H (HF `usyd-community/vitpose-plus-huge`, COCO expert) fine-tuned on SyRIP train_infant.

To reproduce:
1. Run `scripts/vitpose_setup.sbatch`.
2. Train, with `TIME=01:45:00`:
   ```
   sbatch scripts/train.sbatch experiments/syrip/vitpose/vitpose_plus_256x192_infant.yaml \
       MODEL.HF_NAME usyd-community/vitpose-plus-huge TRAIN.LR 1e-5 TRAIN.END_EPOCH 20 \
       TRAIN.LR_STEP [15] OUTPUT_DIR output/<tag>
   ```
3. Score with:
   ```
   CFG=<same cfg> OPTS="MODEL.HF_NAME usyd-community/vitpose-plus-huge" sbatch scripts/eval500.sbatch <list>
   ```

Results, 2 seeds, final checkpoints:
- **Test400 92.8 ± 0.1** (vs 90.7 for the released hrnet_fidip)
- **Test100 95.2 ± 0.2** (vs 93.6)

Checkpoints: `vit2-h20-lr1e-5-s1` and `-s2` (rows above). Per-checkpoint scores are in `results/eval500.csv`.

**Update 2026-10-27 (round 9): best single checkpoint.**
- `output/soups/D-soup-h20.pth` on Explorer, a uniform weight average of `vit2-h20-lr1e-5-s1`
  and `-s2` `final_state.pth`.
  - md5 fb6c2cb06b8e76e7c1223030f4ad0f32, 3.6 GB.
  - Built by `scripts/vit_soups.sbatch` (job 11400326), scored by job 11400360.
  - **Test400 93.1, Test100 95.7.**
- Evaluate it with
  `CFG=experiments/syrip/vitpose/vitpose_plus_256x192_infant.yaml OPTS="MODEL.HF_NAME usyd-community/vitpose-plus-huge"`.
- `D-soup-l3.pth` (ViTPose++-L, 3 seeds): Test400 92.2, Test100 94.7.
- Deleted 2026-10-27, at the user's request: the BN-bug outputs `output/vit-*` (round 8 first
  attempt).

**Update 2026-10-28 (round 12): best system = H20 + Sapiens2-1b keypoint ensemble (no extra training).**

Recipe:
1. Run the H20 checkpoint (or D-soup-h20) with the usual eval on GT boxes.
2. Run `facebook/sapiens2-pose-1b` zero-shot with `tools/test_sapiens2.py` (venv `~/venvs/sapiens2`, flip test, GT boxes).
3. Average the two with `scripts/ensemble_kpts.py` (0.5/0.5).

Scores:

| System | Test400 | Test400_indep | Test100 |
|---|---|---|---|
| H20-s1 + S2-1b | **93.8** | 93.9 | 95.3 |
| H20-s2 + S2-1b | **93.8** | 93.8 | 95.6 |
| D-soup + S2-1b | **94.0** | 94.0 | 95.4 |

- Grouped-CV (200 real images, out-of-fold): 95.94 vs 94.39 for H20 alone.
- Round 13: fine-tuned S2-1b alone 96.10, H20 + S2-1b-ft 96.05 (no advance over 95.94; recommended system unchanged).

**Update 2026-11-03 (round 13b): best single model = Sapiens2-1b fine-tuned on SyRIP; it replaces the two-model ensemble.**

- Checkpoints (Explorer, ~3 GB bf16 each): `~/Infant-Pose-Estimation/output/full-s2ft-s{1,2}/syrip/sapiens2/1b/final_state_bf16.pth`
  (job 11504452 / 11504476; `scripts/sapiens2_ft.sbatch full-s2ft-s<seed> all --epochs 10 --seed <seed> --save`).
- Inference: `~/venvs/sapiens2/bin/python tools/test_sapiens2.py --model facebook/sapiens2-pose-1b --checkpoint <pth> --gt <coco json> --images <dir>` (flip test, GT boxes). The reported scores come from the in-process prediction at the end of training; reloading via `--checkpoint` has not yet been run end-to-end.
- Test500 predictions: `output/audit12/results/full-s2ft-s{1,2}.json` (local, gitignored).

| System | Test400 | Test400_indep | Test100 |
|---|---|---|---|
| S2-1b-ft seed 1 | **94.2** | 94.2 | 95.4 |
| S2-1b-ft seed 2 | **94.3** | 94.3 | 95.3 |
| D-soup + S2-1b-zs (previous best) | 94.0 | 94.0 | 95.4 |

- Pre-registered non-inferiority (margin 0.5 AP on Test400, both seeds) passed: diff +0.20 [-0.10, +0.67] and +0.25 [-0.15, +0.88].
- Averaging in D-soup adds nothing (Test400 94.2 / 94.0).

Round 12 artifacts:

| Location | Contents |
|---|---|
| `output/audit12/` (local; gitignored, SyRIP-derived) | overlap/ (pairs, groups, test subsets, eligible extras) |
| | oof/ (pooled CV predictions) |
| | sapiens2/ (zero-shot predictions + overlays) |
| | results/ (Test500 prediction files incl. ensembles) |
| | labels/ (audit candidates, review page) |
| | mt/ (eligible boxes) |
| Explorer | `output/cv-{h20,rs,mt}-f<k>/` (fold checkpoints, ~3.6 GB each) |
| | `output/audit12/sapiens2/` |
| | `output/cv-s2ft-f<k>/syrip/sapiens2/1b/` (round 13 fine-tune: train.log + held-out predictions; weights not saved) |
| | HF cache with sapiens2-pose-{0.4b,1b} |
