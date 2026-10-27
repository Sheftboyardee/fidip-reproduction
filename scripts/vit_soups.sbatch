#!/bin/bash
# PLAN.md stage A applied to D: uniform weight soups of fine-tuned ViTPose++ seeds (CPU only;
# no BN recalibration -- the decoder BN is frozen during ViTPose fine-tuning).
# Writes output/soups/<tag>.pth and eval lists output/soups/eval_vit_{h,l}.txt for eval500.sbatch.
#SBATCH --job-name=fidip-vit-soups
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/fidip/logs/%x_%j.out

set -euo pipefail
cd ~/Infant-Pose-Estimation
PY=~/venvs/fidip/bin/python
export HF_HOME=/projects/ML_Small_Data/sheft.n/fidip/hf_cache HF_HUB_OFFLINE=1
CFG=experiments/syrip/vitpose/vitpose_plus_256x192_infant.yaml
R=syrip/adaptive_pose_vitpose/vitpose_plus_256x192_infant/final_state.pth
D=output/soups
mkdir -p $D
$PY scripts/weight_average.py --cfg $CFG --out $D/D-soup-h20.pth \
    1:output/vit2-h20-lr1e-5-s1/$R 1:output/vit2-h20-lr1e-5-s2/$R
echo "D-soup-h20 $D/D-soup-h20.pth" > $D/eval_vit_h.txt
$PY scripts/weight_average.py --cfg $CFG --out $D/D-soup-l3.pth \
    1:output/vit2-l-lr1e-5-s1/$R 1:output/vit2-l-lr1e-5-s2/$R 1:output/vit2-l-lr1e-5-s3/$R
echo "D-soup-l3 $D/D-soup-l3.pth" > $D/eval_vit_l.txt
echo done
