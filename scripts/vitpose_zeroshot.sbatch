#!/bin/bash
# PLAN.md stage D1: zero-shot COCO ViTPose models on corrected Test500 with GT boxes + flip test.
# Scores Test500/Test400/Test100 into eval500/results_<jobid>.csv (same format as eval500.sbatch).
#SBATCH --job-name=fidip-vitpose-zs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/fidip/logs/%x_%j.out

set -uo pipefail
cd ~/Infant-Pose-Estimation
PY=~/venvs/fidip/bin/python
export HF_HOME=/projects/ML_Small_Data/sheft.n/fidip/hf_cache HF_HUB_OFFLINE=1
GT500=data/syrip500/annotations/person_keypoints_validate_infant.json
GT100=data/syrip/annotations/validate100/person_keypoints_validate_infant.json
mkdir -p eval500/vitpose
echo "job $SLURM_JOB_ID on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader)"
for M in $(cat scripts/vitpose_models.txt); do
  TAG=zs-$(basename $M)
  RES=eval500/vitpose/$TAG.json
  $PY tools/test_vitpose.py --model $M --gt $GT500 --images data/syrip500/images/validate_infant --out $RES \
    || { echo "$TAG: FAILED"; continue; }
  $PY scripts/subset_ap.py $GT500 $GT100 $RES $TAG | tee -a eval500/results_$SLURM_JOB_ID.csv
done
