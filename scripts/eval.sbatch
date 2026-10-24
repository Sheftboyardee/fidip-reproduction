#!/bin/bash
# Evaluate a FiDIP checkpoint on SyRIP Test100 (GT boxes).
# Usage: sbatch scripts/eval.sbatch <cfg.yaml> <model.pth> [extra cfg overrides...]
#SBATCH --job-name=fidip-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/fidip/logs/%x_%j.out

set -euo pipefail
CFG=${1:?cfg yaml required}; MODEL=${2:?model .pth required}; shift 2
cd ~/Infant-Pose-Estimation
PY=~/venvs/fidip/bin/python
echo "job $SLURM_JOB_ID on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader)"
$PY tools/test_adaptive_model.py --cfg "$CFG" \
    TEST.MODEL_FILE "$MODEL" TEST.USE_GT_BBOX True WORKERS 4 "$@"
