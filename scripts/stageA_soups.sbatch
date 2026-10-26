#!/bin/bash
# PLAN.md stage A: build weight soups / WiSE-FT interpolations listed in scripts/stageA_soups.txt.
# Each line is built twice: <tag>-raw (BN stats averaged) and <tag>-bn (BN recalibrated on
# train_infant). Writes output/soups/<tag>.pth plus eval lists output/soups/eval_{raw,bn}.txt
# for scripts/eval500.sbatch (chain with --dependency=afterok).
#SBATCH --job-name=fidip-soups
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/fidip/logs/%x_%j.out

set -euo pipefail
cd ~/Infant-Pose-Estimation
PY=~/venvs/fidip/bin/python
CFG=experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml
SPEC=${1:-scripts/stageA_soups.txt}
D=output/soups
mkdir -p $D
: > $D/eval_raw.txt
echo "coco-w48 models/coco/w48_384×288.pth" > $D/eval_bn.txt   # zero-shot reference for stage D1
echo "job $SLURM_JOB_ID on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader)"
grep -v '^\s*#' "$SPEC" | while read -r TAG SPECS; do
  [ -z "$TAG" ] && continue
  $PY scripts/weight_average.py --cfg $CFG --out $D/$TAG-raw.pth $SPECS
  echo "$TAG-raw $D/$TAG-raw.pth" >> $D/eval_raw.txt
  $PY scripts/weight_average.py --cfg $CFG --out $D/$TAG-bn.pth --recal-bn $SPECS --opts WORKERS 6
  echo "$TAG-bn $D/$TAG-bn.pth" >> $D/eval_bn.txt
done
echo done
