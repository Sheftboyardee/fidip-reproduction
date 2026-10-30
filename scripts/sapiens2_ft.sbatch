#!/bin/bash
# Round 13: fine-tune Sapiens2 pose on one grouped-CV fold (tools/train_sapiens2.py) on an H200.
# Usage: sbatch -J fidip-<tag>-f<k> scripts/sapiens2_ft.sbatch <tag> <k> [extra train_sapiens2.py args]
#   e.g. sbatch -J fidip-cv-s2ft-f0 scripts/sapiens2_ft.sbatch cv-s2ft 0 --epochs 8
#   smoke: sbatch -J fidip-s2ft-smoke --time=00:30:00 scripts/sapiens2_ft.sbatch s2ft-smoke 0 --smoke
#   full:  sbatch -J fidip-full-s2ft-s1 scripts/sapiens2_ft.sbatch full-s2ft-s1 all --epochs 10 --seed 1 --save
#          (K=all: all 1200 training images, predictions on Test500 in output/<tag>/syrip/sapiens2/<model>/results/)
# Held-out predictions land in output/<tag>-f<k>/syrip/sapiens2/<model>/results/ (scripts/cv_score.py).
#SBATCH --job-name=fidip-s2ft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=01:45:00
#SBATCH --output=/scratch/%u/fidip/logs/%x_%j.out

set -euo pipefail
cd ~/Infant-Pose-Estimation
export HF_HOME=/projects/ML_Small_Data/sheft.n/fidip/hf_cache HF_HUB_OFFLINE=1
S2=~/venvs/sapiens2/bin/python
TAG=$1; K=$2; shift 2
M=${MODEL:-1b}
ROOT=${ROOT:-data/syrip_cv}
echo "job $SLURM_JOB_ID on $(hostname): $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
if [ "$K" = "all" ]; then
  # full training set (1200 images), predictions on Test500
  OUT=output/$TAG/syrip/sapiens2/$M
  ARGS=(--model facebook/sapiens2-pose-$M --train data/syrip/annotations/person_keypoints_train_infant.json
        --images data/syrip/images/train_infant --heldout data/syrip500/annotations/person_keypoints_validate_infant.json
        --heldout-images data/syrip500/images/validate_infant --out $OUT)
else
  OUT=output/$TAG-f$K/syrip/sapiens2/$M
  ARGS=(--model facebook/sapiens2-pose-$M --train $ROOT/f$K/annotations/person_keypoints_train_infant.json
        --images data/syrip/images/train_infant --heldout $ROOT/f$K/annotations/person_keypoints_heldout_infant.json
        --out $OUT)
fi
if [ "${1:-}" = "--smoke" ]; then
  shift
  $S2 tools/train_sapiens2.py "${ARGS[@]}" --check-targets 30
  $S2 tools/train_sapiens2.py "${ARGS[@]}" --max-steps 60 "$@"
else
  $S2 tools/train_sapiens2.py "${ARGS[@]}" "$@"
fi
