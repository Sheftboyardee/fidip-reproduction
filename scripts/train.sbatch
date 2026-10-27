#!/bin/bash
# Train FiDIP. Initialises from the COCO DarkPose checkpoint by default (the shipped config
# leaves MODEL.PRETRAINED and TRAIN.CHECKPOINT empty, i.e. random init).
# Usage: sbatch scripts/train.sbatch <cfg.yaml> [extra cfg overrides...]
#   e.g. sbatch scripts/train.sbatch experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml
#   Use a different OUTPUT_DIR per experiment, e.g. OUTPUT_DIR output/exp_lr3e-4
# Output/checkpoints: ~/Infant-Pose-Estimation/<OUTPUT_DIR>/syrip/<model>/<cfg name>/
#SBATCH --job-name=fidip-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/fidip/logs/%x_%j.out

set -euo pipefail
CFG=${1:?cfg yaml required}; shift
cd ~/Infant-Pose-Estimation
PY=~/venvs/fidip/bin/python
# ViTPose checkpoints (stage D) are pre-downloaded here by scripts/vitpose_setup.sbatch
export HF_HOME=/projects/ML_Small_Data/sheft.n/fidip/hf_cache HF_HUB_OFFLINE=1
case "$CFG" in
  *mobile*) SCRIPT=tools/train_adaptive_model_mobile.py; DEFAULT_INIT="" ;;  # no COCO MobileNet ckpt yet
  *vitpose*) SCRIPT=tools/train_vitpose.py; DEFAULT_INIT="" ;;  # weights come from MODEL.HF_NAME
  *)        SCRIPT=tools/train_adaptive_model_hrnet.py; DEFAULT_INIT="models/coco/w48_384×288.pth" ;;
esac
INIT=${INIT-$DEFAULT_INIT}   # override: INIT=path sbatch ...   (INIT= to start from scratch)
INIT_OPTS=()
[ -n "$INIT" ] && INIT_OPTS=(TRAIN.CHECKPOINT "$INIT")
# One job per OUTPUT_DIR: a duplicate submission (sbatch can report failure yet still queue the
# job) would otherwise auto-resume from / overwrite the same checkpoint.
OUT=output; ARGS=("$@")
for ((k=0; k<${#ARGS[@]}-1; k++)); do [ "${ARGS[k]}" = OUTPUT_DIR ] && OUT=${ARGS[k+1]}; done
mkdir -p "$OUT"
exec 9>"$OUT/.train.lock"
flock -n 9 || { echo "another job is already training into $OUT -- exiting"; exit 1; }
echo "job $SLURM_JOB_ID on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "script=$SCRIPT cfg=$CFG init=${INIT:-<none>} overrides=$*"
$PY $SCRIPT --cfg "$CFG" WORKERS 6 "${INIT_OPTS[@]}" "$@"
