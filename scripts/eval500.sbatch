#!/bin/bash
# Evaluate many checkpoints on corrected Test500 and report Test500 / Test400 / Test100 AP.
# Usage: [CFG=<cfg.yaml>] sbatch scripts/eval500.sbatch <list.txt>   (lines: "<tag> <checkpoint.pth>")
# CFG defaults to the HRNet infant config; use the mobilenet one for adaptive_pose_mobile checkpoints.
# OPTS="..." appends cfg overrides to every inference call (e.g. OPTS="MODEL.HF_NAME usyd-community/vitpose-plus-huge").
# Results appended to eval500/results_<jobid>.csv (tag,set,AP,AP50,AP75,AR,AR50,AR75); combine with
# `cat eval500/results_*.csv`. (The old shared eval500/results.csv lost rows -- don't use it.)
# data/syrip500 = Test500 images (symlink) + corrected GT: see AGENT.md 2026-10-23 Test500 note.
#SBATCH --job-name=fidip-eval500
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/fidip/logs/%x_%j.out

set -uo pipefail
LIST=${1:?list file required}
cd ~/Infant-Pose-Estimation
PY=~/venvs/fidip/bin/python
# ViTPose checkpoints (stage D) are pre-downloaded here by scripts/vitpose_setup.sbatch
export HF_HOME=/projects/ML_Small_Data/sheft.n/fidip/hf_cache HF_HUB_OFFLINE=1
CFG=${CFG:-experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml}
# results land in <OUTPUT_DIR>/syrip/<MODEL.NAME>/<cfg basename>/
MODEL_NAME=$(grep -m1 -E '^\s+NAME:' $CFG | awk '{print $2}')
RUN_DIR=syrip/$MODEL_NAME/$(basename $CFG .yaml)
echo "cfg=$CFG run_dir=$RUN_DIR"
GT500=data/syrip500/annotations/person_keypoints_validate_infant.json
GT100=data/syrip/annotations/validate100/person_keypoints_validate_infant.json
mkdir -p eval500
echo "job $SLURM_JOB_ID on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader)"
while read -r TAG CKPT; do
  [ -z "${TAG:-}" ] || [ "${TAG:0:1}" = "#" ] && continue
  if cat eval500/results*.csv 2>/dev/null | grep -q "^$TAG,test500,"; then echo "$TAG: already scored, skipping"; continue; fi
  OUT=eval500/$TAG
  $PY tools/test_adaptive_model.py --cfg $CFG TEST.MODEL_FILE "$CKPT" TEST.USE_GT_BBOX True \
      DATASET.ROOT data/syrip500/ OUTPUT_DIR $OUT LOG_DIR $OUT/log DEBUG.DEBUG False WORKERS 6 ${OPTS:-} \
      > $OUT.log 2>&1 || { echo "$TAG: eval FAILED (see $OUT.log)"; continue; }
  RES=$OUT/$RUN_DIR/results/keypoints_validate_infant_results_0.json
  # One csv per job: concurrent jobs appending to a single shared csv lost rows (2026-10-23).
  $PY scripts/subset_ap.py $GT500 $GT100 $RES $TAG | tee -a eval500/results_$SLURM_JOB_ID.csv
  # test_adaptive_model.py also dumps a ~600 MB feature.npy per checkpoint that nothing uses.
  rm -f $OUT/$RUN_DIR/feature.npy
done < "$LIST"
