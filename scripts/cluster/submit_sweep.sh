#!/bin/bash
# Submit FiDIP follow-up runs one at a time; never blind-retry (sbatch may queue despite an error).
cd ~/Infant-Pose-Estimation
C=experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml
BASE="TRAIN.LAMBDA 0.0005 TRAIN.PRE_EPOCH 10"
submit() { tag=$1; shift
  if squeue -u $USER -h -o %j | grep -qx "fidip-$tag" || sacct -u $USER -S now-1days -X -n -o JobName%30 | grep -q "fidip-$tag\$"; then echo "$tag: already submitted, skipping"; return; fi
  id=$(sbatch --parsable -J fidip-$tag --time=01:00:00 scripts/train.sbatch $C "$@" OUTPUT_DIR output/$tag LOG_DIR log/$tag 2>&1) ; echo "$tag: $id"; sleep 5; }
submit rep2    $BASE
submit rep3    $BASE
submit lrstep  $BASE TRAIN.LR_STEP "[10,15]"
submit lam5e-3 TRAIN.LAMBDA 0.005 TRAIN.PRE_EPOCH 10
submit lam5e-2 TRAIN.LAMBDA 0.05 TRAIN.PRE_EPOCH 10
