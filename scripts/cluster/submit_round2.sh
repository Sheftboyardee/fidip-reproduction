#!/bin/bash
# Round 2: isolate FiDIP ingredients, B repeats, BN-freeze pre-stage, bounded adversarial loss.
cd ~/Infant-Pose-Estimation
C=experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml
submit() { tag=$1; shift
  if sacct -u $USER -S now-2days -X -n -o JobName%30 | grep -q "fidip-$tag\$"; then echo "$tag: already submitted, skipping"; return; fi
  id=$(sbatch --parsable -J fidip-$tag --time=01:00:00 scripts/train.sbatch $C "$@" OUTPUT_DIR output/$tag LOG_DIR log/$tag 2>&1); echo "$tag: $id"; sleep 5; }
submit B-rep2
submit B-rep3
submit pre-only     TRAIN.PRE_EPOCH 10 TRAIN.LAMBDA 0.0
submit adv-only     TRAIN.PRE_EPOCH 0  TRAIN.LAMBDA 0.0005
submit bnfreeze     TRAIN.PRE_EPOCH 10 TRAIN.LAMBDA 0.0005 TRAIN.PRE_BN_FREEZE True
submit conf5e-4     TRAIN.PRE_EPOCH 10 TRAIN.LAMBDA 0.0005 TRAIN.ADV_LOSS confusion
submit conf5e-3     TRAIN.PRE_EPOCH 10 TRAIN.LAMBDA 0.005  TRAIN.ADV_LOSS confusion
submit conf5e-2     TRAIN.PRE_EPOCH 10 TRAIN.LAMBDA 0.05   TRAIN.ADV_LOSS confusion
