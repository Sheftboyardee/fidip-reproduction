#!/bin/bash
# Round 12, experiment 4: zero-shot Sapiens2 (0.4B, 1B) with GT boxes and flip test on
#  * the 200 labelled real training images (clean for a model never trained on them; compare with
#    the grouped-CV out-of-fold AP of our fine-tuned ViTPose++-H)
#  * corrected Test500 -> Test400 / Test100 (scripts/subset_ap.py), as for every other model.
# Predictions: output/audit12/sapiens2/<model>_{real200,test500}.json; scores in the job log.
#SBATCH --job-name=fidip-sapiens2-zs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/fidip/logs/%x_%j.out

set -euo pipefail
cd ~/Infant-Pose-Estimation
export HF_HOME=/projects/ML_Small_Data/sheft.n/fidip/hf_cache HF_HUB_OFFLINE=1
S2=~/venvs/sapiens2/bin/python
PY=~/venvs/fidip/bin/python
O=output/audit12/sapiens2
mkdir -p $O
echo "job $SLURM_JOB_ID on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader)"
for M in 0.4b 1b; do
  $S2 tools/test_sapiens2.py --model facebook/sapiens2-pose-$M --gt data/syrip_cv/real200_validate.json \
      --images data/syrip/images/train_infant --out $O/${M}_real200.json --overlays 8
  $S2 tools/test_sapiens2.py --model facebook/sapiens2-pose-$M \
      --gt data/syrip500/annotations/person_keypoints_validate_infant.json \
      --images data/syrip500/images/validate_infant --out $O/${M}_test500.json
  $PY scripts/subset_ap.py data/syrip500/annotations/person_keypoints_validate_infant.json \
      data/syrip/annotations/validate100/person_keypoints_validate_infant.json $O/${M}_test500.json sapiens2-$M
  $PY - <<PYEOF
import contextlib, io
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
with contextlib.redirect_stdout(io.StringIO()):
    gt = COCO('data/syrip_cv/real200_validate.json'); dt = gt.loadRes('$O/${M}_real200.json')
    ev = COCOeval(gt, dt, 'keypoints'); ev.params.useSegm = None; ev.evaluate(); ev.accumulate(); ev.summarize()
print('sapiens2-$M,real200,%.4f' % ev.stats[0])
PYEOF
done
