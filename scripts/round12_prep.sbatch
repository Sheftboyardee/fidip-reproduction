#!/bin/bash
# Round 12 inputs that need ViTPose++-H zero-shot (weights cached on /projects):
#  * person boxes for the 52 eligible unannotated images (experiment 3, tools/estimate_boxes.py)
#  * zero-shot predictions on the 200 labelled real images, a second opinion for the label audit
#    (experiment 1, tools/label_audit.py)
#SBATCH --job-name=fidip-r12-prep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/%u/fidip/logs/%x_%j.out

set -euo pipefail
cd ~/Infant-Pose-Estimation
PY=~/venvs/fidip/bin/python
export HF_HOME=/projects/ML_Small_Data/sheft.n/fidip/hf_cache HF_HUB_OFFLINE=1
mkdir -p output/audit12/mt output/audit12/labels
$PY tools/estimate_boxes.py --images output/audit12/overlap/eligible_unlabeled.json --out output/audit12/mt/eligible_boxes.json | tail -3
$PY tools/test_vitpose.py --model usyd-community/vitpose-plus-huge --gt data/syrip_cv/real200_validate.json \
    --images data/syrip/images/train_infant --out output/audit12/labels/zs_h.json
