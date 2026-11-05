#!/bin/bash
# Round 13b: reload saved fine-tuned Sapiens2 weights and predict Test500 (GT boxes, flip test).
# Checks that final_state_bf16.pth reproduces the in-process predictions written at the end of training.
# Usage: sbatch -J fidip-s2eval-s1 scripts/sapiens2_eval.sbatch full-s2ft-s1
# Predictions: output/<tag>/syrip/sapiens2/<model>/results/test500_reload.json
#SBATCH --job-name=fidip-s2eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/fidip/logs/%x_%j.out

set -euo pipefail
cd ~/Infant-Pose-Estimation
export HF_HOME=/projects/ML_Small_Data/sheft.n/fidip/hf_cache HF_HUB_OFFLINE=1
TAG=$1
M=${MODEL:-1b}
O=output/$TAG/syrip/sapiens2/$M
echo "job $SLURM_JOB_ID on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader)"
~/venvs/sapiens2/bin/python tools/test_sapiens2.py --model facebook/sapiens2-pose-$M --checkpoint $O/final_state_bf16.pth \
    --gt data/syrip500/annotations/person_keypoints_validate_infant.json \
    --images data/syrip500/images/validate_infant --out $O/results/test500_reload.json
