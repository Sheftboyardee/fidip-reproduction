#!/bin/bash
# Round 12, experiment 4: separate venv for Sapiens2 (needs Python >= 3.12, torch >= 2.7 and a
# transformers release with Sapiens2ForPoseEstimation, i.e. 5.x), plus the pose checkpoints in the
# /projects HF cache. ~/venvs/fidip (torch 2.3.1, transformers 4.53.3) is left untouched.
# cu126 wheels still include sm_70, so V100s work.
# Usage: sbatch scripts/sapiens2_setup.sbatch
#SBATCH --job-name=fidip-sapiens2-setup
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/fidip/logs/%x_%j.out

set -euo pipefail
UV=~/.local/bin/uv
VENV=~/venvs/sapiens2
export HF_HOME=/projects/ML_Small_Data/sheft.n/fidip/hf_cache
[ -x $VENV/bin/python ] || $UV venv -q --python 3.12 $VENV
export VIRTUAL_ENV=$VENV
$UV pip install -q "torch==2.7.1" "torchvision==0.22.1" --index-url https://download.pytorch.org/whl/cu126
$UV pip install -q --only-binary :all: transformers accelerate safetensors huggingface_hub \
    pycocotools opencv-python-headless scipy pillow json_tricks
$VENV/bin/python -c "from transformers import Sapiens2ForPoseEstimation" 2>/dev/null || {
  echo "released transformers has no Sapiens2 -- installing from main"
  $UV pip install -q "git+https://github.com/huggingface/transformers"; }
for M in facebook/sapiens2-pose-0.4b facebook/sapiens2-pose-1b; do
  $VENV/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('$M', allow_patterns=['*.json', 'model.safetensors', '*.md']))"
done
$VENV/bin/python -c "import transformers, torch; from transformers import Sapiens2ForPoseEstimation; print('OK', transformers.__version__, torch.__version__)"
$UV pip freeze > ~/Infant-Pose-Estimation/output/audit12/requirements-sapiens2.txt
