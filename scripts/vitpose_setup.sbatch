#!/bin/bash
# PLAN.md stage D: add HF transformers to ~/venvs/fidip and pre-download the ViTPose checkpoints
# into the /projects HF cache (no downloads on the login node: its proxy stalls them).
#SBATCH --job-name=fidip-vitpose-setup
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/fidip/logs/%x_%j.out

set -euo pipefail
export VIRTUAL_ENV=~/venvs/fidip
export HF_HOME=/projects/ML_Small_Data/sheft.n/fidip/hf_cache
~/.local/bin/uv pip install -q --only-binary :all: "transformers>=4.48,<4.54"
for M in $(cat ~/Infant-Pose-Estimation/scripts/vitpose_models.txt); do
  ~/venvs/fidip/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('$M'))"
done
~/venvs/fidip/bin/python -c "import transformers, torch; print('OK', transformers.__version__, torch.__version__)"
cd ~/Infant-Pose-Estimation && ~/.local/bin/uv pip freeze > requirements-explorer.txt
