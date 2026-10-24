#!/bin/bash
set -euo pipefail
echo "node $(hostname)"
~/.local/bin/uv venv -q --python 3.12 ~/venvs/fidip
export VIRTUAL_ENV=~/venvs/fidip
UV=~/.local/bin/uv
$UV pip install -q torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
$UV pip install -q --only-binary :all: "numpy==1.26.4" "opencv-python-headless<4.11" easydict json-tricks yacs pycocotools "scipy<1.15" "scikit-image<0.25" "matplotlib<3.10" "pandas<2.3" tensorboardX tensorboard "cython<3.1" shapely dsntnn pyyaml "pillow<11" python-dotenv wandb
~/venvs/fidip/bin/python -c "import torch,cv2,pycocotools,numpy;print(\"OK\", torch.__version__, torch.version.cuda, cv2.__version__, numpy.__version__)"
$UV pip freeze > ~/Infant-Pose-Estimation/requirements-explorer.txt
