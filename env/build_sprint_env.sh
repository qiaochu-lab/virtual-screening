#!/bin/bash
set -x
B=/data/work/vs-benchmark
E=/data/work/envs/sprint
# ml-pyxis 是 git 依赖，服务器访问 GitHub 不稳，改用本地 vendored 副本
$E/bin/pip install --no-cache-dir $B/code/ml-pyxis || exit 1
# 把 pyproject 里的 git 依赖去掉，其余照装
cd $B/code/panspecies-dti
sed -i 's|^ *"ml-pyxis @ git+.*$||' pyproject.toml
$E/bin/pip install --no-cache-dir -e . || exit 1
$E/bin/python -c "import torch, ultrafast; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
