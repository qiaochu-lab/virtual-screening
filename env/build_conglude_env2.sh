#!/bin/bash
set -x
B=/data/work/vs-benchmark
E=/data/work/envs/conglude
# 官方 setup_env.sh 用 conda 装 rdkit（注明「指纹计数跨版本会变」，所以版本必须钉死）。
# conda 求解在本机跑了 20 分钟没出结果，改用 PyPI 同版本：conda 的 2024.03.5 == PyPI 的 2024.3.5
$E/bin/pip install --no-cache-dir "rdkit==2024.3.5" || exit 1
cd $B/code/conglude && $E/bin/pip install --no-cache-dir -e . || exit 1
$E/bin/python -c "
import torch, torch_scatter, rdkit, conglude
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('rdkit', rdkit.__version__)
print('conglude OK')
"
