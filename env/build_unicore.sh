#!/bin/bash
set -x
B=/data/work/vs-benchmark
E=/data/work/envs/litenclip
# Uni-Core 的 setup.py 在构建期 import torch，而 pip 默认在隔离环境里构建
# （那里没有 torch）→ ModuleNotFoundError。--no-build-isolation 让它用当前环境。
# 它要编译 CUDA 扩展，耗时较久。
$E/bin/pip install --no-cache-dir --no-build-isolation $B/code/Uni-Core || exit 1
LD_LIBRARY_PATH=$E/lib $E/bin/python -c "
import torch, unicore, torch_geometric, torch_scatter, rdkit
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('unicore OK, pyg', torch_geometric.__version__, 'rdkit', rdkit.__version__)
"
