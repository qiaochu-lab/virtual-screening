#!/bin/bash
set -x
B=/data/work/vs-benchmark
CONDA=/opt/conda/bin/conda
E=/data/work/envs/litenclip
# environment.yml 里是 cu124，但本机驱动 560.35 支持 CUDA 12.6，cu124 可用
$CONDA create -y -p $E python=3.10.18 || exit 1
$E/bin/pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu124 || exit 1
# torch-geometric 全家桶必须匹配 torch 2.4 + cu124 的 wheel 索引
$E/bin/pip install --no-cache-dir torch-geometric==2.7.0 || exit 1
$E/bin/pip install --no-cache-dir torch_scatter torch_cluster torch_sparse torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.4.0+cu124.html || exit 1
$E/bin/pip install --no-cache-dir "numpy==1.26.4" "rdkit==2025.9.1" || exit 1
# unicore 无 PyPI 包，从源码装（服务器 GitHub 不通，用已下载的副本）
if [ -d "$B/code/Uni-Core" ]; then
  $E/bin/pip install --no-cache-dir $B/code/Uni-Core || exit 1
else
  echo "!! Uni-Core 源码不存在，需另外提供"
fi
LD_LIBRARY_PATH=$E/lib $E/bin/python -c "
import torch, torch_geometric, torch_scatter, torch_cluster, rdkit
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('pyg', torch_geometric.__version__, 'rdkit', rdkit.__version__)
import unicore; print('unicore OK')
"
