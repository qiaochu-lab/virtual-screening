#!/bin/bash
set -x
B=/data/work/vs-benchmark
CONDA=/opt/conda/bin/conda
E=/data/work/envs/conglude
# 照搬官方 setup_env.sh 的 cu121 分支（驱动 560.35 不支持 cu128），
# 只把 -n 具名环境换成 -p，保证一切落在 xqc 下
$CONDA create -y -p $E python=3.11 || exit 1
$E/bin/pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu121 || exit 1
$E/bin/pip install --no-cache-dir torch-geometric torch-scatter \
  -f https://data.pyg.org/whl/torch-2.1.2+cu121.html || exit 1
$CONDA install -y -p $E -c conda-forge rdkit=2024.03.5 libgcc-ng || exit 1
cd $B/code/conglude && $E/bin/pip install --no-cache-dir -e . || exit 1
LD_LIBRARY_PATH=$E/lib $E/bin/python -c "import torch, torch_scatter, conglude; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
