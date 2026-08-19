#!/bin/bash
set -x
B=/data/work/vs-benchmark
CONDA=/opt/conda/bin/conda
E=/data/work/envs/conplex
$CONDA create -y -p $E python=3.10 || exit 1
$E/bin/pip install --no-cache-dir torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121 || exit 1
$E/bin/pip install --no-cache-dir conplex-dti || exit 1
mkdir -p $B/ckpt/conplex
cd $B/ckpt/conplex && wget -q -c \
  https://cb.csail.mit.edu/cb/conplex/data/models/BindingDB_ExperimentalValidModel.pt
ls -lh $B/ckpt/conplex
$E/bin/python -c "import conplex_dti, torch; print('conplex OK, torch', torch.__version__, 'cuda', torch.cuda.is_available())"
