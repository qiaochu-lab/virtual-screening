#!/bin/bash
# Boltz-2 逐配体跑 16 个 FEP 体系（461 个复合物）。
# 目的：把「检索模型 / 共折叠 / 物理方法」三类放到同一批体系上比较——
# 现有的三个数字（检索 ρ≈0.4、Boltz-2 跨靶点 ρ=0.404、FEP+ 文献 0.6-0.8）
# 口径不同、严格说不可比。
# 用 GPU 4/6/7（0-3 是别人的，5 在跑 HypSeek）。
set -u
E=/data/work/envs/boltz2
B=/data/work/vs-benchmark
export HF_ENDPOINT=https://hf-mirror.com
S=$1; GPU=$2
mkdir -p $B/results/logs
CUDA_VISIBLE_DEVICES=$GPU $E/bin/boltz predict $B/boltz_fep/shard_$S \
  --out_dir $B/boltz_fep_out/shard_$S --cache $B/boltz_cache \
  --accelerator gpu --devices 1 --use_msa_server --no_kernels \
  --diffusion_samples 1 --output_format pdb --num_workers 2 \
  > $B/results/logs/boltz_fep_$S.log 2>&1
echo "shard_$S exit=$?" >> $B/results/logs/boltz_fep_done.log
