#!/bin/bash
# T6 串联 rerank 第二版：靶点改成「top-50 里只有 1-6 个 active」的稀疏场景。
# 第一版按 active 总数挑靶点，shortlist 里 27% 都是 active，基线被抬得太高，
# 重排没有发挥空间——那是设计缺陷。这版 5.8%，贴近真实虚筛。
# $1=shard $2=GPU
set -u
E=/data/work/envs/boltz2
B=/data/work/vs-benchmark
S=$1; GPU=$2
mkdir -p $B/results/logs
CUDA_VISIBLE_DEVICES=$GPU $E/bin/boltz predict $B/boltz_rerank2/shard_$S \
  --out_dir $B/boltz_rerank2_out/shard_$S --cache $B/boltz_cache \
  --accelerator gpu --devices 1 --no_kernels \
  --diffusion_samples 1 --output_format pdb --num_workers 2 \
  > $B/results/logs/boltz_rerank2_$S.log 2>&1
echo "rerank2 shard_$S exit=$?" >> $B/results/logs/boltz_rerank_done.log
