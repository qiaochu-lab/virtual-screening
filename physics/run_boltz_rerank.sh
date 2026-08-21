#!/bin/bash
# 串联 rerank：对检索模型 top-50 的 997 个复合物逐个跑 Boltz-2 亲和力。
# $1=shard $2=GPU。MSA 已在 yaml 里指好（复用 T3 结构阶段生成的），不发服务器请求。
set -u
E=/data/work/envs/boltz2
B=/data/work/vs-benchmark
S=$1; GPU=$2
mkdir -p $B/results/logs
CUDA_VISIBLE_DEVICES=$GPU $E/bin/boltz predict $B/boltz_rerank/shard_$S \
  --out_dir $B/boltz_rerank_out/shard_$S --cache $B/boltz_cache \
  --accelerator gpu --devices 1 --no_kernels \
  --diffusion_samples 1 --output_format pdb --num_workers 2 \
  > $B/results/logs/boltz_rerank_$S.log 2>&1
echo "rerank shard_$S exit=$?" >> $B/results/logs/boltz_rerank_done.log
