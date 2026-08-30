#!/bin/bash
# T6 rerank 第四版：唯一的改动是 --diffusion_samples 1 -> 5。
#
# 为什么要跑：Boltz 的结构阶段采样 N 个复合物，按 confidence 排名，
# 只有 rank-0 会被写成 pre_affinity_*.npz 交给亲和力模型
# （见 boltz/data/write/writer.py:177）。N=1 时没有任何筛选——
# 我们等于把一次未经挑选的随机抽样直接喂给了打分模型。
# N=5 时至少有个 best-of-5 的置信度筛选。
#
# 输入直接复用 rerank3 的 yaml，所以这是逐复合物配对的对照：
# 同样的靶点、同样的候选、同样的 MSA，只有采样数不同。
#
# 注意：亲和力阶段的 --diffusion_samples_affinity 一直是默认的 5，
# 两轮都一样，不是变量。
#
# $1=shard $2=GPU
set -u
E=/data/work/envs/boltz2
B=/data/work/vs
S=$1; GPU=$2
mkdir -p $B/results/logs
CUDA_VISIBLE_DEVICES=$GPU $E/bin/boltz predict $B/boltz_rerank3/shard_$S \
  --out_dir $B/boltz_rerank4_out/shard_$S --cache $B/boltz_cache \
  --accelerator gpu --devices 1 --no_kernels \
  --diffusion_samples 5 --output_format pdb --num_workers 2 \
  > $B/results/logs/boltz_rerank4_$S.log 2>&1
rc=$?
n=$(find $B/boltz_rerank4_out/shard_$S -name 'affinity_*.json' 2>/dev/null | wc -l)
echo "[$(date '+%m-%d_%H:%M')] rerank4 shard_$S exit=$rc 出分 $n" >> $B/results/logs/boltz_rerank_done.log
