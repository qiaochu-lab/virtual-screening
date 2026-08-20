#!/bin/bash
# 续跑 Boltz-2 FEP 的亲和力阶段（shard_0 / shard_1）。
#
# 上一轮的状态
# ------------
# 结构预测：455/461 完成（6 个在结构阶段 OOM 被跳过）
# 亲和力  ：shard_2 全部 153 个完成；shard_0 只到 49/153，shard_1 到 41/149，
#           两个 shard 都是在亲和力阶段崩的：
#             FileNotFoundError: ... pre_affinity_tnks2__015.npz
#           —— 结构阶段被跳过的那几个没有 pre_affinity 文件，
#              亲和力阶段照单全收去读，直接抛异常带死整个进程。
#
# 做法
# ----
# 1. 把「没有 pre_affinity npz」的输入挪到 shard_i_failed/，亲和力阶段就不会碰它们
# 2. 原命令重跑：boltz 自己会跳过已完成的（main.py:391 有这个逻辑），
#    只补剩下的 ~212 个，不会重算结构
#
# 那 6 个结构失败的以后单独用更小的 batch 补，占比 1.3%，不挡主流程。
set -u
E=/data/work/envs/boltz2
B=/data/work/vs-benchmark
export HF_ENDPOINT=https://hf-mirror.com
S=$1; GPU=$2
P=$B/boltz_fep_out/shard_$S/boltz_results_shard_$S/predictions
FAILDIR=$B/boltz_fep/shard_${S}_failed
mkdir -p $FAILDIR $B/results/logs

moved=0
for f in $B/boltz_fep/shard_$S/*.yaml; do
  n=$(basename $f .yaml)
  if [ ! -f "$P/$n/pre_affinity_$n.npz" ]; then
    mv "$f" "$FAILDIR/"; moved=$((moved+1))
  fi
done
echo "shard_$S: 挪走结构缺失的 $moved 个，剩 $(ls $B/boltz_fep/shard_$S | wc -l) 个输入"

CUDA_VISIBLE_DEVICES=$GPU $E/bin/boltz predict $B/boltz_fep/shard_$S \
  --out_dir $B/boltz_fep_out/shard_$S --cache $B/boltz_cache \
  --accelerator gpu --devices 1 --use_msa_server --no_kernels \
  --diffusion_samples 1 --output_format pdb --num_workers 2 \
  >> $B/results/logs/boltz_fep_$S.resume.log 2>&1
echo "shard_$S resume exit=$?" >> $B/results/logs/boltz_fep_done.log
