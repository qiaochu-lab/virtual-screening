#!/bin/bash
# Boltz-2 重排，跑在 350 子集的 L4 靶点上（T6-RE 的 Boltz 那半）。
#
# 和前四轮的两个区别，都是必须的：
#
# 1. **靶点限定在最终的 350 子集。** 前四轮跑在全量 L4 上，12 个靶点里只有 5 个
#    落在子集里，所以那几轮不能直接当主结论。
#
# 2. **--inject-actives：把不在 top-200 里的活性补回候选，召回按构造 100%。**
#    不这么做，重排的上限就是 recall@200 —— 350 子集上 L4 只有 22.6%
#    （top-50 更只有 9.4%）。重排一个近八成活性都不在里面的列表，
#    无论物理方法多准都做不出什么，**结果是预定的**，而且分不清
#    「物理重排没用」和「候选里没东西可捞」。后者已知，前者才是没答过的。
#
# ⚠️ --diffusion_samples 用 1 不用 5。
#    上一轮做过 N=5 vs N=1 的严格配对对照（同靶点同配体同 MSA，只改采样数），
#    结论是**没有差别**：P@5 0.317 vs 0.333、AUROC 0.718 vs 0.720。
#    而 N=5 慢 5 倍——实测每个复合物 3.5 分钟，3,747 个要跑 54 小时。
#    机制上也该没差别：结构阶段只把 rank-0 那个样本交给亲和力模型
#    （writer.py:177），N 大只是 best-of-N 的结构筛选，不是多构象重打分。
#
# 4 个 shard × 4 张卡，用户红线是最多 4 张。
set -u
E=/data/work/envs/boltz2
B=/data/work/vs-benchmark
LOG=$B/results/logs
mkdir -p "$LOG"
for S in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$S nohup "$E/bin/boltz" predict "$B/boltz_rerank_sub/shard_$S" \
    --out_dir "$B/boltz_rerank_sub_out/shard_$S" --cache "$B/boltz_cache" \
    --accelerator gpu --devices 1 --no_kernels \
    --diffusion_samples 1 --output_format pdb --num_workers 2 \
    > "$LOG/boltz_rerank_sub_$S.log" 2>&1 &
  echo "[$(date '+%m-%d_%H:%M')] shard_$S 起在 GPU$S (N=1)" >> "$LOG/boltz_rerank_sub.log"
  sleep 20
done
wait
echo "[$(date '+%m-%d_%H:%M')] 四个 shard 全部结束" >> "$LOG/boltz_rerank_sub.log"
