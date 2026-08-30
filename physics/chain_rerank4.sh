#!/bin/bash
# 起 rerank4 的三个分片（3 张卡，在 4 张上限内），跑完自动算分。
# 断网不影响：全程 setsid nohup。
set -u
B=/data/work/vs
PY=/data/work/envs/ligunity/bin/python
LOG=$B/results/logs/chain_rerank4.log
say(){ echo "[$(date '+%m-%d_%H:%M')] $*" >> $LOG; }

say "起 rerank4：diffusion_samples 1->5，复用 rerank3 的 750 个复合物"
for S in 0 1 2; do
  GPU=$((S+5))          # 用 5/6/7，避开 0-3
  setsid nohup bash $B/run_rerank4.sh $S $GPU > /dev/null 2>&1 < /dev/null &
  say "  shard_$S -> GPU $GPU"
  sleep 5
done

# 等三个分片都写完成记录（等结果文件，不等进程名——僵尸进程会把 pgrep 卡住）
say "等三个分片结束"
while [ "$(grep -c 'rerank4 shard_' $B/results/logs/boltz_rerank_done.log 2>/dev/null)" -lt 3 ]; do
  sleep 300
done
say "三个分片都结束: $(grep 'rerank4 shard_' $B/results/logs/boltz_rerank_done.log | tr '\n' ' ')"

for S in 0 1 2; do
  say "  shard_$S 出分 $(find $B/boltz_rerank4_out/shard_$S -name 'affinity_*.json' 2>/dev/null | wc -l) / $(ls $B/boltz_rerank3/shard_$S/*.yaml | wc -l)"
done

say "算分"
cd $B && $PY export_rerank4.py >> $LOG 2>&1
say "完成"
