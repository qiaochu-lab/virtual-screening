#!/bin/bash
# 等 Boltz-2 重排跑完然后汇总。
#
# ⚠️ 判据不用「日志里有没有结束标记」——这一版就是被这个坑掉的：
#    results/logs/boltz_rerank_sub.log 是**追加**模式，里面留着第一次
#    （N=5，被杀掉的那轮）在 09-09 21:51 写下的「四个 shard 全部结束」。
#    第二轮 21:53 才启动，而链在 09-10 09:19 一 grep 就命中了那条陈旧标记，
#    于是拿 0 个分数「汇总」了一遍。
#
#    追加日志里的标记不能当完成信号：它只说明**某一次**跑完了，不说明是这一次。
#
# 改成两个都不依赖外部状态的判据：
#    · 出分数达到 3,400（占 3,747 的 91%，留出解析失败的余量）
#    · 或者：出分数 > 0 且连续 60 分钟没有增长（停滞 = 跑完或卡死，都该汇总）
set -u
B=/data/work/vs-benchmark
P=/data/work/envs/ligunity/bin/python
LOG=$B/results/logs/chain_boltz.log
say(){ echo "[$(date '+%m-%d_%H:%M')] $*" >> "$LOG"; }
count(){ find "$B/boltz_rerank_sub_out" -name 'affinity_*.json' 2>/dev/null | wc -l; }

say "等 Boltz-2 亲和力阶段出分（目标 3,400 / 3,747；或停滞 60 分钟）"
last=-1; stall=0
while :; do
  n=$(count)
  [ "$n" -ge 3400 ] && { say "达标 $n，开始汇总"; break; }
  if [ "$n" -gt 0 ] && [ "$n" -eq "$last" ]; then
    stall=$((stall + 1))
    [ "$stall" -ge 6 ] && { say "出分停在 $n 已 60 分钟，按现有结果汇总"; break; }
  else
    stall=0
  fi
  last=$n
  sleep 600
done

say "汇总（出分 $(count)）"
$P "$B/export_rerank_sub.py" > "$B/results/export/T6_rerank_subset.txt" 2>&1
say "  exit=$? → results/export/T6_rerank_subset.txt"
say "✅ 完成"
