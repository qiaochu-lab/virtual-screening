#!/bin/bash
# 三条链，按顺序等：对接汇总 → Boltz 重排汇总 → NEW-8 蛋白置空。
#
# ⚠️ 不用进程名匹配来判「跑完了没」。
# 今天这个坑出现了三次：grep 的模式匹配到了创建脚本的那个 bash -c 自己的命令行，
# 或者模式和实际进程名对不上（run_swap_fam.sh vs run_swap_fam_pocket.sh）。
# 改成**等输出文件数达标**——这个判据不会匹配到自己，也不依赖进程名。
set -u
B=/data/work/vs-benchmark
P=/data/work/envs/ligunity/bin/python
LOG=$B/results/logs/chain_after_gpu.log
say(){ echo "[$(date '+%m-%d_%H:%M')] $*" >> "$LOG"; }

# ---------- 1. 对接（T6-RE）----------
say "等对接跑完（21 个靶点各 ≥180/200 个打分）"
while :; do
  n=0
  for f in "$B"/dock/*/scores.txt; do
    [ -e "$f" ] || continue
    c=$(grep -cE '^ *1 +-?[0-9]' "$f" 2>/dev/null || echo 0)
    [ "$c" -ge 180 ] && n=$((n+1))
  done
  [ "$n" -ge 21 ] && break
  # 也可能有靶点永远到不了 180（超时/异常），所以设一个总时限兜底
  [ "$SECONDS" -gt 36000 ] && { say "对接超过 10 小时未全完，按现有结果汇总"; break; }
  sleep 300
done
say "对接汇总"
$P "$B/score_dock.py" > "$B/results/export/T6_dock_subset.txt" 2>&1
say "  exit=$? → results/export/T6_dock_subset.txt"

# ---------- 2. Boltz-2 重排 ----------
say "等 Boltz-2 重排跑完（3,747 个 affinity json，或四个 shard 都写完结束标记）"
while :; do
  n=$(find "$B/boltz_rerank_sub_out" -name 'affinity_*.json' 2>/dev/null | wc -l)
  grep -q '四个 shard 全部结束' "$B/results/logs/boltz_rerank_sub.log" 2>/dev/null && break
  [ "$n" -ge 3700 ] && break
  sleep 600
done
say "Boltz-2 汇总（出分 $(find "$B/boltz_rerank_sub_out" -name 'affinity_*.json' 2>/dev/null | wc -l)）"
$P "$B/export_rerank_sub.py" > "$B/results/export/T6_rerank_subset.txt" 2>&1
say "  exit=$? → results/export/T6_rerank_subset.txt"

say "✅ 两条汇总完成。NEW-8 需要人工确认口径后再起，不自动启动。"
