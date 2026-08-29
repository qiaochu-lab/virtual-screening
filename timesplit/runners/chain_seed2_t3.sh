#!/bin/bash
# 等 PCBA 补跑结束 → 跑 seed=2 的 T3（脚本已参数化，不再覆盖 seed1）→ 算分
set -u
B=/data/yicheng/xqc/vs-benchmark
PY=/data/yicheng/xqc/envs/ligunity/bin/python
LOG=$B/results/logs/chain_seed2_t3.log
say(){ echo "[$(date '+%m-%d_%H:%M')] $*" >> $LOG; }

say "等 PCBA 补跑结束"
while ! grep -q '补跑结束' $B/results/logs/rerun_pcba.log 2>/dev/null; do sleep 120; done
say "PCBA 补跑完毕: $(grep 'PCBA exit' $B/results/logs/rerun_pcba.log | tr '\n' ' ')"

say "起 seed=2 T3"
bash $B/run_t3_hypseek_vs.sh 4 2
say "seed=2 T3 退出"
for L in L1 L2 L3 L4; do
  say "  $L 靶点数 $(ls $B/results/t3_raw/hypseek_vs_s2/T3/$L 2>/dev/null | wc -l)"
done

cd $B && $PY collect_t3.py >> $LOG 2>&1
$PY score_t3.py --models hypseek_rk hypseek_vs_s1 hypseek_vs_s2 --layers L1 L2 L3 L4 >> $LOG 2>&1
cd $B/eval && for s in 1 2; do
  echo "===== seed=$s" >> $LOG
  $PY score_ligunity.py $B/results/hypseek_vs_s$s >> $LOG 2>&1
done
say "全部完成"
