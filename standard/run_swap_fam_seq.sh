#!/bin/bash
# Target swap，序列类模型（ConPLex / ConGLUDe / SPRINT）。
#
# 这三个不读口袋，从评测集 jsonl 的 uniprot 去查序列/结构，
# 所以它们的 swap 评测集由 build_swap_eval.py 生成：记录的 uniprot 换成替身、
# 配体池保留原靶点。与口袋类模型共用同一份 swap_manifest.json，
# 配对关系一致，结果可以放在同一张表里比。
#
# 参数照抄各自的 run_t3_*.py 默认值，只改 --eval_dir 和 --out_dir。
#
# 用法: ./run_swap_seq.sh <模型> <GPU> <round>
set -u
M=${1:?模型}; GPU=${2:-4}; RND=${3:-1}
B=/data/work/vs-benchmark
EV=$B/data/T3_swap_eval_family/round${RND}
OUT=$B/results/t3_raw/swapfam_${M}_r${RND}
LOG=$B/results/logs/swapfam_${M}_r${RND}.log
mkdir -p "$OUT" "$B/results/logs"
[ -d "$EV" ] || { echo "swap 评测集不存在: $EV"; exit 1; }
echo "[$(date '+%m-%d_%H:%M')] $M round$RND GPU$GPU 起" >> "$LOG"

case "$M" in
  conplex)
    /data/work/envs/conplex/bin/python run_t3_conplex.py \
      --layers L1 L4 --eval_dir "$EV" --out_dir "$OUT" >> "$LOG" 2>&1 ;;
  conglude)
    /data/work/envs/conglude/bin/python run_t3_conglude.py \
      --layers L1 L4 --eval_dir "$EV" --out_dir "$OUT" --gpu "$GPU" \
      --tag swap_r${RND} >> "$LOG" 2>&1 ;;
  sprint)
    /data/work/envs/sprint/bin/python run_t3_sprint.py \
      --layers L1 L4 --eval_dir "$EV" --out_dir "$OUT" --gpu "$GPU" \
      --work "$B/tmp/sprint_swap_r${RND}" >> "$LOG" 2>&1 ;;
  *) echo "未配置: $M" >> "$LOG"; exit 1 ;;
esac
echo "[$(date '+%m-%d_%H:%M')] $M round$RND exit=$?" >> "$LOG"
for L in L1 L4; do
  echo "  $L: $(find "$OUT" -path "*${L}*" -name 'saved_preds.npy' 2>/dev/null | wc -l)" >> "$LOG"
done
