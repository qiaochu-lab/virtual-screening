#!/bin/bash
# 同家族 target swap（NEW-7B），排在 random swap 之后。
#
# random swap 只能证明「换成无关靶点会崩」，区分不了模型认的是具体靶点还是家族。
# 同家族替身序列相近、口袋相似：还崩 → 模型分辨到具体靶点；不怎么掉 → 只认家族。
#
# ⚠️ L4 只有 7 对。不是抽样问题，是结构性的：L4 的定义就是「靶点和家族都没见过」，
# 所以这些靶点的家族在池子里天然稀疏，同家族替身几乎找不到。L1 有 34 对可用。
# 报告时必须把这条写清楚，别把 n=7 当成正常样本量。
set -u
B=/data/work/vs-benchmark
LOG=$B/results/logs/family_queue.log
MODELS="hypseek_rk ligunity_protein_ranking ligunity_pocket_ranking litenclip drugclip bindclip_randneg bindclip_hardneg"
SEQ="conplex conglude sprint"
say(){ echo "[$(date '+%m-%d_%H:%M')] $*" >> "$LOG"; }

say "等 random swap 队列结束"
while ps -eo args --no-headers | grep -qE '[r]un_swap_(full|seq)\.sh|[s]wap_queue\.sh'; do sleep 120; done
say "random swap 已结束，开始同家族 swap"

# 复用 run_swap_*.sh，只把 t3-root 指到 family 树：临时软链切换
ln -sfn "$B/data/T3_swap_family" "$B/data/T3_swap_active" 2>/dev/null

running(){ ps -eo args --no-headers | grep -cE '[r]un_swap_fam'; }

for RND in 1 2 3; do
  for M in $MODELS $SEQ; do
    d=$B/results/t3_raw/swapfam_${M}_r${RND}
    n=$(find "$d" -name saved_preds.npy 2>/dev/null | wc -l)
    [ "$n" -ge 35 ] && { say "跳过 $M round$RND（已有 $n）"; continue; }
    while [ "$(running)" -ge 4 ]; do sleep 60; done
    # ⚠️ 这里的模式必须匹配**实际起来的**进程名。实际跑的是
    # run_swap_fam_pocket.sh / run_swap_fam_seq.sh（run_swap_fam.sh 只是分发器，
    # 起完就退），写成 'run_swap_fam\.sh' 永远匹配不上，USED 恒为空，
    # 于是选卡只剩「利用率 <15%」这一条——新起的任务还没爬上来时会被判成空闲，
    # 两个任务就派到同一张卡上。实测 2026-09-09 那轮 GPU0 和 GPU1 各被派了两个。
    # 没有 OOM（8.3GB+5.6GB / 24GB 放得下），但这是运气，不是设计。
    USED=$(ps -eo args --no-headers \
           | grep -oE 'run_swap_fam(_pocket|_seq)?\.sh [a-z_0-9]+ ([0-9])' \
           | awk '{print $3}' | tr '\n' ' ')
    G=""
    for i in 0 1 2 3; do
      case " $USED " in *" $i "*) continue;; esac
      u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $i 2>/dev/null)
      [ "${u:-100}" -lt 15 ] && { G=$i; break; }
    done
    [ -z "$G" ] && { say "无空闲 GPU，等待"; sleep 180; continue; }
    cd "$B" && setsid nohup ./run_swap_fam.sh "$M" "$G" "$RND" >/dev/null 2>&1 &
    say "起 $M round$RND GPU$G"
    sleep 40
  done
done
while [ "$(running)" -gt 0 ]; do sleep 120; done
say "同家族 swap 全部结束"
