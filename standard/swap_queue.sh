#!/bin/bash
# Target swap 的剩余任务队列，跑在服务器上（setsid nohup），本地断网无影响。
#
# 约束：同时最多 4 张 GPU（用户红线）。所以这里维护一个 4 槽的调度，
# 有空槽才起下一个，而不是一次全放出去。
#
# 幂等：每个任务先看结果目录够不够数（L1 40 + L4 51），够了就跳过。
# 这样中途挂掉重跑不会白干，也能安全地重复调用。
set -u
B=/data/work/vs-benchmark
LOG=$B/results/logs/swap_queue.log
MAXJOBS=4
say(){ echo "[$(date '+%m-%d_%H:%M')] $*" >> "$LOG"; }

POCKET="hypseek_rk ligunity_protein_ranking ligunity_pocket_ranking litenclip drugclip bindclip_randneg bindclip_hardneg"
SEQ="conplex conglude sprint"

done_already(){   # $1=模型 $2=round —— 结果够不够数
  local m=$1 r=$2 d n1 n4
  d=$B/results/t3_raw/swap_${m}_r${r}
  n1=$(find "$d" -path "*L1*" -name saved_preds.npy 2>/dev/null | wc -l)
  n4=$(find "$d" -path "*L4*" -name saved_preds.npy 2>/dev/null | wc -l)
  [ "$n1" -ge 38 ] && [ "$n4" -ge 44 ]
}

# 用 [r] 这个字符类，让 grep 自己的命令行不匹配自己——否则永远多算一个，
# 4 槽实际只能跑 3 个。
running(){ ps -eo args --no-headers | grep -cE '[r]un_swap_(full|seq)\.sh' ; }

free_gpu(){   # 找一张几乎空闲、且不在我们已用列表里的卡
  local used="$1" i u
  for i in 0 1 2 3 4 5 6 7; do
    case " $used " in *" $i "*) continue;; esac
    u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $i 2>/dev/null)
    [ "${u:-100}" -lt 15 ] && { echo "$i"; return; }
  done
  echo ""
}

say "队列启动：round1 补齐 → round2 → round3"
for RND in 1 2 3; do
  for M in $POCKET $SEQ; do
    if done_already "$M" "$RND"; then
      say "跳过 $M round$RND（已完成）"; continue
    fi
    # 等空槽
    while [ "$(running)" -ge "$MAXJOBS" ]; do sleep 60; done
    # 拿到槽位后再查一次：等待期间这个任务可能已经被别的进程跑完了
    # （第一次检查时它正在跑，所以没被跳过）。不复查就会重跑，
    # 更糟的是两个进程同时写同一个结果目录。
    if done_already "$M" "$RND"; then
      say "跳过 $M round$RND（等待期间已完成）"; continue
    fi
    # 收集在用的卡
    USED=$(ps -eo args --no-headers | grep -oE 'run_swap_(full|seq)\.sh [a-z_]+ ([0-9])' | awk '{print $3}' | tr '\n' ' ')
    G=$(free_gpu "$USED")
    if [ -z "$G" ]; then say "没有空闲 GPU，等 3 分钟"; sleep 180; G=$(free_gpu "$USED"); fi
    [ -z "$G" ] && { say "仍无空闲 GPU，跳过 $M round$RND，稍后重试"; continue; }

    case " $SEQ " in
      *" $M "*) SCRIPT=./run_swap_seq.sh ;;
      *)        SCRIPT=./run_swap_full.sh ;;
    esac
    cd "$B" && setsid nohup $SCRIPT "$M" "$G" "$RND" > /dev/null 2>&1 &
    say "起 $M round$RND GPU$G"
    sleep 45   # 错开启动，避免同时抢显存
  done
  say "round$RND 全部已派发"
done

# 等全部结束再汇总
while [ "$(running)" -gt 0 ]; do sleep 120; done
say "全部任务结束，汇总"
/data/work/envs/ligunity/bin/python $B/timesplit/analysis/score_target_swap.py \
  --models $POCKET $SEQ --rounds 3 >> "$LOG" 2>&1
say "✅ 汇总完成，结果在 results/export/T3_target_swap.csv"
