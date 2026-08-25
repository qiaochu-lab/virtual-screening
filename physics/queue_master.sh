#!/bin/bash
# 总调度：把所有待跑的活按优先级自动排上，卡一空就起，跑完接下一个。
#
# 为什么要一个总调度而不是各挂各的等待器
# --------------------------------------
# 之前 wait_sprint / wait_rerank 各等各的，谁先抢到卡不确定，还可能一起抢
# 超过 4 张的红线。这里集中判断：只在 4-7 号卡里找空闲（0-3 是别人的），
# 且我方同时占用不超过 MAXGPU 张。
#
# 优先级（先补完整性，再做新实验）
#   1. SPRINT × DUD-E        —— T1 九模型只差这两格
#   2. SPRINT × LIT-PCBA
#   3. SPRINT × T3 L1/L2 重试 —— 原以为无解，实为 ulimit -n=1024，提高后 DEKOIS 已通
#   4. L1/L2 串联 rerank      —— T6 的 P0：判定「思路不行」还是「只在新靶点失效」
#                                依赖 MSA 预热完成（msa_l1l2 目录里 ≥25 个 csv）
set -u
B=/data/work/vs-benchmark
PY=/data/work/envs/ligunity/bin/python
LOG=$B/results/logs/queue_master.log
MAXGPU=4
say() { echo "[$(date +%m-%d_%H:%M)] $*" >> $LOG; }

free_gpus() {   # 只看 4-7
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F', ' '$1 >= 4 && $2 < 1000 {print $1}'
}
my_gpu_count() {  # 我方当前占了几张卡
  local n=0
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    if ps -o user= -p "$p" 2>/dev/null | grep -q "^$USER"; then n=$((n+1)); fi
  done
  echo $n
}

done_sprint_dude() { [ "$(ls $B/results/t1_raw/sprint/DUDE 2>/dev/null | wc -l)" -gt 50 ]; }
done_sprint_pcba() { [ "$(ls $B/results/t1_raw/sprint/PCBA 2>/dev/null | wc -l)" -gt 5 ]; }
done_sprint_t3()   { [ "$(ls $B/results/t3_raw/sprint/T3/L1 2>/dev/null | wc -l)" -gt 50 ]; }
done_rerank2()     { [ "$(find $B/boltz_rerank3_out -name 'affinity_*.json' 2>/dev/null | wc -l)" -gt 100 ]; }
msa_ready()        { [ "$(ls $B/data/t3/msa_l1l2/*.csv 2>/dev/null | wc -l)" -ge 25 ]; }

running() { pgrep -u "$USER" -f "$1" > /dev/null; }

declare -A TRIES
try_ok() {   # $1=作业名；连试 3 次仍无产出就放弃，避免像上次那样重启上千遍
  TRIES[$1]=$(( ${TRIES[$1]:-0} + 1 ))
  if [ ${TRIES[$1]} -gt 3 ]; then
    return 1
  fi
  return 0
}
# 注意：这个脚本必须用 setsid nohup 起，不要放 tmux。
# tmux 的 socket 在 /tmp/tmux-1001，这台机器的 /tmp 清理程序会把它删掉，
# tmux server 一死，会话里的调度就没了（已经踩过两次；
# 同一个清理程序还啃掉过 scratchpad 里 git 仓库的 .git/HEAD）。
say "总调度启动，红线 ${MAXGPU} 张卡，只用 4-7"

for round in $(seq 1 2000); do
  mapfile -t FREE < <(free_gpus)
  USED=$(my_gpu_count)
  BUDGET=$(( MAXGPU - USED ))
  [ "$BUDGET" -lt 0 ] && BUDGET=0
  AVAIL=${#FREE[@]}
  [ "$AVAIL" -gt "$BUDGET" ] && AVAIL=$BUDGET

  # ---- 1/2/3：SPRINT 三件，各 1 张卡 ----
  if [ "$AVAIL" -ge 1 ] && ! done_sprint_dude && ! running "layers DUDE" && try_ok sprint_dude; then
    g=${FREE[0]}
    say "SPRINT × DUD-E -> GPU $g"
    setsid nohup sh -c "ulimit -n 65535; cd $B && exec $PY run_t3_sprint.py --layers DUDE \
      --eval_dir $B/data/t1 --seqs $B/data/t1/saprot_seqs.json \
      --out_dir $B/results/t1_raw/sprint --work $B/tmp/sprint_t1 --gpu $g" \
      > $B/results/logs/sprint_T1_DUDE.log 2>&1 < /dev/null &
    sleep 90; continue
  fi
  if [ "$AVAIL" -ge 1 ] && done_sprint_dude && ! done_sprint_pcba && ! running "layers PCBA" && try_ok sprint_pcba; then
    g=${FREE[0]}
    say "SPRINT × LIT-PCBA -> GPU $g"
    setsid nohup sh -c "ulimit -n 65535; cd $B && exec $PY run_t3_sprint.py --layers PCBA \
      --eval_dir $B/data/t1 --seqs $B/data/t1/saprot_seqs.json \
      --out_dir $B/results/t1_raw/sprint --work $B/tmp/sprint_t1 --gpu $g" \
      > $B/results/logs/sprint_T1_PCBA.log 2>&1 < /dev/null &
    sleep 90; continue
  fi
  if [ "$AVAIL" -ge 1 ] && ! done_sprint_t3 && ! running "layers L1 L2" && try_ok sprint_t3; then
    g=${FREE[0]}
    say "SPRINT × T3 L1/L2 重试（ulimit 65535）-> GPU $g"
    setsid nohup sh -c "ulimit -n 65535; cd $B && exec $PY run_t3_sprint.py --layers L1 L2 \
      --out_dir $B/results/t3_raw/sprint --work $B/tmp/sprint_t3b --gpu $g" \
      > $B/results/logs/sprint_T3_L1L2.log 2>&1 < /dev/null &
    sleep 90; continue
  fi

  # ---- 4：L1/L2 rerank，最多 3 张卡 ----
  if ! done_rerank2 && msa_ready && ! running "boltz_rerank3"; then
    if [ ! -d "$B/boltz_rerank3" ]; then
      say "生成 L1/L2 rerank 输入（用预热的 MSA）"
      cd $B && $PY prep_rerank.py --targets 30 --min-hits 1 --max-hits 6 \
        --layer L1 --shards 3 --out $B/boltz_rerank3 \
        --msa_dir $B/data/t3/msa_l1l2 --targets_json $B/data/t3/rerank_l1l2_targets.json \
        >> $LOG 2>&1
      cp $B/data/t3/rerank_manifest.json $B/data/t3/rerank3_manifest.json
    fi
    n=$AVAIL; [ "$n" -gt 3 ] && n=3
    if [ "$n" -ge 1 ]; then
      for i in $(seq 0 $((n-1))); do
        g=${FREE[$i]}
        say "rerank3 shard_$i -> GPU $g"
        # 必须显式指定卡：boltz 只认 CUDA_VISIBLE_DEVICES，--devices 1 只是"用几张"，
        # 不指定就全挤到 GPU 0（上一轮就是这么越界的）
        setsid nohup sh -c "cd $B && CUDA_VISIBLE_DEVICES=$g exec /data/work/envs/boltz2/bin/boltz predict \
          $B/boltz_rerank3/shard_$i --out_dir $B/boltz_rerank3_out/shard_$i \
          --cache $B/boltz_cache --accelerator gpu --devices 1 --no_kernels \
          --diffusion_samples 1 --output_format pdb --num_workers 2" \
          > $B/results/logs/boltz_rerank3_$i.log 2>&1 < /dev/null &
        sleep 20
      done
      sleep 120; continue
    fi
  fi

  if done_sprint_dude && done_sprint_pcba && done_sprint_t3 && done_rerank2; then
    say "全部完成，调度退出"
    break
  fi
  sleep 180
done
