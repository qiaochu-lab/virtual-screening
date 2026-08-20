#!/bin/bash
# 等显卡空出来再启动——现在 8 张全被别人的 vLLM 占着（每个任务 4 张，两个任务）。
#
# 判定「空」= 显存占用 < 1000 MiB，且连续两次检查（间隔 5 分钟）都空，
# 避免别人任务重启的空档里把卡抢走。
# 最多只拿 3 张：2 张续跑 Boltz-2 亲和力，1 张跑 T1 队列。
# （红线是 4 张，留一张余量。）
set -u
B=/data/work/vs-benchmark
LOG=$B/results/logs/wait_and_run.log
free_gpus() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F', ' '$2 < 1000 {print $1}'
}
prev=""
while true; do
  now=$(free_gpus | tr '\n' ' ')
  echo "[$(date +%m-%d_%H:%M)] 空闲卡: [${now}]" >> $LOG
  # 取两次都空闲的
  stable=""
  for g in $now; do case " $prev " in *" $g "*) stable="$stable $g";; esac; done
  cnt=$(echo $stable | wc -w)
  if [ "$cnt" -ge 1 ]; then
    set -- $stable
    echo "[$(date +%m-%d_%H:%M)] 稳定空闲 $cnt 张: $stable —— 启动" >> $LOG
    i=0
    for g in $stable; do
      i=$((i+1))
      case $i in
        1) tmux new-session -d -s bf0r "bash $B/resume_boltz_fep.sh 0 $g" ;;
        2) tmux new-session -d -s bf1r "bash $B/resume_boltz_fep.sh 1 $g" ;;
        3) tmux new-session -d -s t1rest "bash $B/run_t1_rest.sh $g 2>&1 | tee $B/results/logs/t1_rest.log" ;;
        *) break ;;
      esac
      sleep 5
    done
    echo "[$(date +%m-%d_%H:%M)] 已启动 $i 个任务，退出等待循环" >> $LOG
    exit 0
  fi
  prev="$now"
  sleep 300
done
