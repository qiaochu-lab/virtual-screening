#!/bin/bash
# smina 对接 T6 的深 shortlist，并行版。纯 CPU，不碰 GPU。
#
# 为什么从串行改成并行
# --------------------
# 旧版 run_dock.sh 一次跑一个靶点、给它 16 个核，单靶点 90 分钟上限，
# 超时就跳过。注释里把超时归因于「盒子过大」——**这个归因是错的**。
# 拿第一轮 20 个靶点的实测对一下（dock_v1_full_L4）：
#     O75143   6,051 Å³  只打了   4/200
#     O75106  42,488 Å³  打完了 195/200
#     O14874  34,459 Å³  打完了 200/200
# 体积和完成度没有关系。决定耗时的更可能是配体的可旋转键数，不是盒子。
# 所以按体积筛靶点会「筛掉能跑的、留下跑不动的」，不能那么做。
#
# 正确的修法是并行：慢靶点不再堵住整个队列，每个靶点拿到的墙钟时间
# 从「排队等前面所有人」变成「自己跑多久算多久」。
# 21 个靶点 × 4 核 = 84 线程，机器有 104 线程，留出余量给别的任务。
#
# 幂等：已有完整 scores.txt 的靶点直接跳过，中途挂掉重跑不白干。
set -u
B=/data/work/vs
D=/data/work/envs/dock/bin
LOG=$B/results/logs/dock.log
PER=${1:-4}          # 每个靶点几个核
PAR=${2:-16}         # 同时跑几个靶点
TIMEOUT=${3:-21600}  # 单靶点上限，默认 6 小时

say(){ echo "[$(date '+%m-%d_%H:%M')] $*" >> "$LOG"; }

dock_one(){
  local up=$1 d=$B/dock/$1
  # 完整才算数：打分行数 ≥ 配体数的 90%。旧版只看文件存在，
  # 一个只打了 4/200 的 scores.txt 也会被当成已完成而跳过。
  local want got
  want=$(python3 -c "import json;print(len(json.load(open('$d/manifest.json'))['ligands']))")
  if [ -s "$d/scores.txt" ]; then
    got=$(grep -cE '^ *1 +-?[0-9]' "$d/scores.txt" 2>/dev/null || echo 0)
    [ "$got" -ge "$((want * 9 / 10))" ] && { say "$up 已完成 $got/$want，跳过"; return; }
    say "$up 上次只打到 $got/$want，重跑"
  fi
  [ -s "$d/ligands.sdf" ] || { say "$up 无配体，跳过"; return; }
  read cx cy cz sx sy sz <<< $(python3 -c "
import json; m=json.load(open('$d/manifest.json'))
print(*[f'{v:.3f}' for v in m['center']+m['size']])")
  say "$up 开始（$want 个配体，$PER 核）"
  timeout "$TIMEOUT" $D/smina -r "$d/pocket.pdbqt" -l "$d/ligands.sdf" \
      --center_x $cx --center_y $cy --center_z $cz \
      --size_x $sx --size_y $sy --size_z $sz \
      --exhaustiveness 8 --num_modes 1 --cpu "$PER" --seed 1 \
      -o "$d/poses.sdf" > "$d/scores.txt" 2>"$d/smina.err"
  local rc=$?
  got=$(grep -cE '^ *1 +-?[0-9]' "$d/scores.txt" 2>/dev/null || echo 0)
  [ $rc = 124 ] && say "$up 超时（${TIMEOUT}s），保留已打的 $got/$want"
  say "$up exit=$rc 打分 $got/$want"
}
export -f dock_one
export B D LOG PER TIMEOUT

TARGETS=$(python3 -c "import json;print(' '.join(json.load(open('$B/dock/manifest.json'))['targets']))")
say "并行对接启动：$(echo $TARGETS | wc -w) 个靶点，$PAR 路并行 × $PER 核"
printf '%s\n' $TARGETS | xargs -P "$PAR" -I{} bash -c 'dock_one "$@"' _ {}
say "对接全部结束"
