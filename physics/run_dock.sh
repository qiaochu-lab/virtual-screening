#!/bin/bash
# smina 对接 T6 的深 shortlist（20 靶点 × top-200）。纯 CPU，不碰 GPU。
#
# 深度选 top-200 而不是 Boltz-2 那轮的 50
# --------------------------------------
# recall@50 在 L4 只有 17.5%，重排的天花板太低；recall@200 是 34.0%，翻一倍。
# 对接每个分子几秒，撑得住这个深度——这正是引入它的理由。
#
# 配体用的是建 T3 输入时生成的同一份 3D 构象（smina 读不了 SMILES），
# 所以对接与检索模型吃的是同一个构象，比较时少一个混杂变量。
set -u
B=/data/yicheng/xqc/vs-benchmark
D=/data/yicheng/xqc/envs/dock/bin
LOG=$B/results/logs/dock.log
NPROC=${1:-12}

for up in $(python3 -c "import json;print(' '.join(json.load(open('$B/dock/manifest.json'))['targets']))"); do
  d=$B/dock/$up
  [ -s "$d/scores.txt" ] && continue
  [ -s "$d/ligands.sdf" ] || { echo "[$(date +%H:%M)] $up 无配体，跳过" >> $LOG; continue; }
  read cx cy cz sx sy sz <<< $(python3 -c "
import json; m=json.load(open('$d/manifest.json'))
print(*[f'{v:.3f}' for v in m['center']+m['size']])")
  echo "[$(date +%H:%M)] $up 开始" >> $LOG
  # 单靶点上限 90 分钟：盒子体积差一倍耗时就差很多，
  # O00411 的盒子 36,978 Å³（正常 19-28k）跑了近 7 小时把队列堵死。
  # 超时就跳过并记下来，不让一个异常靶点拖垮整批。
  timeout 5400 $D/smina -r $d/pocket.pdbqt -l $d/ligands.sdf \
      --center_x $cx --center_y $cy --center_z $cz \
      --size_x $sx --size_y $sy --size_z $sz \
      --exhaustiveness 8 --num_modes 1 --cpu $NPROC --seed 1 \
      -o $d/poses.sdf > $d/scores.txt 2>$d/smina.err
  rc=$?; [ $rc = 124 ] && echo "[$(date +%H:%M)] $up 超时跳过（盒子过大）" >> $LOG
  echo "[$(date +%H:%M)] $up exit=$rc 打分行数 $(grep -cE '^ *1 +-?[0-9]' $d/scores.txt 2>/dev/null)" >> $LOG
done
echo "[$(date +%H:%M)] 对接全部结束" >> $LOG
