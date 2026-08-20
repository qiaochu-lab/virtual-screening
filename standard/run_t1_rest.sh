#!/bin/bash
# 补齐 T1：LiTENCLIP 和 HypSeek 在三个标准基准上跑，顺带把 CASF 补上（T2 的第三套数据）。
#
# 为什么是这两个模型
# ------------------
# T1 现在只有 DrugCLIP / BindCLIP×2 / LigUnity×2，五个。LiTENCLIP 和 HypSeek
# 与 LigUnity 同源（同一份 test_datasets、同一套 test.py 分支），加个 --test-task
# 就能跑，不需要任何数据准备——先把便宜的补掉。
# ConGLUDe / ConPLex / SPRINT 要给 DUD-E/DEKOIS/LIT-PCBA 的靶点另外准备序列或结构，
# 是另一件事，不在这个脚本里。
#
# 顺带跑 CASF：test.py 本来就有这个分支（inference_pdbbind），
# 而 CASF-2016 是 T2 一直缺的第三套数据（打分函数领域的标准集）。
#
# 注：LiTENCLIP 的 test_datasets 里缺 DUD-E / casf.lmdb（只有 DEKOIS 和 lit_pcba），
# 已把 LigUnity 的同名目录软链过去——两边本来就是同一份数据。
#
# 单卡串行跑 GPU 5——0-3 是别人的，4/6/7 在跑 Boltz-2 FEP，
# 总占用保持在 4 张卡以内。
set -u
B=/data/work/vs-benchmark
E=/data/work/envs/litenclip
GPU=${1:-5}
L=$B/results/logs
mkdir -p $L

run_one() {   # $1=模型 $2=任务
  local M=$1 T=$2 OUT=$B/results/$1
  # 幂等：这个 (模型,任务) 已经有结果就跳过，方便中断后重跑
  if [ -d "$OUT/$T" ] && [ "$(ls $OUT/$T 2>/dev/null | wc -l)" -gt 5 ]; then
    echo "[$(date +%H:%M)] $M/$T 已存在，跳过"; return
  fi
  echo "[$(date +%H:%M)] 开始 $M/$T"
  case $M in
    litenclip)
      cd $B/code/LiTENCLIP
      LD_LIBRARY_PATH=$E/lib CUDA_VISIBLE_DEVICES=$GPU $E/bin/python ./unimol/test.py \
        ./test_datasets --user-dir ./unimol --valid-subset test \
        --results-path $OUT --num-workers 2 --ddp-backend c10d --batch-size 128 \
        --task test_task --loss rank_softmax --arch liten_clip --bf16 --seed 1 \
        --path $B/ckpt/litenclip/checkpoint.best_valid_bedroc_0.50.pt \
        --log-interval 100 --log-format simple \
        --max-pocket-atoms 511 --test-task $T \
        > $L/litenclip_$T.log 2>&1 ;;
    hypseek_rk)
      cd $B/code/HypSeek
      export PYTHONPATH="$B/code/HypSeek/unimol:${PYTHONPATH:-}"
      export HF_ENDPOINT=https://hf-mirror.com HF_HOME=$B/hf_cache
      LD_LIBRARY_PATH=$E/lib CUDA_VISIBLE_DEVICES=$GPU $E/bin/python ./unimol/test.py \
        "$B/code/LigUnity/test_datasets" --user-dir ./unimol --valid-subset test \
        --results-path $OUT --num-workers 2 --ddp-backend c10d \
        --distributed-world-size 1 --batch-size 128 \
        --task test_task --loss three_hybrid_loss --arch three_hybrid_model \
        --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --seed 1 \
        --path $B/ckpt/hypseek/checkpoint_avg_41-50_rk.pt \
        --log-interval 100 --log-format simple \
        --max-pocket-atoms 511 --test-task $T \
        > $L/hypseek_$T.log 2>&1 ;;
  esac
  # 必须先存下来：$(date ...) 会覆盖 $?
  local rc=$?
  echo "[$(date +%H:%M)] $M/$T 退出码=$rc"
  [ $rc -ne 0 ] && tail -3 $L/${M%_rk}_$T.log
}

for M in litenclip hypseek_rk; do
  for T in DUDE DEKOIS PCBA CASF; do
    run_one $M $T
  done
done
echo "[$(date +%H:%M)] 全部结束"
