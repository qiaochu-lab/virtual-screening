#!/bin/bash
# Re-run the three models whose T1 score files are missing, after
# patch_t1_save_preds.py. Writes to results/t1preds_check/<model>/ -- NEVER into
# the canonical results/<model>/, so the existing verified embeddings and the
# published table cannot be damaged by a bad run. Files are copied across only
# after verify_t1_preds.py passes.
#
# One worker per model, three GPUs total (the 4-GPU ceiling is a hard rule here).
# Each worker runs DUDE -> DEKOIS -> PCBA serially. PCBA is last on purpose: it
# is the one with real risk (3.8 GB, up to 360k molecules on a single target).
set -u
B=/data/work/vs-benchmark
L=$B/results/logs
mkdir -p $L

run_lit() {   # $1=benchmark $2=gpu
  local T=$1 G=$2 E=/data/work/envs/litenclip
  local OUT=$B/results/t1preds_check/litenclip
  if [ "$(find $OUT/$T -name saved_preds.npy 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "[$(date +%H:%M)] litenclip/$T 已有结果，跳过"; return
  fi
  mkdir -p $OUT
  echo "[$(date +%H:%M)] 开始 litenclip/$T (GPU $G)"
  cd $B/code/LiTENCLIP
  LD_LIBRARY_PATH=$E/lib CUDA_VISIBLE_DEVICES=$G $E/bin/python ./unimol/test.py \
    ./test_datasets --user-dir ./unimol --valid-subset test \
    --results-path $OUT --num-workers 2 --ddp-backend c10d --batch-size 128 \
    --task test_task --loss rank_softmax --arch liten_clip --bf16 --seed 1 \
    --path $B/ckpt/litenclip/checkpoint.best_valid_bedroc_0.50.pt \
    --log-interval 100 --log-format simple \
    --max-pocket-atoms 511 --test-task $T \
    > $L/t1preds_litenclip_$T.log 2>&1
  echo "[$(date +%H:%M)] litenclip/$T 退出码=$?  preds=$(find $OUT/$T -name saved_preds.npy 2>/dev/null | wc -l)"
}

run_lig() {   # $1=variant $2=benchmark $3=gpu
  local V=$1 T=$2 G=$3 E=/data/work/envs/ligunity
  local OUT=$B/results/t1preds_check/$V
  if [ "$(find $OUT/$T -name saved_preds.npy 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "[$(date +%H:%M)] $V/$T 已有结果，跳过"; return
  fi
  mkdir -p $OUT
  echo "[$(date +%H:%M)] 开始 $V/$T (GPU $G)"
  cd $B/code/LigUnity
  PATH=$E/bin:$PATH CUDA_VISIBLE_DEVICES=$G $E/bin/python ./unimol/test.py \
    ./test_datasets --user-dir ./unimol --valid-subset test \
    --results-path $OUT --num-workers 2 --ddp-backend=c10d --batch-size 128 \
    --task test_task --loss rank_softmax --arch $V \
    --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --seed 1 \
    --path $B/ckpt/ligunity/LigUnity_VS/${V}_vs/checkpoint_avg_41-50.pt \
    --log-interval 100 --log-format simple \
    --max-pocket-atoms 511 --test-task $T \
    > $L/t1preds_${V}_$T.log 2>&1
  echo "[$(date +%H:%M)] $V/$T 退出码=$?  preds=$(find $OUT/$T -name saved_preds.npy 2>/dev/null | wc -l)"
}

( for T in DUDE DEKOIS PCBA; do run_lit $T 4; done ) &
( for T in DUDE DEKOIS PCBA; do run_lig pocket_ranking  $T 5; done ) &
( for T in DUDE DEKOIS PCBA; do run_lig protein_ranking $T 6; done ) &
wait
echo "[$(date +%H:%M)] 全部结束"
find $B/results/t1preds_check -name saved_preds.npy | sed "s|.*t1preds_check/||" | cut -d/ -f1,2 | sort | uniq -c
