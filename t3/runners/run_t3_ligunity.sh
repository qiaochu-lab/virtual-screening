#!/bin/bash
# LigUnity 在 T3 上跑。$1=变体(pocket_ranking|protein_ranking)  $2=GPU
# 参数照搬官方 test.sh（--task test_task / --loss rank_softmax / ./test_datasets），
# 只把 --test-task 换成 T3、加 --t3-root、并把 batch 从 256 降到 8
# （T3 分子最大 336 个原子，DEKOIS 才 50，UniMol 注意力 O(n^2) 会 OOM）。
set -u
export PATH=/data/yicheng/xqc/envs/ligunity/bin:$PATH
B=/data/yicheng/xqc/vs-benchmark
V=$1; GPU=$2
C=$B/ckpt/ligunity/LigUnity_VS/${V}_vs/checkpoint_avg_41-50.pt
OUT=$B/results/t3_raw/ligunity_${V}
mkdir -p $OUT $B/results/logs

# 幂等保护：这个变体如果已经跑完（四层都有结果），直接跳过。
# 需要它是因为我们把 LigUnity 提前拉到空闲卡上跑了，
# 而队列里第 7 步还会再调一次——没这个保护会白跑几小时。
DONE=$(find $OUT/T3 -name saved_preds.npy 2>/dev/null | wc -l)
NLAY=$(ls $OUT/T3 2>/dev/null | wc -l)
if [ "$DONE" -gt 900 ] && [ "$NLAY" -ge 4 ]; then
  echo "ligunity_$V 已完成（$DONE 个靶点 / $NLAY 层），跳过"
  exit 0
fi
if [ "$DONE" -gt 0 ]; then
  echo "ligunity_$V 上次只跑到 $DONE 个靶点，重跑"
fi
cd $B/code/LigUnity
CUDA_VISIBLE_DEVICES=$GPU python ./unimol/test.py "./test_datasets" --user-dir ./unimol \
  --valid-subset test --results-path "$B/results/t3_raw/ligunity_${V}" \
  --num-workers 4 --ddp-backend=c10d --batch-size 8 \
  --task test_task --loss rank_softmax --arch ${V} \
  --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --seed 1 \
  --path "$C" --log-interval 100 --log-format simple \
  --max-pocket-atoms 511 --test-task T3 --t3-root "$B/data/T3_6A" \
  > "$B/results/logs/ligunity_${V}_T3.log" 2>&1
echo "ligunity_$V 退出码=$?"
