#!/bin/bash
# 按指定口袋阈值跑三个 UniMol 模型。$1 = 阈值整数（4 / 6 / 8）
# 只用 GPU 5、6：三个任务里两个共用 6 卡（单任务实测 4-11 GB，24 GB 够）。
set -u
export PATH=/data/work/envs/ligunity/bin:$PATH
B=/data/work/vs-benchmark
T=$1
ROOT=$B/data/T3_${T}A
L=$B/results/logs
mkdir -p $L
if [ ! -d "$ROOT" ]; then echo "数据目录 $ROOT 不存在，跳过"; exit 1; fi
run() {  # $1=仓库 $2=权重 $3=结果目录 $4=GPU $5=标签
  cd $B/code/$1
  nohup env CUDA_VISIBLE_DEVICES=$4 python ./unimol/test.py --user-dir ./unimol "./data" \
    --valid-subset test --results-path "$3" \
    --num-workers 4 --ddp-backend=c10d --batch-size 8 \
    --task drugclip --loss in_batch_softmax --arch drugclip \
    --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --seed 1 \
    --path "$2" --log-interval 100 --log-format simple \
    --max-pocket-atoms 511 --test-task T3 --t3-root "$ROOT" \
    > "$L/$5_T3_${T}a.log" 2>&1 &
  echo "  $5  GPU$4  PID=$!"
}
R=$B/results/t3_raw_${T}a
run DrugCLIP $B/ckpt/drugclip/checkpoint_best.pt    $R/drugclip          5 drugclip
sleep 5
run BindCLIP $B/ckpt/bindclip/BindCLIP_randneg.pt   $R/bindclip_randneg  6 bindclip_randneg
sleep 5
run BindCLIP $B/ckpt/bindclip/BindCLIP_hardneg.pt   $R/bindclip_hardneg  6 bindclip_hardneg
disown -a
echo "阈值 ${T}Å 三个任务已启动"
