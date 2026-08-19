#!/bin/bash
set -u
export PATH=/data/work/envs/ligunity/bin:$PATH
B=/data/work/vs-benchmark
L=$B/results/logs
run() {  # $1=模型目录 $2=权重 $3=结果目录 $4=GPU $5=标签
  cd $B/code/$1
  nohup env CUDA_VISIBLE_DEVICES=$4 python ./unimol/test.py --user-dir ./unimol "./data" \
    --valid-subset test --results-path "$3" \
    --num-workers 4 --ddp-backend=c10d --batch-size 32 \
    --task drugclip --loss in_batch_softmax --arch drugclip \
    --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --seed 1 \
    --path "$2" --log-interval 100 --log-format simple \
    --max-pocket-atoms 511 --test-task DEKOIS \
    > "$L/$5_DEKOIS.log" 2>&1 &
  echo "  $5  GPU$4  PID=$!"
}
run DrugCLIP $B/ckpt/drugclip/checkpoint_best.pt        $B/results/drugclip          4 drugclip
sleep 5
run BindCLIP $B/ckpt/bindclip/BindCLIP_randneg.pt       $B/results/bindclip_randneg  5 bindclip_randneg
sleep 5
run BindCLIP $B/ckpt/bindclip/BindCLIP_hardneg.pt       $B/results/bindclip_hardneg  6 bindclip_hardneg
disown -a
