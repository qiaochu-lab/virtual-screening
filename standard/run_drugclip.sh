#!/bin/bash
set -u
export PATH=/data/work/envs/ligunity/bin:$PATH
cd /data/work/vs-benchmark/code/DrugCLIP
B=/data/work/vs-benchmark
W=$B/ckpt/drugclip/checkpoint_best.pt
R=$B/results/drugclip
L=$B/results/logs
mkdir -p "$R" "$L"

run() {  # $1=TASK $2=GPU
  nohup env CUDA_VISIBLE_DEVICES=$2 python ./unimol/test.py --user-dir ./unimol "./data" \
    --valid-subset test --results-path "$R" \
    --num-workers 8 --ddp-backend=c10d --batch-size 64 \
    --task drugclip --loss in_batch_softmax --arch drugclip \
    --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --seed 1 \
    --path "$W" --log-interval 100 --log-format simple \
    --max-pocket-atoms 511 --test-task $1 \
    > "$L/drugclip_$1.log" 2>&1 &
  echo "DrugCLIP $1  GPU$2  PID=$!"
}
run DUDE 0
sleep 8
run PCBA 1
disown -a
