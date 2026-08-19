#!/bin/bash
set -u
export PATH=/data/work/envs/ligunity/bin:$PATH
cd /data/work/vs-benchmark/code/BindCLIP
B=/data/work/vs-benchmark
VAR=$1      # hardneg | randneg
GPU=$2
TASK=$3     # DUDE | PCBA
W=$B/ckpt/bindclip/BindCLIP_${VAR}.pt
R=$B/results/bindclip_${VAR}
L=$B/results/logs
mkdir -p "$R" "$L"
nohup env CUDA_VISIBLE_DEVICES=$GPU python ./unimol/test.py --user-dir ./unimol "./data" \
  --valid-subset test --results-path "$R" \
  --num-workers 8 --ddp-backend=c10d --batch-size 64 \
  --task drugclip --loss in_batch_softmax --arch drugclip \
  --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --seed 1 \
  --path "$W" --log-interval 100 --log-format simple \
  --max-pocket-atoms 511 --test-task $TASK \
  > "$L/bindclip_${VAR}_${TASK}.log" 2>&1 &
echo "BindCLIP $VAR $TASK  GPU$GPU  PID=$!"
disown -a
