#!/bin/bash
set -u
export PATH=/data/work/envs/ligunity/bin:$PATH
cd /data/work/vs-benchmark/code/BindCLIP
B=/data/work/vs-benchmark
VAR=$1; GPU=$2
nohup env CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 \
  python ./unimol/test.py --user-dir ./unimol "./data" \
  --valid-subset test --results-path "$B/results/bindclip_${VAR}" \
  --num-workers 4 --ddp-backend=c10d --batch-size 8 \
  --task drugclip --loss in_batch_softmax --arch drugclip \
  --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --seed 1 \
  --path "$B/ckpt/bindclip/BindCLIP_${VAR}.pt" --log-interval 100 --log-format simple \
  --max-pocket-atoms 511 --test-task PCBA \
  > "$B/results/logs/bindclip_${VAR}_PCBA.log" 2>&1 &
echo "BindCLIP $VAR PCBA (bs=8) GPU$GPU PID=$!"
disown -a
