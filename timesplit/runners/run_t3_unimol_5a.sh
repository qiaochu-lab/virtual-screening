#!/bin/bash
# 在 T3 上跑三个 UniMol 系模型。
#
# batch-size 从官方的 32 降到 8：T3 的分子来自 ChEMBL/BindingDB，
# 最大 336 个原子（DEKOIS 最大才 50）。UniMol 的注意力是 O(n^2)，
# 官方参数按 DUD-E/DEKOIS 的小分子调的，直接搬过来 70% 的靶点会 CUDA OOM。
# 另外在 test_t3_target 里逐靶点 empty_cache（1,044 个靶点，碎片会累积）。
# 其余参数与 run_dekois.sh 保持一致，保证横评时推理路径无差异。
# 只用 GPU 5、6。三个任务里两个共用 6 卡——单任务实测占 4-11 GB，
# 24 GB 的 3090 放得下两个（之前 6 个进程挤 3 张卡也跑得动）。
set -u
export PATH=/data/work/envs/ligunity/bin:$PATH
B=/data/work/vs-benchmark
L=$B/results/logs
mkdir -p $L
run() {  # $1=模型目录 $2=权重 $3=结果目录 $4=GPU $5=标签
  cd $B/code/$1
  nohup env CUDA_VISIBLE_DEVICES=$4 python ./unimol/test.py --user-dir ./unimol "./data" \
    --valid-subset test --results-path "$3" \
    --num-workers 4 --ddp-backend=c10d --batch-size 8 \
    --task drugclip --loss in_batch_softmax --arch drugclip \
    --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --seed 1 \
    --path "$2" --log-interval 100 --log-format simple \
    --max-pocket-atoms 511 --test-task T3 --t3-root "$B/data/T3_5A" \
    > "$L/$5_T3_5a.log" 2>&1 &
  echo "  $5  GPU$4  PID=$!"
}
run DrugCLIP $B/ckpt/drugclip/checkpoint_best.pt      $B/results/t3_raw_5a/drugclip          5 drugclip
sleep 5
run BindCLIP $B/ckpt/bindclip/BindCLIP_randneg.pt     $B/results/t3_raw_5a/bindclip_randneg  6 bindclip_randneg
sleep 5
run BindCLIP $B/ckpt/bindclip/BindCLIP_hardneg.pt     $B/results/t3_raw_5a/bindclip_hardneg  6 bindclip_hardneg
disown -a
