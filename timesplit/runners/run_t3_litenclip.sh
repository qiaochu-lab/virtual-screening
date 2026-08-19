#!/bin/bash
# LiTENCLIP 在 T3 上跑。$1=GPU
# 参数照搬官方 scripts/test_litenclip.sh 的 DEKOIS 分支（--arch liten_clip /
# --task test_task / --loss rank_softmax），只把 --test-task 换成 T3、加 --t3-root。
# batch 从官方 256 降到 8：T3 分子最大 336 原子（DEKOIS 才 50），
# UniMol 注意力 O(n^2)，官方参数会 CUDA OOM（DrugCLIP 上已实测）。
set -u
B=/data/work/vs-benchmark
E=/data/work/envs/litenclip
GPU=${1:-4}
cd $B/code/LiTENCLIP
LD_LIBRARY_PATH=$E/lib CUDA_VISIBLE_DEVICES=$GPU $E/bin/python ./unimol/test.py ./test_datasets \
  --user-dir ./unimol --valid-subset test \
  --results-path $B/results/t3_raw/litenclip \
  --num-workers 2 --ddp-backend c10d --batch-size 8 \
  --task test_task --loss rank_softmax --arch liten_clip \
  --bf16 --seed 1 \
  --path $B/ckpt/litenclip/checkpoint.best_valid_bedroc_0.50.pt \
  --log-interval 100 --log-format simple \
  --max-pocket-atoms 511 --test-task T3 --t3-root $B/data/T3_6A \
  > $B/results/logs/litenclip_T3.log 2>&1
echo "litenclip 退出码=$?"
