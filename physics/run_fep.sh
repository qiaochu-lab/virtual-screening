#!/bin/bash
# 在 JACS + Merck 的 16 个 FEP 体系上跑各模型（2026-08-19 加入计划）。
#
# 为什么用这套：它测的是「同一靶点内按结合强弱排序」——正是我们 T2 测出
# 所有检索模型接近零的那个能力。而 JACS/Merck 是自由能领域用了十年的标准集，
# 物理方法的数字文献里可查，能把我们自建 T2 的结论锚定到公认基准上。
set -u
B=/data/work/vs-benchmark
L=$B/results/logs
mkdir -p $L

run_ligunity() {  # $1=变体 $2=GPU
  local V=$1 GPU=$2
  export PATH=/data/work/envs/ligunity/bin:$PATH
  cd $B/code/LigUnity
  CUDA_VISIBLE_DEVICES=$GPU python ./unimol/test.py ./test_datasets \
    --user-dir ./unimol --valid-subset test \
    --results-path "$B/results/fep/ligunity_${V}" \
    --num-workers 2 --ddp-backend=c10d --batch-size 8 \
    --task test_task --loss rank_softmax --arch ${V} \
    --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --seed 1 \
    --path "$B/ckpt/ligunity/LigUnity_VS/${V}_vs/checkpoint_avg_41-50.pt" \
    --log-interval 100 --log-format simple \
    --max-pocket-atoms 511 --test-task FEP \
    > "$L/fep_ligunity_${V}.log" 2>&1
  echo "ligunity_$V 退出码=$?"
}

run_litenclip() {  # $1=GPU
  local GPU=$1 E=/data/work/envs/litenclip
  cd $B/code/LiTENCLIP
  LD_LIBRARY_PATH=$E/lib CUDA_VISIBLE_DEVICES=$GPU $E/bin/python ./unimol/test.py ./test_datasets \
    --user-dir ./unimol --valid-subset test \
    --results-path "$B/results/fep/litenclip" \
    --num-workers 2 --ddp-backend c10d --batch-size 8 \
    --task test_task --loss rank_softmax --arch liten_clip \
    --seed 1 \
    --path "$B/ckpt/litenclip/checkpoint.best_valid_bedroc_0.50.pt" \
    --log-interval 100 --log-format simple \
    --max-pocket-atoms 2048 --test-task FEP \
    > "$L/fep_litenclip.log" 2>&1
  echo "litenclip 退出码=$?"
}

case "${1:-all}" in
  lig_pocket)  run_ligunity pocket_ranking  "${2:-4}" ;;
  lig_protein) run_ligunity protein_ranking "${2:-6}" ;;
  litenclip)   run_litenclip "${2:-7}" ;;
esac
