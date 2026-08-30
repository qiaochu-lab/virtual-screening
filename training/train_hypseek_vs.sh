#!/usr/bin/env bash
# 训练 HypSeek 的**虚筛权重** _vs —— 官方只公开了排序权重 _rk，这个没有。
#
# 为什么值得训
# ------------
# _rk 是在 FEP 验证集上按排序挑的 checkpoint，却在我们测的每个维度都第一
# （DUD-E/DEKOIS/LIT-PCBA 三个虚筛基准 + T3 四层 + CASF）。
# 那么专门按虚筛指标挑的权重会怎样，是个没人答过的问题。
#
# 与官方 train.sh 的差别（都是必要改动，不是调参）
#   · mode=CASF：用 valid_bedroc 选 best checkpoint，这才是"虚筛权重"的选法；
#     FEP 模式选出来的就是已经公开的 _rk
#   · data_path 指向 LigUnity 的 test_datasets（HypSeek 自己不带数据，
#     README 里也写明数据来自 LigUnity）
#   · n_gpu 4 -> 3：我方同时占卡不超过 4 张，留一张给 T5 的推理
#   · save_root 从写死的 /save_root 改到工作区
#   · batch 24 -> 6 + update_freq 4：**这是必须的修正，不是调参**。
#     官方 batch 24 在 24GB 的 3090 上每个 batch 都 OOM，而 unicore 会
#     捕获 OOM 静默跳过该批——于是 50 个 epoch 一个梯度都没应用过
#     （实测正式那轮 94,000 次 "recover from OOM"，权重与预训练逐位相同）。
#     4 × 6(累积) × 4(卡) = 96，与官方有效批量一致。
#   · seed 可传参：合作者反映这类模型对种子敏感，单次结果说明不了问题，
#     所以第一次跑完要换种子复跑，报均值而不是单点
set -u
SEED=${1:-1}
GPUS=${2:-"5,6,7"}
B=/data/work/vs
E=/data/work/envs/litenclip

data_path="$B/code/LigUnity/test_datasets"
save_root="$B/train/hypseek_vs_seed${SEED}"
save_dir="${save_root}/savedir"; tmp_save_dir="${save_root}/tmp"; tsb_dir="${save_root}/tsb"
mkdir -p "$save_dir" "$tmp_save_dir" "$tsb_dir" "$B/results/logs"

n_gpu=$(echo "$GPUS" | tr ',' '\n' | wc -l)
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export PYTHONPATH="$B/code/HypSeek/unimol:${PYTHONPATH:-}"
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=$B/hf_cache
# 显存碎片整理：DDP 下每个 rank 的峰值比单卡探针高，batch 6 仍偶发 OOM，
# 而 unicore 的 OOM 恢复在 DDP 下不安全——一个 rank 跳批、其他继续通信会
# NCCL 失步直接 SIGABRT。所以宁可 batch 4 + 累积 6 步（4×6×4卡=96，同官方）。
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $B/code/HypSeek
CUDA_VISIBLE_DEVICES="$GPUS" LD_LIBRARY_PATH=$E/lib $E/bin/torchrun \
  --nproc_per_node=${n_gpu} --master_port=$((10068 + SEED)) \
  $E/bin/unicore-train "${data_path}" \
    --user-dir ./unimol \
    --task train_task --arch three_hybrid_model --loss three_hybrid_loss \
    --train-subset train --valid-subset valid --valid-set CASF \
    --best-checkpoint-metric valid_bedroc --maximize-best-checkpoint-metric \
    --num-workers 8 --ddp-backend c10d \
    --max-pocket-atoms 256 \
    --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-8 --clip-norm 1.0 \
    --lr-scheduler polynomial_decay --lr 1e-4 --warmup-ratio 0.06 \
    --max-epoch 50 --batch-size 4 --batch-size-valid 8 --update-freq 6 \
    --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
    --seed ${SEED} \
    --tensorboard-logdir ${tsb_dir} --log-interval 100 --log-format simple \
    --validate-interval 1 --all-gather-list-size 2048000 \
    --save-dir ${save_dir} --tmp-save-dir ${tmp_save_dir} \
    --keep-best-checkpoints 8 --keep-last-epochs 10 --find-unused-parameters \
    --finetune-pocket-model $B/ckpt/hypseek/pretrain/pocket_pre_220816.pt \
    --finetune-mol-model $B/ckpt/hypseek/pretrain/mol_pre_no_h_220816.pt \
    --max-lignum 16 --learn-curv --protein-similarity-thres 1.0 \
  > $B/results/logs/train_hypseek_vs_seed${SEED}.log 2>&1
echo "hypseek_vs seed=${SEED} exit=$?" >> $B/results/logs/train_done.log
