#!/bin/bash
# HypSeek 在 T3 上跑。$1=GPU
# 用的是**排序权重** checkpoint_avg_41-50_rk.pt —— 官方只公开了这个，
# 虚筛权重 _vs.pt 未上传（HuggingFace THU-ATOM/HypSeek-AIRDD 里只有 _rk）。
# 这恰恰是最该跑 T2 的权重：其余七个模型用的都是虚筛导向权重，
# 亲和力排序全部接近零；HypSeek 的排序权重能区分
# 「短板来自架构」还是「来自训练目标与 checkpoint 选择」。
#
# 环境借用 litenclip（同为 torch 2.4 + unicore，推理路径依赖已补齐）。
set -u
B=/data/yicheng/xqc/vs-benchmark
E=/data/yicheng/xqc/envs/litenclip
GPU=${1:-5}
SEED=${2:-1}          # 之前 ckpt 与输出路径都硬编码成 seed1，给 seed2 用会静默覆盖 seed1 结果
cd $B/code/HypSeek
export PYTHONPATH="$B/code/HypSeek/unimol:${PYTHONPATH:-}"
# 它的序列塔要下 facebook/esm2_t12_35M_UR50D，服务器连不上 HuggingFace，走镜像
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$B/hf_cache
LD_LIBRARY_PATH=$E/lib CUDA_VISIBLE_DEVICES=$GPU $E/bin/python ./unimol/test.py \
  "$B/code/LigUnity/test_datasets" --user-dir ./unimol --valid-subset test \
  --results-path "$B/results/t3_raw/hypseek_vs_s$SEED" \
  --num-workers 2 --ddp-backend c10d --distributed-world-size 1 \
  --batch-size 8 \
  --task test_task --loss three_hybrid_loss --arch three_hybrid_model \
  --fp16 --seed 1 \
  --path "$B/train/hypseek_vs_seed$SEED/savedir/checkpoint_best.pt" \
  --log-interval 100 --log-format simple \
  --max-pocket-atoms 511 --test-task T3 --t3-root "$B/data/T3_6A" \
  > "$B/results/logs/hypseek_vs_s${SEED}_T3.log" 2>&1
echo "hypseek 退出码=$?"
