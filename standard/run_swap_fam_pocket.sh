#!/bin/bash
# Target swap 全量：候选配体池不动，只换靶点身份。
#
# 目录布局的关键：模型的蛋白序列**按目录名查**，不在 lmdb 里。所以不能只替换
# _pocket.lmdb——build_target_swap.py 把目录命名成替身靶点 T'，再把原靶点 T 的
# 配体池软链进去。于是 swap 结果目录名是 T'，回读时必须用 swap_manifest.json
# 映射回 T 才能与正确口袋的结果配对。
# （试跑时我按 T 去查 swap 目录，取到的是「T 的口袋 + 别人的配体」，标签全对不上。）
#
# 每个模型的 arch / loss / 精度 / checkpoint 都不同，全部照抄各自的 run_t3_*.sh，
# 只改 --t3-root 和 --results-path 两处，保证 swap 与正确口袋走同一条推理路径。
#
# 用法: ./run_swap_full.sh <模型> <GPU> <round>
set -u
M=${1:?模型}; GPU=${2:-4}; RND=${3:-1}
B=/data/work/vs-benchmark
ROOT=$B/data/T3_swap_family/round${RND}
OUT=$B/results/t3_raw/swapfam_${M}_r${RND}
LOG=$B/results/logs/swapfam_${M}_r${RND}.log
mkdir -p "$OUT" "$B/results/logs"
[ -d "$ROOT" ] || { echo "swap 树不存在: $ROOT"; exit 1; }
echo "[$(date '+%m-%d_%H:%M')] $M round$RND GPU$GPU 起" >> "$LOG"

case "$M" in
  ligunity_pocket_ranking|ligunity_protein_ranking)
    V=${M#ligunity_}
    export PATH=/data/work/envs/ligunity/bin:$PATH
    cd $B/code/LigUnity
    CUDA_VISIBLE_DEVICES=$GPU python ./unimol/test.py "./test_datasets" --user-dir ./unimol \
      --valid-subset test --results-path "$OUT" \
      --num-workers 4 --ddp-backend=c10d --batch-size 8 \
      --task test_task --loss rank_softmax --arch ${V} \
      --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --seed 1 \
      --path "$B/ckpt/ligunity/LigUnity_VS/${V}_vs/checkpoint_avg_41-50.pt" \
      --log-interval 100 --log-format simple \
      --max-pocket-atoms 511 --test-task T3 --t3-root "$ROOT" >> "$LOG" 2>&1 ;;

  litenclip)
    E=/data/work/envs/litenclip
    cd $B/code/LiTENCLIP
    LD_LIBRARY_PATH=$E/lib CUDA_VISIBLE_DEVICES=$GPU $E/bin/python ./unimol/test.py ./test_datasets \
      --user-dir ./unimol --valid-subset test --results-path "$OUT" \
      --num-workers 2 --ddp-backend c10d --batch-size 8 \
      --task test_task --loss rank_softmax --arch liten_clip --bf16 --seed 1 \
      --path $B/ckpt/litenclip/checkpoint.best_valid_bedroc_0.50.pt \
      --log-interval 100 --log-format simple \
      --max-pocket-atoms 511 --test-task T3 --t3-root "$ROOT" >> "$LOG" 2>&1 ;;

  hypseek_rk)
    E=/data/work/envs/litenclip
    cd $B/code/HypSeek
    export PYTHONPATH="$B/code/HypSeek/unimol:${PYTHONPATH:-}"
    export HF_ENDPOINT=https://hf-mirror.com HF_HOME=$B/hf_cache
    LD_LIBRARY_PATH=$E/lib CUDA_VISIBLE_DEVICES=$GPU $E/bin/python ./unimol/test.py \
      "$B/code/LigUnity/test_datasets" --user-dir ./unimol --valid-subset test \
      --results-path "$OUT" \
      --num-workers 2 --ddp-backend c10d --distributed-world-size 1 --batch-size 8 \
      --task test_task --loss three_hybrid_loss --arch three_hybrid_model \
      --fp16 --seed 1 --path "$B/ckpt/hypseek/checkpoint_avg_41-50_rk.pt" \
      --log-interval 100 --log-format simple \
      --max-pocket-atoms 511 --test-task T3 --t3-root "$ROOT" >> "$LOG" 2>&1 ;;

  drugclip|bindclip_randneg|bindclip_hardneg)
    # 参数照抄 run_t3_unimol.sh：--task drugclip / --loss in_batch_softmax /
    # --arch drugclip，数据目录是各自代码库下的 ./data
    export PATH=/data/work/envs/ligunity/bin:$PATH
    case "$M" in
      drugclip)         DIR=DrugCLIP; CK=$B/ckpt/drugclip/checkpoint_best.pt ;;
      bindclip_randneg) DIR=BindCLIP; CK=$B/ckpt/bindclip/BindCLIP_randneg.pt ;;
      bindclip_hardneg) DIR=BindCLIP; CK=$B/ckpt/bindclip/BindCLIP_hardneg.pt ;;
    esac
    cd "$B/code/$DIR"
    CUDA_VISIBLE_DEVICES=$GPU python ./unimol/test.py --user-dir ./unimol "./data" \
      --valid-subset test --results-path "$OUT" \
      --num-workers 4 --ddp-backend=c10d --batch-size 8 \
      --task drugclip --loss in_batch_softmax --arch drugclip \
      --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --seed 1 \
      --path "$CK" --log-interval 100 --log-format simple \
      --max-pocket-atoms 511 --test-task T3 --t3-root "$ROOT" >> "$LOG" 2>&1 ;;

  *) echo "未配置: $M（序列模型走 run_swap_seq.sh；其余照抄 run_t3_*.sh 加分支）" >> "$LOG"; exit 1 ;;
esac

rc=$?
echo "[$(date '+%m-%d_%H:%M')] $M round$RND exit=$rc" >> "$LOG"
for L in L1 L4; do
  echo "  $L: $(ls "$OUT/T3/$L" 2>/dev/null | wc -l)" >> "$LOG"
done
