#!/bin/bash
# 跑作者在 GitHub issue #4 里公开的官方 HypSeek 权重。
#
# 为什么要跑
# ----------
# 此前我们所有 HypSeek 数字用的都是官方 _rk（当时只有它公开）。合作者 2026-09-07
# 指出：_rk 是按 benchmark 选出来的 checkpoint，拿它测 DUD-E / PCBA 存在数据泄漏，
# 应该 _vs 跑虚筛、_rk 跑排序，分开测。issue #4 里作者同时给出了两个权重，
# 提问者验证过 _vs 能精确复现论文（EF1% 51.43 vs 论文 51.44）。
#
# 同时测 alpha_prot 两档
# ---------------------
# HypSeek 训练时 alpha_prot=1（启用蛋白序列通路），但 test_task.py 里默认取 0：
#     alpha_prot = getattr(self.args, "alpha_prot", 0)
# 而论文自己说去掉序列通路会掉性能。issue #4 的提问者实测 alpha_prot=1 时
# DUD-E EF1% 从 51.43 升到 53.04，比论文报的还高，作者未回复原因。
# 我们两档都报，让这个不一致显式化。
#
# 用法: ./run_official_hypseek.sh <vs|rk> [GPU]
set -u
WHICH=${1:?用法: $0 <vs|rk> [GPU]}
GPU=${2:-4}
B=/data/work/vs-benchmark
E=/data/work/envs/litenclip
CK=$B/ckpt/hypseek/official_checkpoint_avg_41-50_${WHICH}.pt
LOG=$B/results/logs/official_${WHICH}.log

[ -s "$CK" ] || { echo "checkpoint 不存在: $CK"; exit 1; }
mkdir -p "$B/results/logs"
say(){ echo "[$(date '+%m-%d_%H:%M')] $*" | tee -a "$LOG"; }
say "起：官方 _${WHICH}，GPU $GPU，checkpoint $(md5sum "$CK" | cut -c1-12)"

run(){   # $1=alpha_prot  $2=test-task  $3=results-path  $4=batch
  cd "$B/code/HypSeek"
  CUDA_VISIBLE_DEVICES=$GPU LD_LIBRARY_PATH=$E/lib \
  PYTHONPATH="$B/code/HypSeek/unimol" \
  HF_ENDPOINT=https://hf-mirror.com HF_HOME=$B/hf_cache \
  $E/bin/python ./unimol/test.py "$B/code/LigUnity/test_datasets" \
    --user-dir ./unimol --valid-subset test \
    --results-path "$3" \
    --num-workers 2 --ddp-backend c10d --distributed-world-size 1 \
    --batch-size "$4" \
    --task test_task --loss three_hybrid_loss --arch three_hybrid_model \
    --fp16 --seed 1 --path "$CK" \
    --log-interval 100 --log-format simple \
    --max-pocket-atoms 511 --test-task "$2" \
    ${2:+$([ "$2" = T3 ] && echo "--t3-root $B/data/T3_6A")} \
    --alpha-prot "$1" \
    > "$B/results/logs/official_${WHICH}_a$1_$2.log" 2>&1
  return $?
}

# alpha_prot 是我们新加的命令行参数，先确认 test.py 认；不认就退回默认值跑一档
if ! grep -q 'alpha-prot' "$B/code/HypSeek/unimol/test.py" 2>/dev/null; then
  say "⚠️ test.py 不接受 --alpha-prot，只跑默认档（alpha_prot=0）"
  ALPHAS="default"
else
  ALPHAS="0 1"
fi

for A in $ALPHAS; do
  TAG="hypseek_official_${WHICH}"
  [ "$A" != "default" ] && [ "$A" != "0" ] && TAG="${TAG}_a${A}"
  say "--- alpha_prot=$A ---"

  for BENCH in DUDE DEKOIS PCBA CASF; do
    if [ "$A" = "default" ]; then
      cd "$B/code/HypSeek"
      CUDA_VISIBLE_DEVICES=$GPU LD_LIBRARY_PATH=$E/lib PYTHONPATH="$B/code/HypSeek/unimol" \
      HF_ENDPOINT=https://hf-mirror.com HF_HOME=$B/hf_cache \
      $E/bin/python ./unimol/test.py "$B/code/LigUnity/test_datasets" \
        --user-dir ./unimol --valid-subset test --results-path "$B/results/$TAG" \
        --num-workers 2 --ddp-backend c10d --distributed-world-size 1 --batch-size 128 \
        --task test_task --loss three_hybrid_loss --arch three_hybrid_model --fp16 \
        --seed 1 --path "$CK" --log-interval 100 --log-format simple \
        --max-pocket-atoms 511 --test-task $BENCH \
        > "$B/results/logs/official_${WHICH}_${BENCH}.log" 2>&1
      say "  T1/$BENCH exit=$?"
    else
      run "$A" "$BENCH" "$B/results/$TAG" 128; say "  T1/$BENCH exit=$?"
    fi
  done

  say "  T1 算分"
  cd "$B/eval" && /data/work/envs/ligunity/bin/python score_ligunity.py \
      "$B/results/$TAG" >> "$LOG" 2>&1

  # T3 只在 _vs 上跑（虚筛任务），_rk 的 T3 我们已有
  if [ "$WHICH" = "vs" ]; then
    say "  T3 四层"
    if [ "$A" = "default" ]; then
      cd "$B/code/HypSeek"
      CUDA_VISIBLE_DEVICES=$GPU LD_LIBRARY_PATH=$E/lib PYTHONPATH="$B/code/HypSeek/unimol" \
      HF_ENDPOINT=https://hf-mirror.com HF_HOME=$B/hf_cache \
      $E/bin/python ./unimol/test.py "$B/code/LigUnity/test_datasets" \
        --user-dir ./unimol --valid-subset test \
        --results-path "$B/results/t3_raw/$TAG" \
        --num-workers 2 --ddp-backend c10d --distributed-world-size 1 --batch-size 8 \
        --task test_task --loss three_hybrid_loss --arch three_hybrid_model --fp16 \
        --seed 1 --path "$CK" --log-interval 100 --log-format simple \
        --max-pocket-atoms 511 --test-task T3 --t3-root "$B/data/T3_6A" \
        > "$B/results/logs/official_${WHICH}_T3.log" 2>&1
      say "  T3 exit=$?"
    else
      run "$A" T3 "$B/results/t3_raw/$TAG" 8; say "  T3 exit=$?"
    fi
    for L in L1 L2 L3 L4; do
      say "    $L 靶点数 $(ls "$B/results/t3_raw/$TAG/T3/$L" 2>/dev/null | wc -l)"
    done
  fi
done

say "汇总 T3"
cd "$B" && /data/work/envs/ligunity/bin/python collect_t3.py >> "$LOG" 2>&1
/data/work/envs/ligunity/bin/python score_t3.py \
  --models hypseek_rk hypseek_official_vs hypseek_vs_collab hypseek_vs_s1 \
  --layers L1 L2 L3 L4 >> "$LOG" 2>&1
say "全部完成"
