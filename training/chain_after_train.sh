#!/bin/bash
# 训练跑完 → 自检权重真的动了 → 评测 → 起 seed 2。全程 setsid，断网不影响。
#
# 自检这步是新加的：前两轮"跑完"其实一个参数都没更新（batch 24 在 24GB 上
# 每批都 OOM，unicore 静默跳过）。所以现在每轮训练结束都先比对权重与预训练初值，
# 距离为 0 就直接停下报错，不再往下浪费评测时间。
set -u
B=/data/work/vs
E=/data/work/envs/litenclip
PY=/data/work/envs/ligunity/bin/python
LOG=$B/results/logs/chain_after_train.log
say() { echo "[$(date +%m-%d_%H:%M)] $*" >> $LOG; }

for SEED in 1 2; do
  say "等 seed=$SEED 训练结束"
  while pgrep -u "$USER" -f unicore-train > /dev/null; do sleep 300; done

  CK=$B/train/hypseek_vs_seed${SEED}/savedir/checkpoint_best.pt
  [ -f "$CK" ] || { say "seed=$SEED 没有 checkpoint，跳过"; continue; }

  # ---- 自检：权重必须真的动过 ----
  MOVED=$($E/bin/python - "$CK" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu")["model"]
B = "/data/work/vs"
mol = torch.load(f"{B}/ckpt/hypseek/pretrain/mol_pre_no_h_220816.pt", map_location="cpu")
mol = mol["model"] if "model" in mol else mol
d = 0.0
for k, v in ck.items():
    if k.startswith("mol_model."):
        kk = k[len("mol_model."):]
        if kk in mol and mol[kk].shape == v.shape:
            d += float((v.float() - mol[kk].float()).norm()) ** 2
print(f"{d ** 0.5:.6f}")
PY
)
  say "seed=$SEED 分子塔与预训练初值的距离 = $MOVED"
  case "$MOVED" in
    0.000000) say "⚠️ seed=$SEED 权重未更新，跳过评测（同前两轮的 OOM 静默跳批）"; continue;;
  esac

  say "seed=$SEED 评测开始"
  for BENCH in DUDE DEKOIS PCBA CASF; do
    cd $B/code/HypSeek
    CUDA_VISIBLE_DEVICES=4 LD_LIBRARY_PATH=$E/lib PYTHONPATH=$B/code/HypSeek/unimol \
    HF_ENDPOINT=https://hf-mirror.com HF_HOME=$B/hf_cache \
    $E/bin/python ./unimol/test.py "$B/code/LigUnity/test_datasets" \
      --user-dir ./unimol --valid-subset test \
      --results-path $B/results/hypseek_vs_s${SEED} \
      --num-workers 2 --ddp-backend c10d --distributed-world-size 1 --batch-size 128 \
      --task test_task --loss three_hybrid_loss --arch three_hybrid_model --fp16 \
      --seed 1 --path $CK --log-interval 100 --log-format simple \
      --max-pocket-atoms 511 --test-task $BENCH \
      > $B/results/logs/hypseek_vs_s${SEED}_$BENCH.log 2>&1
    say "  seed=$SEED × $BENCH exit=$?"
  done
  cd $B && CUDA_VISIBLE_DEVICES=4 bash run_t3_hypseek_vs.sh 4 >> $LOG 2>&1
  say "seed=$SEED T3 完成"

  cd $B/eval && $PY score_ligunity.py $B/results/hypseek_vs_s${SEED} >> $LOG 2>&1
  say "seed=$SEED 算分完成"

  if [ "$SEED" = 1 ]; then
    say "起 seed=2 训练"
    setsid nohup bash $B/train_hypseek_vs.sh 2 "4,5,6,7" > /dev/null 2>&1 < /dev/null &
    sleep 60
  fi
done
say "链结束"
