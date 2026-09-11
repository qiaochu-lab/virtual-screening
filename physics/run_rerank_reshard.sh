#!/bin/bash
# Boltz-2 重排的**续跑**：把四个 shard 剩下的亲和力计算重新均分到 4 张卡。
#
# 为什么要重新分片，而不是让原来的 shard 接着跑：
#
#   原分片是按**复合物**切的，四个 shard 各自独立推进，崩掉之后进度差得极远：
#     shard_0 剩 294   shard_1 剩 0（已完整）   shard_2 剩 847   shard_3 剩 0
#   墙上时间由最慢的 shard_2 决定 ≈ 847 × 25 秒 ≈ 6.4 小时，
#   而这期间另外三张卡是空的。重新均分到 4 张 ≈ 285 条/卡 ≈ 2.0 小时。
#
# ⚠️ 关键：**不能重跑结构阶段**。结构阶段四个 shard 其实都跑完了（3,729/3,729），
#    30 小时里贵的那一半已经在盘上，残的只有亲和力。如果新 shard 的 out_dir 里
#    没有已有的结构预测，Boltz 会**重新算结构**——1,141 条 × 约 3 分钟 = 57 GPU 小时，
#    比不重新分片还糟得多。
#
# 怎么避免：Boltz 判定「已存在」用的是
#     filter_inputs_structure:  {d.name for d in (outdir/"predictions").iterdir() if d.is_dir()}
#     filter_inputs_affinity:   (outdir/"predictions"/id/f"affinity_{id}.json").exists()
# 两处都是 pathlib，**都跟随软链**。所以把原 shard 的 predictions/<id> 目录软链进
# 新 shard 的 predictions/ 下，结构阶段就会整体跳过，亲和力结果也直接写回原目录树
# （软链指过去的是真实目录），汇总脚本的 glob 不用改，也不会重复计数。
#
# 这一条在正式跑之前用 3 条记录验过：
#     Found some existing predictions (3), skipping ...
#     Running affinity prediction for 3 inputs.
#   出分落回 shard_2 的真实目录，探针目录里 0 个真实 json。
#
# ⚠️ 杀进程不用会匹配到自己的模式。本脚本名 run_rerank_reshard.sh 不含
#    "run_rerank_sub.sh"，pgrep 的模式也不会匹配 bash 读脚本时的 argv。
#    （这个坑今天出现过四次，见 PATCHES.md。）
set -u
E=/data/work/envs/boltz2
B=/data/work/vs-benchmark
IN=$B/boltz_rerank_sub
OUT=$B/boltz_rerank_sub_out
RS=$B/boltz_rerank_reshard          # 新输入目录（软链）
RO=$B/boltz_rerank_reshard_out      # 新输出目录（predictions 全是软链）
LOG=$B/results/logs
NCH=4
mkdir -p "$LOG"
say(){ echo "[$(date '+%m-%d_%H:%M')] $*" | tee -a "$LOG/boltz_rerank_reshard.log"; }

# ---------- 1. 停掉旧的跑法 ----------
say "停掉旧的 run_rerank_sub.sh 及其 boltz 进程"
pkill -f 'run_rerank_sub\.sh' 2>/dev/null
for p in $(pgrep -f 'boltz predict' 2>/dev/null); do kill "$p" 2>/dev/null; done
sleep 15
for p in $(pgrep -f 'boltz predict' 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
sleep 5
say "  残留 boltz 进程 $(pgrep -cf 'boltz predict' 2>/dev/null || echo 0) 个"

# ---------- 2. 收集所有「有结构、无亲和力」的记录 ----------
rm -rf "$RS" "$RO"; mkdir -p "$RS" "$RO"
TODO=$(mktemp); NOSTRUCT=$(mktemp)
for S in 0 1 2 3; do
  P=$OUT/shard_$S/boltz_results_shard_$S/predictions
  [ -d "$P" ] || continue
  for y in "$IN/shard_$S"/*.yaml; do
    [ -e "$y" ] || continue
    d=$(basename "$y" .yaml)
    [ -f "$P/$d/affinity_$d.json" ] && continue          # 已出分
    if [ -f "$P/$d/pre_affinity_$d.npz" ]; then
      echo "$d|$S|$y|$P/$d" >> "$TODO"
    else
      echo "$d|$S" >> "$NOSTRUCT"                        # 毒丸，本应已隔离
    fi
  done
done
n_todo=$(wc -l < "$TODO"); n_bad=$(wc -l < "$NOSTRUCT")
say "待补亲和力 $n_todo 条；无结构产物（毒丸）$n_bad 条"
if [ "$n_bad" -gt 0 ]; then
  say "  ⚠️ 仍有毒丸没隔离，它们会让亲和力阶段整个退出，先移走："
  sed 's/|.*//' "$NOSTRUCT" | head -5 | while read -r d; do say "    $d"; done
  mkdir -p "$B/boltz_rerank_sub_quarantine"
  while IFS='|' read -r d S; do mv "$IN/shard_$S/$d.yaml" "$B/boltz_rerank_sub_quarantine/" 2>/dev/null; done < "$NOSTRUCT"
fi
if [ "$n_todo" -eq 0 ]; then say "没有要补的，直接汇总"; else

# ---------- 3. 均分成 NCH 份，建软链 ----------
i=0
while IFS='|' read -r d S y p; do
  k=$((i % NCH)); i=$((i+1))
  mkdir -p "$RS/c$k" "$RO/c$k/boltz_results_c$k/predictions"
  ln -sfn "$y" "$RS/c$k/$d.yaml"
  ln -sfn "$p" "$RO/c$k/boltz_results_c$k/predictions/$d"
done < "$TODO"
for k in $(seq 0 $((NCH-1))); do
  say "  c$k: $(ls "$RS/c$k"/*.yaml 2>/dev/null | wc -l) 条 → GPU$k"
done

# ---------- 4. 四张卡起跑（用户红线：最多 4 张）----------
for k in $(seq 0 $((NCH-1))); do
  [ -d "$RS/c$k" ] || continue
  CUDA_VISIBLE_DEVICES=$k nohup "$E/bin/boltz" predict "$RS/c$k" \
    --out_dir "$RO/c$k" --cache "$B/boltz_cache" \
    --accelerator gpu --devices 1 --no_kernels \
    --diffusion_samples 1 --output_format pdb --num_workers 2 \
    > "$LOG/boltz_rerank_reshard_c$k.log" 2>&1 &
  say "c$k 起在 GPU$k"
  sleep 20
done
wait
say "四个分片的进程都退出了"

# ---------- 5. 结构阶段有没有被误触发（这是最该查的失败模式）----------
for k in $(seq 0 $((NCH-1))); do
  if grep -qa "Running structure prediction" "$LOG/boltz_rerank_reshard_c$k.log" 2>/dev/null; then
    say "  ⚠️ c$k 触发了结构阶段——软链没被识别，检查 predictions 软链是否有效"
  fi
done
fi

# ---------- 6. 逐靶点核对，不满不汇总 ----------
# ⚠️ 不用「进程退出了」或日志里的结束标记当完成判据。上一轮就是因为追加日志里
#    留着上一次被杀掉那轮的「四个 shard 全部结束」，汇总链空跑了一次，
#    还把 68% 的部分结果当完整结果发布了一次。判据只认**产出文件数**。
tot=0; want=0; ok=1
for S in 0 1 2 3; do
  w=$(ls "$IN/shard_$S"/*.yaml 2>/dev/null | wc -l)
  g=$(find "$OUT/shard_$S" -name 'affinity_*.json' 2>/dev/null | wc -l)
  tot=$((tot+g)); want=$((want+w))
  [ "$g" -lt "$w" ] && { say "⚠️ shard_$S 仍未跑满：$g / $w"; ok=0; } || say "shard_$S 完成 $g/$w"
done
say "合计 $tot / $want"
if [ "$ok" = 1 ]; then
  say "✅ 全部跑满，汇总"
  /data/work/envs/ligunity/bin/python "$B/export_rerank_sub.py" \
    > "$B/results/export/T6_rerank_subset.txt" 2>&1
  say "  exit=$? → results/export/T6_rerank_subset.txt"
else
  say "❌ 没跑满，不汇总。export_rerank_sub.py 自己也有逐靶点闸门会拦。"
fi
