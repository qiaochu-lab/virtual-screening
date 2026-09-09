#!/bin/bash
# 子集一变，所有依赖它的数字都得重算。这个脚本把「依赖子集」的分析串成一条，
# 免得手工漏掉某一项——2026-09-09 从 quota 250 切到 350 时就差点漏掉泄漏诊断那三项。
#
# 不依赖子集、因而不在这里的：T1（用公开基准）、HypSeek 官方权重的 T1 部分。
# 单独跑的：target swap（要重新推理，走 swap_queue.sh）。
#
# 用法: ./recompute_all.sh [子集CSV]
set -u
B=/data/work/vs-benchmark
P=/data/work/envs/ligunity/bin/python
S=${1:-$B/results/export/T3_vsds_matched.csv}
E=$B/results/export
L=$B/results/logs
MODELS='drugclip bindclip_randneg bindclip_hardneg ligunity_pocket_ranking ligunity_protein_ranking litenclip hypseek_rk conglude conplex sprint'
mkdir -p "$E" "$L"
cd "$B"
say(){ echo "[$(date '+%H:%M')] $*"; }

say "子集: $(basename "$S") — $(( $(wc -l < "$S") - 1 )) 条"

say "① summary.json 重算十个模型（下游都读它）"
$P score_t3.py --models $MODELS --layers L1 L2 L3 L4 > "$L/rc_score_t3.log" 2>&1
say "   exit=$?"

say "② T3 主表"
$P timesplit/analysis/score_subset.py --subset "$S" > "$L/rc_t3.log" 2>&1; say "   exit=$?"

say "③ T2 排序"
$P timesplit/analysis/score_t2_subset.py --subset "$S" > "$L/rc_t2.log" 2>&1; say "   exit=$?"

say "④ T5-a 结构来源"
$P t5_structure_source.py --models $MODELS --subset "$S" \
   --out "$E/T5_structure_source_subset.csv" > "$L/rc_t5a.log" 2>&1; say "   exit=$?"

say "⑤ T5-b 口袋阈值"
$P t5_threshold_curve.py --subset "$S" > "$E/T5_pocket_threshold_subset.txt" 2>&1; say "   exit=$?"

say "⑥ T5-c holo/apo"
$P t5_apo_compare.py --subset "$S" > "$E/T5_apo_subset.txt" 2>&1; say "   exit=$?"

say "⑦ T6 召回天花板"
$P shortlist_recall.py --subset "$S" > "$E/T6_recall_subset.txt" 2>&1; say "   exit=$?"

say "⑧ 靶点内部冗余"
$P timesplit/analysis/target_redundancy.py --targets "$S" \
   --out "$E/T3_target_redundancy.csv" > "$L/rc_redund.log" 2>&1; say "   exit=$?"

say "⑨ 新颖度分档 EF"
$P timesplit/analysis/novelty_tiered_ef.py --subset "$S" \
   --models ligunity_protein_ranking ligunity_pocket_ranking hypseek_rk litenclip drugclip conglude \
   --out "$E/T3_novelty_tiered_ef.csv" > "$L/rc_novelty_ef.log" 2>&1; say "   exit=$?"

say "⑩ 纯配体基线 + 归一化（从逐靶点 CSV 重新汇总，不重算指纹）"
$P - "$S" "$E" <<'EOF'
import csv, collections, sys
import numpy as np
S, E = sys.argv[1], sys.argv[2]
keep = {(r["layer"], r["uniprot"]) for r in csv.DictReader(open(S))}

lo = collections.defaultdict(list)
for r in csv.DictReader(open(f"{E}/T3_ligand_only.csv")):
    if (r["layer"], r["uniprot"]) in keep:
        lo[r["layer"]].append({k: float(r[k]) for k in ("ef1","ef5","bedroc","auroc","pr_auc")})
print("纯配体基线（上界，EF@1% 结构上限 51.00）")
out=[["layer","n_targets","ef1","ef5","bedroc","auroc","pr_auc","pct_of_ceiling"]]
base={}
for L in ("L1","L2","L3","L4"):
    v = lo.get(L)
    if not v: continue
    m = {k: float(np.mean([x[k] for x in v])) for k in v[0]}
    base[L] = m["ef1"]
    print("  %s n=%3d  EF1%% %6.2f (%.1f%% of 51)  AUROC %.4f"
          % (L, len(v), m["ef1"], 100*m["ef1"]/51, m["auroc"]))
    out.append([L, len(v)] + [f"{m[k]:.4f}" for k in ("ef1","ef5","bedroc","auroc","pr_auc")]
               + [f"{100*m['ef1']/51:.1f}"])
csv.writer(open(f"{E}/T3_ligand_only_subset.csv","w",newline="")).writerows(out)

mm = collections.defaultdict(dict)
for r in csv.DictReader(open(f"{E}/T3_main_vsds_subset.csv")):
    if r["layering"] == "corrected":
        mm[r["model"]][r["layer"]] = float(r["ef1"])
out2=[["model","layer","model_ef1","ligand_only_ceiling_ef1","fraction_of_ceiling"]]
print("\n归一化：模型 EF1%% ÷ 纯配体上界")
print("  %-26s %7s %7s %7s %7s" % ("模型","L1","L2","L3","L4"))
for m in sorted(mm, key=lambda x: -mm[x].get("L1", 0)):
    cells=[]
    for L in ("L1","L2","L3","L4"):
        v, b = mm[m].get(L), base.get(L)
        if v and b:
            cells.append(f"{100*v/b:.0f}%")
            out2.append([m, L, f"{v:.4f}", f"{b:.4f}", f"{v/b:.4f}"])
        else:
            cells.append("—")
    print("  %-26s %7s %7s %7s %7s" % (m, *cells))
csv.writer(open(f"{E}/T3_normalized_by_ceiling.csv","w",newline="")).writerows(out2)
EOF
say "   exit=$?"

say "✅ 全部重算完成。target swap 需单独跑 swap_queue.sh（要重新推理）"
