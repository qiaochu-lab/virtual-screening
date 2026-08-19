"""T5-a：同一批靶点上，用实验 holo 结构 vs Boltz-2 预测结构，结果差多少。

这是 T5「结构鲁棒性」的核心问题之一：模型对结构来源有多敏感？
材料是现成的——组装 T3 数据时每个靶点用的是哪种结构已逐靶点记录在
data/T3_6A/manifest.json 里，直接按来源切一刀即可，不用额外算。

注意这**不是**随机对照：一个靶点有没有实验结构本身就不随机
（研究得多的靶点才有），所以两组的靶点难度本来就可能不同。
因此同时报序列模型 ConPLex 作为参照——它不用结构，
若它在两组间也有同样的差距，说明差距来自靶点本身而非结构来源。
"""
import json, os
import numpy as np
from scipy import stats
B = "/data/yicheng/xqc/vs-benchmark"

man = json.load(open(f"{B}/data/T3_6A/manifest.json"))
src = {}
for L, d in man.items():
    for up, info in (d.get("per_target") or {}).items():
        src[(L, up)] = info["pocket_source"]

s = json.load(open(f"{B}/results/t3/summary.json"))
print("T5-a  结构来源对照（EF1%）")
print("=" * 78)
print("%-20s %-4s %6s %14s %6s %14s %9s" %
      ("模型", "层", "n(holo)", "实验 holo", "n(pred)", "Boltz-2 预测", "p"))
print("-" * 78)
for m in sorted(s):
    for L in ["L3", "L4"]:            # 只有新靶点层两种来源都有
        if L not in s[m]: continue
        rows = s[m][L]["per_target"]
        h = [r["ef1"] for r in rows if src.get((L, r["uniprot"])) == "pdb_holo"]
        p_ = [r["ef1"] for r in rows if src.get((L, r["uniprot"])) == "boltz2_pred"]
        if len(h) < 8 or len(p_) < 8:
            continue
        pv = stats.mannwhitneyu(h, p_, alternative="two-sided").pvalue
        f = lambda v: f"{np.mean(v):.2f}±{np.std(v,ddof=1)/np.sqrt(len(v)):.2f}"
        print("%-20s %-4s %6d %14s %6d %14s %9.4f%s" %
              (m, L, len(h), f(h), len(p_), f(p_), pv, "  *" if pv < 0.05 else ""))
print("-" * 78)
print("ConPLex 是序列模型、不用结构，若它也有同样差距，说明差距来自靶点本身")
