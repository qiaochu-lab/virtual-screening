"""Quantify the effect of contamination: stratify ConGLUDe's L3/L4 results by
whether it saw that target during training.

If the "seen" group is clearly better -> contamination did inflate its
score, and its T3 result needs to be flagged separately. ConPLex serves as
a negative control: its training set has nothing to do with ConGLUDe's, so
if it also shows a difference between the same two groups, that difference
comes from the targets themselves rather than from contamination.
"""
import glob, json, os
import numpy as np
from scipy import stats
B = "/data/work/vs-benchmark"
D = f"{B}/tmp/conglude_train/LB_train_val/info"

seen = set()
for p in glob.glob(f"{D}/info_dicts/*.json"):
    try: d = json.load(open(p))
    except Exception: continue
    for u in (d.get("uniprot_ids") or []):
        if u: seen.add(u)
    t = d.get("target_name")
    if t and len(t) <= 10: seen.add(t)

s = json.load(open(f"{B}/results/t3/summary.json"))
print("ConGLUDe 训练靶点 %d 个\n" % len(seen))
print("%-12s %-4s %22s %22s %10s" % ("模型","层","它见过的靶点","它没见过的","p"))
print("-" * 76)
for m in ["conglude", "conplex"]:
    if m not in s: continue
    for L in ["L3", "L4"]:
        if L not in s[m]: continue
        rows = s[m][L]["per_target"]
        a = [r["ef1"] for r in rows if r["uniprot"] in seen]
        b = [r["ef1"] for r in rows if r["uniprot"] not in seen]
        if len(a) < 8 or len(b) < 8: continue
        p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
        f = lambda v: f"{np.mean(v):.2f}±{np.std(v,ddof=1)/np.sqrt(len(v)):.2f} (n={len(v)})"
        print("%-12s %-4s %22s %22s %10.4f%s" % (m, L, f(a), f(b), p, "  *" if p<0.05 else ""))
print("-" * 76)
print("ConPLex 是阴性对照——它与 ConGLUDe 的训练集无关，")
print("若它在两组间也有同样差异，说明差异来自靶点本身而非污染。")
