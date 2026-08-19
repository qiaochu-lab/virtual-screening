"""量化污染的影响：把 ConGLUDe 在 L3/L4 的结果按「它训练时见没见过」分层。

若「见过」组明显更好 → 污染确实抬高了它的成绩，其 T3 结果需单独标注。
用 ConPLex 作阴性对照：它的训练集与 ConGLUDe 无关，
若它在同样两组间也有差异，说明差异来自靶点本身而非污染。
"""
import glob, json, os
import numpy as np
from scipy import stats
B = "/data/yicheng/xqc/vs-benchmark"
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
