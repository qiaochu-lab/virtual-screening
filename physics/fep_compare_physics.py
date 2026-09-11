"""Compare the retrieval models against a physics method (Uni-FEP) on the same batch of FEP systems.

The physics method's reference values come from dptech-corp/Uni-FEP-Benchmarks,
which reports Kendall tau and R^2. Our retrieval models previously reported
Spearman; Kendall is computed here too so the two are directly comparable.

Warning: two points must be stated:
1. Those are **Uni-FEP** numbers, not Schrodinger FEP+ itself. Both are
   free-energy-perturbation-family methods, so the magnitude can serve as a
   reference point, but it must not be described as "FEP+'s result".
2. The ligand counts match our data one-to-one (BACE 36, CDK2 16,
   thrombin 11, ...), confirming these are the same systems -- the
   precondition for comparability, already checked.
"""
import json, os
import numpy as np
from scipy import stats

B = "/data/work/vs-benchmark"
ref = json.load(open(f"{B}/data/t3/fep_reference.json"))
phys = {**ref["jacs"], **ref["merck"]}
JACS = set(ref["jacs"])

rows = {}
for m in sorted(os.listdir(f"{B}/results/fep")):
    root = f"{B}/results/fep/{m}/FEP"
    if not os.path.isdir(root): continue
    per = {}
    for t in sorted(os.listdir(root)):
        d = f"{root}/{t}"
        try:
            p = np.load(f"{d}/saved_preds.npy"); y = np.load(f"{d}/saved_labels.npy")
        except Exception: continue
        if len(p) != len(y) or np.std(p) == 0: continue
        per[t] = {"kendall": stats.kendalltau(p, y).statistic,
                  "spearman": stats.spearmanr(p, y).statistic, "n": len(y)}
    if per: rows[m] = per

print("同一批 FEP 体系：检索模型 vs 物理方法（Kendall τ）")
print("=" * 96)
hdr = "%-11s %-6s %5s %5s" % ("体系", "集合", "我们n", "参考n")
for m in rows: hdr += " %20s" % m[:20]
hdr += " %12s" % "物理(Uni-FEP)"
print(hdr); print("-" * 96)
for t in sorted(phys):
    if not any(t in v for v in rows.values()): continue
    tag = "JACS" if t in JACS else "Merck"
    ourn = next((v[t]["n"] for v in rows.values() if t in v), 0)
    line = "%-11s %-6s %5d %5d" % (t, tag, ourn, phys[t]["n"])
    for m in rows:
        line += " %20s" % (f"{rows[m][t]['kendall']:+.3f}" if t in rows[m] else "—")
    line += " %12.3f" % phys[t]["kendall"]
    print(line)
print("-" * 96)
line = "%-11s %-6s %5s %5s" % ("均值", "", "", "")
for m in rows:
    line += " %20.3f" % np.mean([v["kendall"] for v in rows[m].values()])
line += " %12.3f" % np.mean([v["kendall"] for v in phys.values()])
print(line)

print("\n分集合（Kendall τ 均值）")
print("-" * 64)
print("%-26s %10s %10s" % ("方法", "JACS", "Merck"))
for m in rows:
    j = [v["kendall"] for t, v in rows[m].items() if t in JACS]
    k = [v["kendall"] for t, v in rows[m].items() if t not in JACS]
    print("%-26s %10.3f %10.3f" % (m, np.mean(j), np.mean(k)))
pj = [v["kendall"] for t, v in phys.items() if t in JACS]
pk = [v["kendall"] for t, v in phys.items() if t not in JACS]
print("%-26s %10.3f %10.3f" % ("物理方法 (Uni-FEP)", np.mean(pj), np.mean(pk)))
