"""Ranking-ability evaluation on the FEP benchmark (JACS 8 + Merck 8 = 16 systems).

This is the standard set the free-energy-calculation field has used for a
decade, and it tests "ranking by binding strength within the same target" --
exactly the ability our T2 measured every retrieval model at near zero on.
Its value here: physics methods (FEP+) have published numbers on these
systems, which lets our own T2 conclusion be anchored against a recognised
benchmark.

Note the official implementation only reports R^2, and zeroes it whenever
corr<0 -- which conflates "ranked backwards" with "completely unrelated".
Spearman (signed) is reported here as well.
"""
import json, os
import numpy as np
from scipy import stats

B = "/data/work/vs-benchmark"
JACS = {"bace","cdk2","jnk1","mcl1","p38","ptp1b","thrombin","tyk2"}

rows = {}
for m in sorted(os.listdir(f"{B}/results/fep")):
    root = f"{B}/results/fep/{m}/FEP"
    if not os.path.isdir(root): continue
    per = {}
    for t in sorted(os.listdir(root)):
        d = f"{root}/{t}"
        try:
            p = np.load(f"{d}/saved_preds.npy"); y = np.load(f"{d}/saved_labels.npy")
        except Exception:
            continue
        if len(p) != len(y) or np.std(p) == 0: continue
        per[t] = (stats.spearmanr(p, y).statistic,
                  stats.pearsonr(p, y).statistic, len(y))
    if per: rows[m] = per

sysnames = sorted({t for v in rows.values() for t in v})
print("FEP 基准：各体系 Spearman ρ（模型打分 vs 实测亲和力）")
print("=" * (14 + 26 * len(rows)))
print("%-11s %4s %5s %s" % ("体系","集合","n", " ".join(f"{m[:24]:>24s}" for m in rows)))
print("-" * (14 + 26 * len(rows)))
for t in sysnames:
    tag = "JACS" if t in JACS else "Merck"
    n = next((v[t][2] for v in rows.values() if t in v), 0)
    cells = []
    for m in rows:
        cells.append(f"{rows[m][t][0]:+.3f}".rjust(24) if t in rows[m] else "—".rjust(24))
    print("%-11s %4s %5d %s" % (t, tag, n, " ".join(cells)))
print("-" * (14 + 26 * len(rows)))

print("\n汇总")
print("=" * 72)
print("%-26s %10s %10s %12s %10s" % ("模型","Spearman","Pearson","方向正确率","体系数"))
print("-" * 72)
for m, per in rows.items():
    sp = np.array([v[0] for v in per.values()])
    pe = np.array([v[1] for v in per.values()])
    print("%-26s %+10.3f %+10.3f %11.0f%% %10d"
          % (m, sp.mean(), pe.mean(), (sp > 0).mean()*100, len(sp)))

print("\n分集合（JACS vs Merck）")
print("-" * 52)
for m, per in rows.items():
    j = [v[0] for t, v in per.items() if t in JACS]
    k = [v[0] for t, v in per.items() if t not in JACS]
    print("%-26s JACS %+.3f (n=%d)   Merck %+.3f (n=%d)"
          % (m, np.mean(j), len(j), np.mean(k), len(k)))
print("\n参照：FEP+ 等物理方法在这些体系上的 Pearson r 文献值通常在 0.6–0.8")
