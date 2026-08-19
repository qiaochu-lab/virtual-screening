"""FEP 基准（JACS 8 + Merck 8 = 16 个体系）的排序能力评测。

这套是自由能计算领域十年来的标准集，测的是「同一靶点内按结合强弱排序」——
正是我们 T2 测出所有检索模型接近零的那个能力。用它的价值在于：
物理方法（FEP+）在这些体系上的数字文献里可查，能把自建 T2 的结论
锚定到公认基准上。

注意官方实现只报 R²，且 corr<0 时归零——那会把「排序方向反了」
和「完全无关」混为一谈。这里同时报 Spearman（带符号）。
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
