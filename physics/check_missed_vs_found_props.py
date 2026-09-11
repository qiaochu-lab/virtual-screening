"""missed / found 两组活性的分子性质对照——支撑 export_rerank_sub.py 的主分析解读。

**为什么需要这个。** 主分析比的是 Boltz 在「检索漏掉的活性 vs 诱饵」和
「检索找到的活性 vs 诱饵」上的 AUROC。如果 missed 明显低于 found，有两个解释：

    (a) 物理和检索的盲区重合——级联加物理这一级补不上什么
    (b) 补回的那批活性本身对任何方法都更难（更大、更新、更怪）

两个数分不开这个歧义，所以要单独查分子性质。

**结论：(b) 站不住，但必须逐靶点配对才看得出来。**
混合起来比，两组差别大得吓人（MW 中位 482 vs 422，p=3.3e-23）；
逐靶点配对之后全部消失。那个 p=3e-23 是**靶点构成效应**——检索失败的靶点
碰巧整体配体更大——不是同一靶点内部的分子差异。典型的 Simpson 悖论，
和这个项目里「均值会骗人」那条教训是同一个形状。

⚠️ **这条控制自身的限制**：能配对的靶点按构造是检索**部分成功**的那批
（两组都要有足够样本）。检索完全失败的靶点上（3 个靶点 found 精确为 0），
missed 活性有没有系统差异，这个方法测不了。
"""
import json
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen
from scipy import stats
RDLogger.DisableLog("rdApp.*")

m = json.load(open("/data/work/vs-benchmark/data/t3/rerank_manifest.json"))
tn = m["topn"]
props = {"MW": Descriptors.MolWt, "重原子": Descriptors.HeavyAtomCount,
         "logP": Crippen.MolLogP, "TPSA": Descriptors.TPSA,
         "可旋转键": Descriptors.NumRotatableBonds}
rows = []
for e in m["entries"]:
    if e["label"] != 1:
        continue
    mol = Chem.MolFromSmiles(e["smi"])
    if mol is None:
        continue
    rows.append((e["uniprot"], e["rank"] >= tn,
                 {k: f(mol) for k, f in props.items()}))
print(f"可解析活性 {len(rows)}")

print("\n=== 混合起来比（所有靶点倒进一个池子）===")
for k in props:
    a = [r[2][k] for r in rows if r[1]]      # missed
    b = [r[2][k] for r in rows if not r[1]]  # found
    p = stats.mannwhitneyu(a, b).pvalue
    print(f"  {k:6} missed 中位 {np.median(a):8.2f}   found 中位 {np.median(b):8.2f}   p={p:.2e}")

print("\n=== 逐靶点配对（两组都 >=3 个才算）===")
ups = sorted({r[0] for r in rows})
use = []
for up in ups:
    a = [r for r in rows if r[0] == up and r[1]]
    b = [r for r in rows if r[0] == up and not r[1]]
    if len(a) >= 3 and len(b) >= 3:
        use.append((up, a, b))
print(f"可配对靶点 {len(use)}: {[u for u, _, _ in use]}")
for k in props:
    d = [np.median([x[2][k] for x in a]) - np.median([x[2][k] for x in b])
         for _, a, b in use]
    p = stats.wilcoxon(d).pvalue if not np.allclose(d, 0) else float("nan")
    print(f"  {k:6} 中位差(missed-found) {np.median(d):+8.2f}   "
          f"missed 更大的 {sum(x > 0 for x in d)}/{len(d)}   p={p:.3f}")

print("\n=== 同样的配对，但只在 found>=10 的靶点上（AUROC 真正可估的那批）===")
use2 = [(u, a, b) for u, a, b in use if len(b) >= 10]
print(f"靶点 {len(use2)}: {[u for u, _, _ in use2]}")
for k in props:
    d = [np.median([x[2][k] for x in a]) - np.median([x[2][k] for x in b])
         for _, a, b in use2]
    p = stats.wilcoxon(d).pvalue if not np.allclose(d, 0) else float("nan")
    print(f"  {k:6} 中位差 {np.median(d):+8.2f}   missed 更大的 "
          f"{sum(x > 0 for x in d)}/{len(d)}   p={p:.3f}")
