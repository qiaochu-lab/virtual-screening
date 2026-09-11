"""The control the collaborator asked about: can molecular weight alone
reproduce AIMNet2's L3/L4 signal?

Why this is a critical control
-------------------------------
Physics-based scoring is inherently correlated with molecule size: a bigger
molecule makes more contacts with the pocket, so the interaction energy is
lower (i.e. "better"). And pAffinity itself is also positively correlated
with molecular weight (bigger molecules tend to bind more strongly). So a
"model" that only looks at molecular weight could get a positive Spearman
correlation out of nothing.

If molecular weight alone reproduces AIMNet2's +0.22/+0.12 on L3/L4, then that
"signal" is a size artifact, not physical recognition.
"""
import csv, collections
import numpy as np
from scipy import stats
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog("rdApp.*")

rows = list(csv.DictReader(open("{B}/data/t3/aimnet_t3_ligands.csv")))
byt = collections.defaultdict(list)
for r in rows:
    m = Chem.MolFromSmiles(r["smiles"])
    if m is None:
        continue
    try:
        pa = float(r["paff"]); comp = -float(r["composite"])
    except (ValueError, TypeError):
        continue
    inter = None
    try:
        inter = -float(r["interaction"])
    except (ValueError, TypeError):
        pass
    sm = None
    try:
        sm = -float(r["smina"])
    except (ValueError, TypeError):
        pass
    byt[(r["layer"], r["uniprot"])].append(
        dict(paff=pa, comp=comp, inter=inter, smina=sm,
             mw=Descriptors.MolWt(m), ha=m.GetNumHeavyAtoms()))

def rho(v, key):
    a = [x[key] for x in v]
    b = [x["paff"] for x in v]
    if any(x is None for x in a) or np.std(a) == 0 or np.std(b) == 0:
        return None
    r = stats.spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else None

cols = [("AIMNet2 复合", "comp"), ("AIMNet2 相互作用", "inter"),
        ("smina", "smina"), ("分子量", "mw"), ("重原子数", "ha")]
print("逐靶点 Spearman（模型分数 vs 实测 pAffinity），按层取均值")
print("%-4s %6s " % ("层", "靶点") + "".join("%18s" % n for n, _ in cols))
print("-" * 98)
store = collections.defaultdict(dict)
for L in ("L1", "L2", "L3", "L4"):
    keys = [k for k in byt if k[0] == L]
    cells = []
    for n, c in cols:
        vs = [rho(byt[k], c) for k in keys]
        vs = [x for x in vs if x is not None]
        store[L][c] = vs
        cells.append("%+.4f (n=%d)" % (np.mean(vs), len(vs)) if vs else "—")
    print("%-4s %6d " % (L, len(keys)) + "".join("%18s" % x for x in cells))

print()
print("逐靶点配对：AIMNet2 复合 vs 分子量（同一批靶点）")
print("%-4s %6s %12s %12s %10s %10s %8s" %
      ("层", "n", "AIMNet2", "分子量", "Δ", "配对 p", "AIM 胜"))
for L in ("L1", "L2", "L3", "L4"):
    keys = [k for k in byt if k[0] == L]
    pairs = [(rho(byt[k], "comp"), rho(byt[k], "mw")) for k in keys]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    if len(pairs) < 5:
        continue
    a = np.array([x[0] for x in pairs]); b = np.array([x[1] for x in pairs])
    p = stats.wilcoxon(a, b).pvalue
    floor = 2 / 2**len(pairs)
    print("%-4s %6d %12.4f %12.4f %+10.4f %10.4g %6d/%d   (下界 %.4f)" %
          (L, len(pairs), a.mean(), b.mean(), a.mean()-b.mean(), p,
           int((a > b).sum()), len(pairs), floor))
