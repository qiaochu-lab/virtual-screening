"""Per-target pairing: retrieval models vs. the molecular-weight baseline.
Means don't count -- this project has already been burned by that three
times.

Reported by our finalized review standard: per-target table takes priority,
paired p comes with n and a floor, and the sign test counts how many are
"above the null hypothesis" (here, null = no difference between the two,
alternative = the difference is > 0).
"""
import json, collections, os
import numpy as np
from scipy import stats
from math import comb
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog("rdApp.*")

B = "/data/work/vs-benchmark"
MIN_ACT = 10
MODELS = ["hypseek_rk", "ligunity_protein_ranking", "ligunity_pocket_ranking",
          "litenclip", "drugclip"]

cache = {}
def mw(smi):
    if smi not in cache:
        m = Chem.MolFromSmiles(smi)
        cache[smi] = Descriptors.MolWt(m) if m else None
    return cache[smi]

def model_order(up, L, n, rec, labels):
    act = {x["smiles"] for x in rec["actives"]}
    def ok(seq):
        if seq is None or len(seq) != n: return None
        return seq if {seq[i] for i in range(n) if labels[i] == 1} == act else None
    r = ok([x["smiles"] for x in rec["actives"]] + [x["smiles"] for x in rec["decoys"]])
    if r is not None: return r
    p = f"{B}/data/T3_6A/{L}/{up}/{up}_lig.lmdb"
    if not os.path.exists(p): return None
    try:
        import lmdb, pickle
        e = lmdb.open(p, subdir=False, readonly=True, lock=False)
        out = []
        with e.begin() as t:
            for _k, v in t.cursor(): out.append(pickle.loads(v)["smi"])
        e.close()
    except Exception: return None
    return ok(out)

recs = {L: [json.loads(x) for x in open(f"{B}/data/t3/eval/{L}.jsonl")]
        for L in ("L1", "L2", "L3", "L4")}

print("逐靶点配对：模型 vs 分子量基线（同靶点、同活性集合）")
print("%-24s %-4s %5s %9s %9s %9s %10s %12s" %
      ("模型", "层", "n", "模型 ρ", "分子量 ρ", "Δ", "模型胜", "二项 p"))
print("-" * 92)
for m in MODELS:
    for L in ("L1", "L2", "L3", "L4"):
        d = f"{B}/results/t3_raw/{m}/T3/{L}"
        if not os.path.isdir(d): d = f"{B}/results/t3/{m}/{L}"
        if not os.path.isdir(d): continue
        pa_m, pa_w = [], []
        for r in recs[L]:
            up = r["uniprot"]
            try:
                p = np.load(f"{d}/{up}/saved_preds.npy").reshape(-1)
                y = np.load(f"{d}/{up}/saved_labels.npy")
            except Exception: continue
            if len(p) != len(y): continue
            order = model_order(up, L, len(y), r, y)
            if order is None: continue
            aff = {a["smiles"]: float(a["paff"]) for a in r["actives"]}
            pairs = []
            for i in range(len(y)):
                if y[i] != 1: continue
                s = order[i]
                if s in aff and mw(s) is not None:
                    pairs.append((float(p[i]), aff[s], mw(s)))
            if len(pairs) < MIN_ACT: continue
            sc = np.array([x[0] for x in pairs]); ap = np.array([x[1] for x in pairs])
            w = np.array([x[2] for x in pairs])
            if np.std(sc) == 0 or np.std(ap) == 0 or np.std(w) == 0: continue
            r1 = stats.spearmanr(sc, ap).statistic
            r2 = stats.spearmanr(w, ap).statistic
            if np.isfinite(r1) and np.isfinite(r2):
                pa_m.append(r1); pa_w.append(r2)
        if len(pa_m) < 5: continue
        a, b = np.array(pa_m), np.array(pa_w)
        diff = a - b
        nz = diff[diff != 0]
        k = int((nz > 0).sum()); n = len(nz)
        # Warning: two-sided exact binomial test. The first version only
        # computed P(X>=k), which only detects "the model wins" -- when
        # molecular weight wins it would return p=1, misreading "the model
        # is significantly worse" as "no difference".
        if n:
            pk = [comb(n, i) for i in range(n + 1)]
            bp = min(1.0, sum(x for x in pk if x <= pk[k]) / 2**n)
        else:
            bp = float("nan")
        print("%-24s %-4s %5d %+9.3f %+9.3f %+9.3f %8d/%-3d %11.4g %s" %
              (m, L, len(a), a.mean(), b.mean(), diff.mean(), k, n, bp, ("模型赢" if k > n/2 else "分子量赢") if bp < 0.05 else ""))
    print()
