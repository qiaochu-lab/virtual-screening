"""Scoring power and ranking power on CASF-2016 (T2's third dataset).

Why this dataset
------------
T2 currently has two datasets with opposite conclusions: the FEP benchmark
(same chemical series) gets ρ≈0.4, while our own T3 (cross-series) gets
ρ≈0. CASF-2016 sits in between — **same target, but different ligand
scaffolds** — and it is the standard set the scoring-function field has used
for a decade, with physics/empirical scoring-function numbers available in
the literature for comparison.

Two official conventions, reported separately
------------
* scoring power: correlation computed jointly over all 285 complexes
  (cross-target, tests absolute affinity)
* ranking power: rank the 5 ligands within each target, then average over
  targets (within-target, tests ranking)
These measure different things, and reporting them mixed together produces
contradictory conclusions — T2 already learned this the hard way earlier.

Where the score comes from
----------
The model only writes embeddings to disk. score = dot product of the paired
pocket vector and molecule vector, consistent with the official
ensemble_result.py (one-to-one correspondence within the same complex, no
cross-pairing).

Targets are grouped by the uniprot field in casf_label_seq.json — CASF's 57
clusters are already structured as "5 ligands for the same protein", and
grouping by uniprot recovers that structure.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

B = "/data/work/vs-benchmark"
LAB = f"{B}/code/LigUnity/test_datasets/casf_label_seq.json"


def load_truth():
    out = {}
    for e in json.load(open(LAB)):
        pdb = e["pockets"][0]
        out[pdb] = (e["ligands"][0]["act"], e.get("uniprot", "?"))
    return out


def main(models):
    truth = load_truth()
    print("CASF-2016（285 个复合物）")
    print("=" * 78)
    print("%-24s %8s %10s %10s %12s %8s" %
          ("模型", "n", "打分ρ", "打分r", "靶点内排序ρ", "靶点数"))
    print("-" * 78)
    for m in models:
        d = f"{B}/results/{m}/PDBBind"
        try:
            ids = json.load(open(f"{d}/test_pdbbind_ids.json"))
            mol = np.load(f"{d}/test_mol_reps.npy")
            poc = np.load(f"{d}/test_pocket_reps.npy")
        except Exception as e:
            print(f"{m}: 读不到（{e}）")
            continue
        if not (len(ids) == len(mol) == len(poc)):
            print(f"{m}: 长度对不上 ids={len(ids)} mol={len(mol)} poc={len(poc)}，跳过")
            continue
        score = np.einsum("ij,ij->i", poc, mol)          # paired dot product
        y, grp = [], []
        for pdb in ids:
            a, up = truth.get(pdb, (np.nan, "?"))
            y.append(a); grp.append(up)
        y = np.array(y, dtype=float)
        ok = ~np.isnan(y)
        sp = stats.spearmanr(score[ok], y[ok]).statistic
        pr = stats.pearsonr(score[ok], y[ok]).statistic

        per = defaultdict(list)
        # score is a loaded array, while y/grp are built in the loop — a length
        # mismatch must raise, since a silent truncation would pair each target
        # with someone else's score.
        for s, a, g in zip(score, y, grp, strict=True):
            if not np.isnan(a):
                per[g].append((s, a))
        rhos = [stats.spearmanr([x[0] for x in v], [x[1] for x in v]).statistic
                for v in per.values() if len(v) >= 3 and np.std([x[0] for x in v]) > 0]
        rhos = [r for r in rhos if not np.isnan(r)]
        print("%-24s %8d %10.3f %10.3f %12.3f %8d" %
              (m, int(ok.sum()), sp, pr, float(np.mean(rhos)) if rhos else float("nan"),
               len(rhos)))

    print("-" * 78)
    print("\n怎么读")
    print("· 打分 ρ/r = 跨靶点比绝对结合强度（CASF 的 scoring power）")
    print("· 靶点内排序 ρ = 同一靶点 5 个不同骨架的配体谁强谁弱（ranking power）")
    print("· 与 T2 另外两套对照：FEP 同系列 ρ≈0.4，T3 跨系列 ρ≈0；")
    print("  CASF 是「同靶点、跨骨架」，正好落在中间，用来定位分界线在哪")


if __name__ == "__main__":
    main(sys.argv[1:] or ["pocket_ranking", "protein_ranking"])
