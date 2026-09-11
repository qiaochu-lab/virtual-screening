"""Chemical-series oracle ceiling (a target-conditioned ligand-similarity
oracle).

Warning: **the file name is a historical leftover -- this quantity is not a
"pure-ligand baseline".** Although it never looks at the protein structure,
it reads **that target's known active molecules** -- information no model
under evaluation ever gets. So it does not measure "how well can you do
without looking at the protein"; it measures "**given you already know what
binds this target, how far can pure chemical similarity get you**" -- it is
a **ceiling**.

The real pure-ligand baseline is in `ligand_only_learned.py`: it does not
even know which target it is, and its result is below random. The two differ
by two orders of magnitude, and mixing up the names would reverse the
conclusion.

Original description: using only 2D fingerprint similarity, can T3's decoys
be separated at all?

Why this is needed
--------------------
Mattsson & Walters (bioRxiv 2026.06.29.735309) point out that protein-ligand
affinity benchmarks are commonly propped up by data leakage -- a **pure
ligand model that never sees the protein** reaches r=0.66 on FEP+. If a
virtual-screening benchmark can be solved by a pure-ligand signal, then what
it measures is not "pocket-ligand matching".

This script runs the same check on T3: score each candidate molecule by "the
maximum Tanimoto to that target's known actives" (leave-one-out for actives,
never against itself), then compute EF/BEDROC/AUROC as usual.

Note this baseline has access to more information than any model -- it can
see that target's real active molecules, while a model cannot. So it is a
**ceiling**: what it measures is "how clustered this batch of actives is in
chemical space", i.e. how severe the analogous-series bias is, not what a
model can achieve.

How to read the result
-------------------------
- Close to random (EF~=1, AUROC~=0.5) -> the decoy design succeeded;
  cross-target real actives genuinely cannot be separated by "how
  drug-like it looks", so T3's enrichment can only come from
  target-specific information.
- Clearly above random -> this layer's actives cluster among themselves,
  a model could score well simply by memorizing the chemical series, and
  this floor must be subtracted when reporting EF.
"""
import argparse
import csv
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")

B = "/data/work/vs-benchmark"
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fp(smi):
    m = Chem.MolFromSmiles(smi)
    return GEN.GetFingerprint(m) if m else None


def one_target(rec):
    """Returns (uniprot, layer, scores, labels); leave-one-out for active
    molecules against themselves."""
    af = [fp(a["smiles"]) for a in rec["actives"]]
    df = [fp(d["smiles"]) for d in rec["decoys"]]
    keep_a = [i for i, x in enumerate(af) if x is not None]
    keep_d = [i for i, x in enumerate(df) if x is not None]
    A = [af[i] for i in keep_a]
    D = [df[i] for i in keep_d]
    if len(A) < 2 or not D:
        return None
    s_a = []
    for i, x in enumerate(A):                       # leave-one-out: exclude itself
        others = A[:i] + A[i + 1:]
        s_a.append(max(DataStructs.BulkTanimotoSimilarity(x, others)))
    s_d = [max(DataStructs.BulkTanimotoSimilarity(x, A)) for x in D]
    scores = np.array(s_a + s_d, dtype=np.float64)
    labels = np.array([1] * len(s_a) + [0] * len(s_d), dtype=np.int8)
    return rec["uniprot"], rec["layer"], scores, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="+", default=["L1", "L2", "L3", "L4"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-dir", default=f"{B}/data/t3/eval")
    ap.add_argument("--metrics", default=f"{B}/eval")
    ap.add_argument("--out", default=f"{B}/results/export/T3_ligand_only.csv")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, args.metrics)
    from metrics import enrichment_factor as ef_at, bedroc, roc_auc, pr_auc

    rows = [["layer", "uniprot", "n_actives", "n_decoys",
             "ef1", "ef5", "bedroc", "auroc", "pr_auc"]]
    per_layer = defaultdict(list)
    for L in args.layers:
        recs = [json.loads(l) for l in open(f"{args.eval_dir}/{L}.jsonl")]
        with ProcessPoolExecutor(args.workers) as ex:
            for r in ex.map(one_target, recs, chunksize=4):
                if r is None:
                    continue
                up, lay, s, y = r
                m = (ef_at(s, y, 0.01), ef_at(s, y, 0.05),
                     bedroc(s, y), roc_auc(s, y), pr_auc(s, y))
                rows.append([lay, up, int(y.sum()), int((1 - y).sum())]
                            + [f"{v:.4f}" for v in m])
                per_layer[lay].append(m)
        print(f"{L} 完成 {len(per_layer[L])} 个靶点", flush=True)

    print("\n纯配体基线（只用 2D 指纹，不看蛋白）")
    print("=" * 72)
    print("%-5s %8s %9s %9s %9s %9s %9s" %
          ("层", "靶点", "EF1%", "EF5%", "BEDROC", "AUROC", "PR-AUC"))
    print("-" * 72)
    for L in args.layers:
        v = per_layer.get(L)
        if not v:
            continue
        a = np.array(v)
        print("%-5s %8d %9.2f %9.2f %9.4f %9.4f %9.4f"
              % (L, len(v), *a.mean(axis=0)))
    print("-" * 72)
    print("随机基线： EF=1.00  BEDROC≈0.02  AUROC=0.500  PR-AUC≈0.020")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\n逐靶点写入 {args.out}")


if __name__ == "__main__":
    main()
