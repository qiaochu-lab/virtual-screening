"""Chemical-series oracle ceiling, run on the classic benchmarks -- how
much of T3's 98.7% is a product of its own construction.

Why this control is necessary
--------------------
On T3, this oracle (scores by "max Tanimoto to the target's other known
actives", leave-one-out) reaches 98.7% of the theoretical ceiling. That is
easy to misread as "virtual-screening benchmarks in general can be solved
by pure chemistry alone."

But Part 1's scaffold statistics show **T3's actives are roughly twice as
clustered as the classic benchmarks'**: scaffold/active median 0.44-0.49,
against DUD-E's **0.997** (one scaffold per active, because DUD-E
deduplicates by Bemis-Murcko scaffold during construction), DEKOIS 0.825,
LIT-PCBA 0.782.

So how much of that 98.7% is "the chemical world is just like this" versus
"T3's construction produced this" can only be known by running the same
oracle on the classic benchmarks. A reviewer will ask this; better to
answer it ourselves first.

Convention matches `ligand_only_baseline.py` exactly: ECFP4 r=2 fpSize=2048,
leave-one-out on actives, EF@1% rounded with ceil, ties broken by expected
value.
"""
import argparse
import csv
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
B = "/data/work/vs-benchmark"
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
BENCH = [("DUD-E", "DUDE"), ("DEKOIS 2.0", "DEKOIS"), ("LIT-PCBA", "PCBA")]


def fp(smi):
    m = Chem.MolFromSmiles(smi)
    return GEN.GetFingerprint(m) if m else None


def one_target(rec):
    af = [fp(a["smiles"]) for a in rec["actives"]]
    df = [fp(d["smiles"]) for d in rec["decoys"]]
    A = [x for x in af if x is not None]
    D = [x for x in df if x is not None]
    if len(A) < 2 or not D:
        return None
    s_a = [max(DataStructs.BulkTanimotoSimilarity(x, A[:i] + A[i + 1:]))
           for i, x in enumerate(A)]
    s_d = [max(DataStructs.BulkTanimotoSimilarity(x, A)) for x in D]
    return (rec.get("name") or rec["uniprot"],
            np.array(s_a + s_d, dtype=np.float64),
            np.array([1] * len(s_a) + [0] * len(s_d), dtype=np.int8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=f"{B}/data/t1")
    ap.add_argument("--metrics", default=f"{B}/eval")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=f"{B}/results/export/T1_oracle_ceiling.csv")
    args = ap.parse_args()
    sys.path.insert(0, args.metrics)
    from metrics import bedroc, enrichment_factor, roc_auc

    rows = [["benchmark", "target", "n_mol", "n_active", "ef1", "ef1_ceiling",
             "pct_of_ceiling", "bedroc", "auroc"]]
    print("化学系列 oracle 上界，跑在传统 benchmark 上")
    print("=" * 88)
    print("%-12s %6s %9s %9s %12s %9s %9s"
          % ("基准", "靶点", "EF@1%", "理论上限", "占上限", "BEDROC", "AUROC"))
    print("-" * 88)
    for label, key in BENCH:
        recs = [json.loads(x) for x in open(f"{args.eval_dir}/{key}.jsonl")]
        with ProcessPoolExecutor(args.workers) as ex:
            out = [r for r in ex.map(one_target, recs, chunksize=1) if r]
        ef, ceil_, bd, au, pct = [], [], [], [], []
        for name, s, y in out:
            n, na = len(y), int(y.sum())
            # ceiling = min(1/fraction, n_total/n_active), not 1/fraction
            c = min(100.0, n / na)
            e = enrichment_factor(s, y, 0.01)
            ef.append(e); ceil_.append(c); pct.append(e / c)
            bd.append(bedroc(s, y, 80.5)); au.append(roc_auc(s, y))
            rows.append([label, name, n, na, f"{e:.4f}", f"{c:.4f}",
                         f"{e/c:.4f}", f"{bd[-1]:.4f}", f"{au[-1]:.4f}"])
        print("%-12s %6d %9.2f %9.2f %11.1f%% %9.4f %9.4f"
              % (label, len(out), np.mean(ef), np.mean(ceil_),
                 100 * np.mean(pct), np.mean(bd), np.mean(au)))
    print("-" * 88)
    print("\n对照 T3（同一 oracle、同一口径）：四层都是 ~98.7% 占上限。")
    print("差距如果很大，说明 T3 的 98.7% 有相当一部分来自它自己的构造方式，")
    print("而不是「虚筛基准普遍能被纯化学解掉」。")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\n写入 {args.out}")


if __name__ == "__main__":
    main()
