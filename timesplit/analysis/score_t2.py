"""T2 affinity ranking: can a model rank **relative binding strength**, not
just separate actives from inactives.

Why this task is nearly free
------------------------
Every active ligand in the T3 eval set carries a measured pAffinity, and
we've already stored every model's score for every molecule. Line the two up
and compute correlation — that's T2, no model needs to be re-run.

Why it tests a **qualitatively different** ability
--------------------------------
T1/T3 ask "can you pick out the actives", T2 asks "can you rank which one
binds stronger". This class of model is trained with contrastive learning
(pull binding pairs together, push non-binding pairs apart), and **the
objective function places no constraint at all on affinity order**. So it is
entirely possible to see "good enrichment but near-random ranking" — which
is itself a reportable finding.

Computed using only actives (decoys have no measured value). Each target
needs at least 10 actives to be counted; below that, the Spearman estimate
is too noisy.
"""
import argparse
import json
import os

import numpy as np
from scipy import stats

B = "/data/work/vs-benchmark"
MIN_ACT = 10


def load_truth(layer):
    """uniprot -> pAff list, in the same order as the smiles/inchikey list.

    The eval set is written with all actives first, then all decoys, and
    each model's runner scores in that same order, so the first n_actives
    scores correspond to the actives' pAff.
    """
    out = {}
    p = f"{B}/data/t3/eval/{layer}.jsonl"
    if not os.path.exists(p):
        return out
    for line in open(p):
        d = json.loads(line)
        out[d["uniprot"]] = [a["paff"] for a in d["actives"]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--layers", nargs="+", default=["L1", "L2", "L3", "L4"])
    ap.add_argument("--out", default=f"{B}/results/t3/summary_t2.json")
    args = ap.parse_args()

    truth = {L: load_truth(L) for L in args.layers}
    summary = {}

    print("T2 亲和力排序（只用 active，模型打分 vs 实测 pAffinity）")
    print("=" * 74)
    print("%-20s %-4s %7s %16s %16s" % ("模型", "层", "靶点", "Spearman ρ", "Kendall τ"))
    print("-" * 74)
    for m in args.models:
        summary[m] = {}
        for L in args.layers:
            d = f"{B}/results/t3/{m}/{L}"
            if not os.path.isdir(d):
                continue
            rho, tau, ns = [], [], []
            for up in sorted(os.listdir(d)):
                pa = truth[L].get(up)
                if not pa or len(pa) < MIN_ACT:
                    continue
                try:
                    s = np.load(f"{d}/{up}/saved_preds.npy")
                    lab = np.load(f"{d}/{up}/saved_labels.npy")
                except Exception:
                    continue
                act = np.nonzero(lab == 1)[0]
                # Lengths must match; otherwise the ordering assumption doesn't
                # hold, so skip rather than risk a mismatch
                if len(act) != len(pa):
                    continue
                sc = s[act]
                if np.std(sc) == 0:
                    continue
                r = stats.spearmanr(sc, pa)
                t = stats.kendalltau(sc, pa)
                if np.isfinite(r.statistic):
                    rho.append(r.statistic)
                    tau.append(t.statistic)
                    ns.append(len(pa))
            if not rho:
                continue
            rho, tau = np.array(rho), np.array(tau)
            summary[m][L] = {
                "n_targets": len(rho),
                "spearman": float(rho.mean()),
                "spearman_sem": float(rho.std(ddof=1) / np.sqrt(len(rho))),
                "kendall": float(tau.mean()),
                "frac_positive": float((rho > 0).mean()),
                "median_n_actives": int(np.median(ns)),
            }
            print("%-20s %-4s %7d %16s %16s"
                  % (m, L, len(rho),
                     f"{rho.mean():+.3f}±{rho.std(ddof=1)/np.sqrt(len(rho)):.3f}",
                     f"{tau.mean():+.3f}"))
    print("-" * 74)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=1)

    print("\n各模型「排序方向正确」的靶点占比（ρ>0，随机应为 50%）:")
    for m, d in summary.items():
        cells = [f"{L} {d[L]['frac_positive']*100:.0f}%" for L in args.layers if L in d]
        if cells:
            print(f"  {m:20s} " + "   ".join(cells))
    print("\n注：这类模型的对比学习目标不约束亲和力顺序，")
    print("    所以「富集好但排序接近随机」是可能的，且本身就是结论。")
    print(f"\n已写入 {args.out}")


if __name__ == "__main__":
    main()
