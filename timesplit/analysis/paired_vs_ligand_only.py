"""Per-target pairing: retrieval models vs the target-blind ligand-only baseline.

The companion to paired_vs_mw.py. That one asks whether a model beats molecular
weight; this one asks whether it beats a regressor that was never told which
target it is scoring — ECFP4 -> pAffinity, cross-validated with the target held
out (see t2_ligand_only.py). The baseline's predictions are read back per
molecule so both sides are scored on **exactly the same ligands of exactly the
same target**, which is what makes the pairing legitimate.

Means are not reported as evidence here. This project has been wrong three times
by comparing means, so the statistic is the per-target sign test: how many
targets the model wins, with an **exact two-sided** binomial p (one-sided would
return p=1 whenever the baseline wins, reading "significantly worse" as "no
difference"), and BH-FDR across every model x layer cell.
"""
import collections
import csv
import json
import os
from math import comb

import numpy as np
from scipy import stats

B = "/data/work/vs-benchmark"
MIN_ACT = 10
MODELS = ["hypseek_official_vs", "hypseek_rk", "ligunity_protein_ranking",
          "ligunity_pocket_ranking", "litenclip", "drugclip"]
LAYERS = ["L1", "L2", "L3", "L4"]


def model_order(up, L, n, rec, labels):
    """The molecule order a model's score array is in — jsonl order or lmdb cursor
    order. Returns None unless the actives land exactly where the labels say."""
    act = {x["smiles"] for x in rec["actives"]}

    def ok(seq):
        if seq is None or len(seq) != n:
            return None
        return seq if {seq[i] for i in range(n) if labels[i] == 1} == act else None

    r = ok([x["smiles"] for x in rec["actives"]] + [x["smiles"] for x in rec["decoys"]])
    if r is not None:
        return r
    p = f"{B}/data/T3_6A/{L}/{up}/{up}_lig.lmdb"
    if not os.path.exists(p):
        return None
    try:
        import pickle
        import lmdb
        e = lmdb.open(p, subdir=False, readonly=True, lock=False)
        out = []
        with e.begin() as t:
            for _k, v in t.cursor():
                out.append(pickle.loads(v)["smi"])
        e.close()
    except Exception:
        return None
    return ok(out)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=f"{B}/results/export/T2_ligand_only_predictions.csv")
    ap.add_argument("--label", default="ligand-only", help="name for the baseline column")
    ap.add_argument("--out", default=f"{B}/results/export/T2_paired_vs_ligand_only.csv")
    a = ap.parse_args()

    base = {}
    for r in csv.DictReader(open(a.pred)):
        base[(r["layer"], r["uniprot"], r["smiles"])] = float(r["pred_paff"])
    print(f"baseline predictions: {len(base):,} molecules")

    recs = {L: [json.loads(x) for x in open(f"{B}/data/t3/eval/{L}.jsonl")]
            for L in LAYERS}

    rows = [["model", "layer", "n_targets", "model_rho", "baseline_rho", "delta",
             "model_wins", "n_decided", "p_two_sided"]]
    collected = []
    print(f"\nper-target pairing: model vs the {a.label} baseline "
          f"(same target, same ligands)")
    print("%-26s %-4s %5s %9s %11s %9s %11s %12s"
          % ("model", "layer", "n", "model rho", "baseline rho", "delta",
             "model wins", "two-sided p"))
    print("-" * 96)
    for m in MODELS:
        for L in LAYERS:
            d = f"{B}/results/t3_raw/{m}/T3/{L}"
            if not os.path.isdir(d):
                d = f"{B}/results/t3/{m}/{L}"
            if not os.path.isdir(d):
                continue
            pa_m, pa_b = [], []
            for r in recs[L]:
                up = r["uniprot"]
                try:
                    p = np.load(f"{d}/{up}/saved_preds.npy").reshape(-1)
                    y = np.load(f"{d}/{up}/saved_labels.npy")
                except Exception:
                    continue
                if len(p) != len(y):
                    continue
                order = model_order(up, L, len(y), r, y)
                if order is None:
                    continue
                aff = {x["smiles"]: float(x["paff"]) for x in r["actives"]}
                trip = []
                for i in range(len(y)):
                    if y[i] != 1:
                        continue
                    s = order[i]
                    b = base.get((L, up, s))
                    if s in aff and b is not None:
                        trip.append((float(p[i]), aff[s], b))
                if len(trip) < MIN_ACT:
                    continue
                sc = np.array([x[0] for x in trip])
                ap_ = np.array([x[1] for x in trip])
                bl = np.array([x[2] for x in trip])
                if np.std(sc) == 0 or np.std(ap_) == 0 or np.std(bl) == 0:
                    continue
                r1 = stats.spearmanr(sc, ap_).statistic
                r2 = stats.spearmanr(bl, ap_).statistic
                if np.isfinite(r1) and np.isfinite(r2):
                    pa_m.append(r1); pa_b.append(r2)
            if len(pa_m) < 5:
                continue
            A, Bv = np.array(pa_m), np.array(pa_b)
            diff = A - Bv
            nz = diff[diff != 0]
            k = int((nz > 0).sum()); n = len(nz)
            if n:
                pk = [comb(n, i) for i in range(n + 1)]
                p2 = min(1.0, sum(x for x in pk if x <= pk[k]) / 2 ** n)
            else:
                p2 = float("nan")
            collected.append((m, L, len(A), A.mean(), Bv.mean(), diff.mean(), k, n, p2))
            print("%-26s %-4s %5d %+9.3f %+11.3f %+9.3f %7d/%-4d %12.4g"
                  % (m, L, len(A), A.mean(), Bv.mean(), diff.mean(), k, n, p2))

    # BH-FDR across every cell -- nominal p is not the standard here
    ps = sorted((c[8], i) for i, c in enumerate(collected) if np.isfinite(c[8]))
    N = len(ps)
    rejected = set()
    for rank, (pv, i) in enumerate(ps, start=1):
        if pv <= 0.05 * rank / N:
            rejected = {j for _, j in ps[:rank]}
    print("-" * 96)
    print(f"\nBH-FDR over {N} cells (alpha=0.05): {len(rejected)} rejected")
    for i, c in enumerate(collected):
        m, L, nt, am, bm, dm, k, n, p2 = c
        verdict = ""
        if i in rejected:
            verdict = "model wins" if k > n / 2 else "BASELINE wins"
        rows.append([m, L, nt, f"{am:.4f}", f"{bm:.4f}", f"{dm:.4f}", k, n, f"{p2:.6g}"])
        if verdict:
            print(f"  {m:26s} {L}  {verdict}  (p={p2:.4g})")

    csv.writer(open(a.out, "w", newline="")).writerows(rows)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
