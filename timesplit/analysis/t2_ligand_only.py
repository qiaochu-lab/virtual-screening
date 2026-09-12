"""E2 — the two pure-ligand ranking baselines T2 never had.

T1 and T3 are bracketed from both sides: a target-conditioned similarity oracle
reaches ~99% of the structural ceiling, and a classifier that never learns which
target it is scoring lands below random. T2's ranking numbers had neither bound,
so "per-target Spearman +0.26" could not be read against anything.

Two baselines, both on T2's exact protocol — actives only, Spearman(prediction,
measured pAffinity) within each target, averaged across targets:

  oracle   Predict an active's pAff as the mean pAff of the five most
           ECFP4-similar **other actives of the same target** (leave-one-out).
           It reads that target's answers, which no evaluated model is given,
           so it is an upper bound, not a baseline the models compete with.

  learned  ECFP4 -> pAff regression, with GroupKFold by **uniprot** so the model
           never sees the target it is scoring. Grouping by target rather than
           by molecule is the point: a congeneric series split across a fold
           boundary would put near-duplicates on both sides and leak.

How to read the pair
--------------------
If `learned` sits near zero while `oracle` is high, T2's signal is entirely
"how similar is this molecule to what already binds this target" — the same
sentence the T3 leakage audit arrives at, reached independently on the ranking
task. If `learned` is clearly above zero, pAffinity has a component predictable
from chemistry alone (size, lipophilicity and the like), and every absolute T2
correlation needs that floor subtracted before it means anything.

Two regressor families are run because one is not evidence: ridge and random
forest have very different inductive biases on sparse binary fingerprints, and
a shared answer is worth more than either alone.

Note on a class of bug this script cannot have: everything here is computed from
the evaluation jsonl itself, so there is no join between a model's score array
and a separately ordered label array. The molecule-order mistakes that produced
two wrong conclusions elsewhere in this project are structurally impossible.
"""
import argparse
import csv
import json
import os

import numpy as np
from scipy import stats

B = "/data/work/vs-benchmark"
LAYERS = ["L1", "L2", "L3", "L4"]
MIN_ACT = 5          # Spearman on fewer than five points is not worth reporting
N_NEIGHBOURS = 5     # matches Graber's kNN ligand baseline


def fingerprints(smiles):
    """ECFP4 as a packed bit matrix. None for molecules RDKit cannot parse."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        out.append(None if m is None else gen.GetFingerprintAsNumPy(m).astype(np.uint8))
    return out


def load(eval_dir, layer, subset=None):
    """Per-target actives: smiles, pAff. Targets with < MIN_ACT actives are dropped."""
    recs = []
    with open(f"{eval_dir}/{layer}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if subset is not None and (layer, r["uniprot"]) not in subset:
                continue
            acts = r["actives"]
            if len(acts) < MIN_ACT:
                continue
            recs.append({"uniprot": r["uniprot"],
                         "smiles": [a["smiles"] for a in acts],
                         "paff": np.array([float(a["paff"]) for a in acts])})
    return recs


def oracle_rho(fps, paff):
    """Leave-one-out top-N-similar-neighbour prediction, then Spearman."""
    n = len(fps)
    ok = [i for i in range(n) if fps[i] is not None]
    if len(ok) < MIN_ACT:
        return None, 0
    X = np.array([fps[i] for i in ok], dtype=np.float32)
    y = paff[ok]
    inter = X @ X.T
    size = X.sum(axis=1)
    union = size[:, None] + size[None, :] - inter
    tan = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    np.fill_diagonal(tan, -1.0)                       # leave one out
    k = min(N_NEIGHBOURS, len(ok) - 1)
    idx = np.argpartition(-tan, k - 1, axis=1)[:, :k]
    pred = y[idx].mean(axis=1)
    if np.std(pred) == 0 or np.std(y) == 0:
        return None, len(ok)
    return float(stats.spearmanr(pred, y).statistic), len(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=f"{B}/data/t3/eval")
    ap.add_argument("--subset", help="restrict to a (layer, uniprot) subset CSV")
    ap.add_argument("--models", nargs="+", default=["ridge", "rf"])
    ap.add_argument("--out", default=f"{B}/results/export/T2_ligand_only.csv")
    ap.add_argument("--per-target-out", default=f"{B}/results/export/T2_ligand_only_per_target.csv")
    ap.add_argument("--pred-out", default=f"{B}/results/export/T2_ligand_only_predictions.csv",
                    help="per-molecule out-of-fold predictions, so the paired test can "
                         "score the baseline on exactly the ligands a model was scored on")
    a = ap.parse_args()

    subset = None
    if a.subset:
        subset = {(r["layer"], r["uniprot"]) for r in csv.DictReader(open(a.subset))}
        print(f"subset: {len(subset)} (layer, target) records")

    rows = [["baseline", "layer", "n_targets", "spearman_mean", "spearman_sem",
             "frac_positive", "median_n_actives"]]
    per_rows = [["baseline", "layer", "uniprot", "n_actives", "spearman"]]

    store = {}
    for L in LAYERS:
        recs = load(a.eval_dir, L, subset)
        if not recs:
            continue
        for r in recs:
            r["fps"] = fingerprints(r["smiles"])
        store[L] = recs
        print(f"{L}: {len(recs)} targets, {sum(len(r['paff']) for r in recs):,} actives")

    # ---------- oracle ----------
    for L, recs in store.items():
        vals, ns = [], []
        for r in recs:
            rho, n = oracle_rho(r["fps"], r["paff"])
            if rho is None or np.isnan(rho):
                continue
            vals.append(rho); ns.append(n)
            per_rows.append(["oracle", L, r["uniprot"], n, f"{rho:.4f}"])
        if vals:
            v = np.array(vals)
            rows.append(["oracle", L, len(v), f"{v.mean():.4f}",
                         f"{v.std(ddof=1) / np.sqrt(len(v)):.4f}",
                         f"{(v > 0).mean():.3f}", int(np.median(ns))])
            print(f"  oracle  {L}: n={len(v):4d}  rho={v.mean():+.4f}  frac+={(v > 0).mean():.3f}")

    # ---------- learned ----------
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold

    allX, ally, allg, allmeta = [], [], [], []
    for L, recs in store.items():
        for r in recs:
            for i, fp in enumerate(r["fps"]):
                if fp is None:
                    continue
                allX.append(fp); ally.append(r["paff"][i])
                allg.append(r["uniprot"]); allmeta.append((L, r["uniprot"]))
    X = np.array(allX, dtype=np.float32)
    y = np.array(ally, dtype=np.float32)
    g = np.array(allg)
    print(f"\nlearned: {X.shape[0]:,} molecules x {X.shape[1]} bits, "
          f"{len(set(allg)):,} target groups")

    pred_rows = [["baseline", "layer", "uniprot", "smiles", "pred_paff", "true_paff"]]
    allsmi = []
    for L, recs in store.items():
        for r in recs:
            for i, fp in enumerate(r["fps"]):
                if fp is not None:
                    allsmi.append(r["smiles"][i])

    for name in a.models:
        pred = np.zeros_like(y)
        gkf = GroupKFold(n_splits=5)
        for tr, te in gkf.split(X, y, groups=g):
            est = (Ridge(alpha=1.0) if name == "ridge" else
                   RandomForestRegressor(n_estimators=100, n_jobs=-1,
                                         min_samples_leaf=5, random_state=0))
            est.fit(X[tr], y[tr])
            pred[te] = est.predict(X[te])
        for i, (L, up) in enumerate(allmeta):
            pred_rows.append([f"learned_{name}", L, up, allsmi[i],
                              f"{pred[i]:.4f}", f"{y[i]:.4f}"])
        by = {}
        for i, (L, up) in enumerate(allmeta):
            by.setdefault((L, up), ([], []))
            by[(L, up)][0].append(pred[i]); by[(L, up)][1].append(y[i])
        agg = {}
        for (L, up), (p, t) in by.items():
            if len(p) < MIN_ACT or np.std(p) == 0 or np.std(t) == 0:
                continue
            rho = stats.spearmanr(p, t).statistic
            if np.isnan(rho):
                continue
            agg.setdefault(L, []).append((rho, len(p)))
            per_rows.append([f"learned_{name}", L, up, len(p), f"{rho:.4f}"])
        for L in LAYERS:
            if L not in agg:
                continue
            v = np.array([x[0] for x in agg[L]]); ns = [x[1] for x in agg[L]]
            rows.append([f"learned_{name}", L, len(v), f"{v.mean():.4f}",
                         f"{v.std(ddof=1) / np.sqrt(len(v)):.4f}",
                         f"{(v > 0).mean():.3f}", int(np.median(ns))])
            print(f"  {name:9s} {L}: n={len(v):4d}  rho={v.mean():+.4f}  frac+={(v > 0).mean():.3f}")

    if pred_rows:
        csv.writer(open(a.pred_out, "w", newline="")).writerows(pred_rows)
        print(f"wrote {a.pred_out}  ({len(pred_rows) - 1:,} molecules)")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    csv.writer(open(a.out, "w", newline="")).writerows(rows)
    csv.writer(open(a.per_target_out, "w", newline="")).writerows(per_rows)
    print(f"\nwrote {a.out}\n      {a.per_target_out}")


if __name__ == "__main__":
    main()
