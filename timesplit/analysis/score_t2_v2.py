"""T2 affinity ranking — corrected version, aligned by molecule identity
instead of by index order.

Why this needed a rewrite
------------
The old score_t2.py took values like this:
    act = np.nonzero(lab == 1)[0]     # index of actives in the model's order (ascending)
    sc  = s[act]
    ρ   = spearman(sc, pa)            # pa comes from the eval-set jsonl's active order
This assumes "the k-th active in the model's order is the k-th active in
the eval set". That holds for ConGLUDe / ConPLex (they iterate the jsonl
directly); it **does not hold** for the UniMol family
(DrugCLIP/BindCLIP/LigUnity/LiTENCLIP/HypSeek) — they read from lmdb, and
lmdb keys are strings, so iteration order is lexicographic
(0, 1, 10, 100, 1000, …), not the numeric order they were written in.

Consequence: scores and affinities get paired up scrambled, and the
correlation is flattened toward zero. This turns out to explain something
that had never quite made sense — ConGLUDe (the only model that follows
jsonl order) has the highest ρ on T3 of all models (L1 +0.129), while the
seven structure models all sit near 0. It isn't stronger — it's just the
only one this bug didn't scramble.

The fix
--------
Align by **molecule identity**: first reconstruct the molecule order the
model actually saw (lmdb cursor order or jsonl order), then use InChIKey to
match each active to its own pAffinity before computing correlation.
InChIKey is computed on the fly from SMILES, so it doesn't rely on any
ordering assumption.

The old-convention numbers are also reported alongside, so the size of this
correction can be audited.
"""
import argparse
import json
import os
import pickle

import lmdb
import numpy as np
from rdkit import Chem, RDLogger
from scipy import stats

# Order guard: verify the LMDB cursor order matches saved_labels before
# concatenating per-molecule data. This bug cost T2 two wrong conclusions,
# and it fails completely silently. See eval/order_guard.py
import sys as _sys
_sys.path.insert(0, "/data/yicheng/xqc/vs-benchmark/eval")
from order_guard import assert_cursor_order as _assert_order
_assert_order()


RDLogger.DisableLog("rdApp.*")
B = "/data/work/vs-benchmark"
MIN_ACT = 10


def model_smiles(up, L, n_pred, rec):
    """The molecule order the model saw -> [smiles]; tries both layouts, returns None if neither matches."""
    jsonl = [m["smiles"] for m in rec["actives"]] + [m["smiles"] for m in rec["decoys"]]
    if len(jsonl) == n_pred:
        return jsonl
    p = f"{B}/data/T3_6A/{L}/{up}/{up}_lig.lmdb"
    if not os.path.exists(p):
        return None
    e = lmdb.open(p, subdir=False, readonly=True, lock=False)
    out = []
    with e.begin() as t:
        for _k, v in t.cursor():      # cursor order = the order the model saw
            out.append(pickle.loads(v)["smi"])
    e.close()
    return out if len(out) == n_pred else None


def ikey(s, cache):
    if s in cache:
        return cache[s]
    m = Chem.MolFromSmiles(s)
    try:
        k = Chem.MolToInchiKey(m) if m is not None else ""
    except Exception:
        k = ""
    cache[s] = k
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--layers", nargs="+", default=["L1", "L2", "L3", "L4"])
    ap.add_argument("--out", default=f"{B}/results/t3/summary_t2_v2.json")
    args = ap.parse_args()

    EV = {}
    for L in args.layers:
        p = f"{B}/data/t3/eval/{L}.jsonl"
        EV[L] = {json.loads(x)["uniprot"]: json.loads(x) for x in open(p)} \
            if os.path.exists(p) else {}

    cache, summary = {}, {}
    print("T2 亲和力排序：按分子身份对齐（修正）vs 按下标顺序（旧口径）")
    print("=" * 92)
    print("%-26s %-4s %7s %14s %14s %10s %9s" %
          ("模型", "层", "靶点", "修正 ρ", "旧口径 ρ", "修正 τ", "ρ>0 占比"))
    print("-" * 92)
    for m in args.models:
        summary[m] = {}
        for L in args.layers:
            d = f"{B}/results/t3_raw/{m}/T3/{L}"
            if not os.path.isdir(d):
                d = f"{B}/results/t3/{m}/{L}"
            if not os.path.isdir(d):
                continue
            new_r, old_r, new_t, ns, skip = [], [], [], [], 0
            ups = []   # per-target trace: used when re-aggregating over a different target subset, without re-running
            for up in sorted(os.listdir(d)):
                rec = EV[L].get(up)
                if rec is None:
                    continue
                try:
                    s = np.load(f"{d}/{up}/saved_preds.npy").reshape(-1)
                    lab = np.load(f"{d}/{up}/saved_labels.npy")
                except Exception:
                    continue
                if len(s) != len(lab):
                    continue
                act_idx = np.nonzero(lab == 1)[0]
                if len(act_idx) < MIN_ACT:
                    continue
                smis = model_smiles(up, L, len(s), rec)
                if smis is None:
                    skip += 1
                    continue
                aff = {ikey(a["smiles"], cache): float(a["paff"]) for a in rec["actives"]}
                pairs = []
                for i in act_idx:
                    k = ikey(smis[i], cache)
                    if k in aff:
                        pairs.append((float(s[i]), aff[k]))
                if len(pairs) < MIN_ACT:
                    skip += 1
                    continue
                sc = np.array([x[0] for x in pairs])
                pa = np.array([x[1] for x in pairs])
                if np.std(sc) == 0 or np.std(pa) == 0:
                    continue
                r = stats.spearmanr(sc, pa).statistic
                t = stats.kendalltau(sc, pa).statistic
                if not np.isfinite(r):
                    continue
                new_r.append(r); new_t.append(t); ns.append(len(pairs))
                ups.append(up)

                # Old convention: model active indices (ascending) matched directly
                # against the eval-set active order
                pa_old = [float(a["paff"]) for a in rec["actives"]]
                if len(pa_old) == len(act_idx):
                    ro = stats.spearmanr(s[act_idx], pa_old).statistic
                    if np.isfinite(ro):
                        old_r.append(ro)
            if not new_r:
                continue
            new_r = np.array(new_r)
            summary[m][L] = {"n_targets": len(new_r),
                             "spearman": float(new_r.mean()),
                             "spearman_sem": float(new_r.std(ddof=1) / np.sqrt(len(new_r))),
                             "kendall": float(np.mean(new_t)),
                             "frac_positive": float((new_r > 0).mean()),
                             "median_n_actives": int(np.median(ns)),
                             "spearman_old": float(np.mean(old_r)) if old_r else None,
                             "per_target": [
                                 {"uniprot": u, "spearman": float(a),
                                  "kendall": float(b), "n_actives": int(c)}
                                 for u, a, b, c in zip(ups, new_r, new_t, ns)]}
            print("%-26s %-4s %7d %14s %14s %10.3f %8.0f%%" %
                  (m, L, len(new_r),
                   f"{new_r.mean():+.3f}±{new_r.std(ddof=1)/np.sqrt(len(new_r)):.3f}",
                   f"{np.mean(old_r):+.3f}" if old_r else "—",
                   float(np.mean(new_t)), (new_r > 0).mean() * 100))
    print("-" * 92)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=1)
    print(f"\n写入 {args.out}")
    print("旧口径对 ConGLUDe/ConPLex 是对的（它们本来就按 jsonl 顺序打分），"
          "两列应当基本一致；结构模型两列差多少，就是这个 bug 压低了多少。")


if __name__ == "__main__":
    main()
