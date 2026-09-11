"""Recompute the T3 main table after removing "(target, molecule) pairs
already present in the training set".

Why this is needed
------------
T3 only does a temporal split (records ingested from 2025 onward); it does
not do a content-level set difference. check_pair_contamination.py measured
this: **20.9% of L1's pairs already exist in the training set** (only 0.01%
for L2, 0% for L3/L4). In other words, the L1 control layer is
systematically inflated, and the headline finding "L1->L4 decay is 64-77%"
is an **upper bound**.

This script removes the contaminated actives directly from the scoring
arrays and recomputes, to get a clean lower bound. No GPU needed: every
model's per-molecule scores are already on disk — this is just recomputing
metrics over a different set of indices.

How the molecule order is aligned
------------
Different models construct their input differently, so their on-disk order
differs too:
  * UniMol family (DrugCLIP/BindCLIP/LigUnity/LiTENCLIP/HypSeek) read
    data/T3_6A/{L}/{up}/{up}_lig.lmdb, skipping molecules with no conformer
    -> order follows lmdb
  * The rest (ConPLex/ConGLUDe/SPRINT) iterate the eval-set jsonl directly
    -> order is actives+decoys
The script infers which order applies from the length; if neither matches,
it skips that target and counts it — **it never guesses**, since a wrong
guess would attribute contamination to the wrong molecule, which is worse
than not checking at all.
"""
import argparse
import json
import os
import pickle
import sys

import lmdb
import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
B = "/data/work/vs-benchmark"
sys.path.insert(0, f"{B}/eval")
from metrics import bedroc, enrichment_factor, roc_auc   # noqa: E402

TD = f"{B}/code/LigUnity/test_datasets"
CACHE = f"{B}/data/t3/train_pairs.json"


def train_pairs():
    """The (uniprot, inchikey) pairs present in the training set. Computed once and cached, so a re-run is instant."""
    if os.path.exists(CACHE):
        d = json.load(open(CACHE))
        print(f"训练对（缓存）: {len(d):,}")
        return {tuple(x.split("\t")) for x in d}
    lab = json.load(open(f"{TD}/train_label_blend_seq_full.json"))
    pairs = set()
    for a in lab:
        up = a.get("uniprot")
        if not up:
            continue
        for l in a.get("ligands", []):
            smi = l.get("smi") if isinstance(l, dict) else None
            if not smi:
                continue
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            try:
                pairs.add((up, Chem.MolToInchiKey(m)))
            except Exception:
                pass
    json.dump(["\t".join(p) for p in sorted(pairs)], open(CACHE, "w"))
    print(f"训练对: {len(pairs):,}（已缓存）")
    return pairs


def lmdb_smis(path):
    if not os.path.exists(path):
        return None
    e = lmdb.open(path, subdir=False, readonly=True, lock=False)
    out = []
    with e.begin() as t:
        i = 0
        while True:
            raw = t.get(str(i).encode())
            if raw is None:
                break
            out.append(pickle.loads(raw)["smi"])
            i += 1
    e.close()
    return out


def contaminated_mask(up, L, n_pred, ikey_cache):
    """Returns a boolean array of length n_pred: True = this (target, molecule)
    combination already exists in the training set.

    Returns None if the order can't be matched; the caller then skips this target.
    """
    rec = EVAL[L].get(up)
    if rec is None:
        return None
    jsonl_smis = [m["smiles"] for m in rec["actives"]] + [m["smiles"] for m in rec["decoys"]]
    n_act = len(rec["actives"])

    smis = None
    if len(jsonl_smis) == n_pred:
        smis, acts = jsonl_smis, set(range(n_act))
    else:
        ls = lmdb_smis(f"{B}/data/T3_6A/{L}/{up}/{up}_lig.lmdb")
        if ls is not None and len(ls) == n_pred:
            # In lmdb, actives come first; the count is inferred from the eval set's active smiles set
            aset = {m["smiles"] for m in rec["actives"]}
            smis = ls
            acts = {i for i, s in enumerate(ls) if s in aset}
    if smis is None:
        return None

    mask = np.zeros(n_pred, dtype=bool)
    for i in acts:
        s = smis[i]
        ik = ikey_cache.get(s)
        if ik is None:
            m = Chem.MolFromSmiles(s)
            ik = Chem.MolToInchiKey(m) if m is not None else ""
            ikey_cache[s] = ik
        if ik and (up, ik) in TRAIN:
            mask[i] = True
    return mask


def metrics(p, y):
    return dict(ef1=enrichment_factor(p, y, 0.01),
                ef5=enrichment_factor(p, y, 0.05),
                bedroc=bedroc(p, y, 80.5),
                auroc=roc_auc(p, y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--layers", nargs="+", default=["L1", "L2", "L3", "L4"])
    ap.add_argument("--raw", default=f"{B}/results/t3_raw")
    args = ap.parse_args()

    ikey_cache = {}
    print("%-26s %-4s %6s %8s %8s %8s %8s %8s" %
          ("模型", "层", "靶点", "原EF1", "净EF1", "原AUROC", "净AUROC", "删掉分子"))
    print("-" * 92)
    summary = {}
    for m in args.models:
        for L in args.layers:
            # Two on-disk layouts: the UniMol family is t3_raw/<model>/T3/<layer>,
            # ConGLUDe/ConPLex is results/t3/<model>/<layer>
            d = f"{args.raw}/{m}/T3/{L}"
            if not os.path.isdir(d):
                d = f"{B}/results/t3/{m}/{L}"
            if not os.path.isdir(d):
                continue
            o, c, ndrop, nskip = [], [], 0, 0
            for up in sorted(os.listdir(d)):
                pp, lp = f"{d}/{up}/saved_preds.npy", f"{d}/{up}/saved_labels.npy"
                if not (os.path.exists(pp) and os.path.exists(lp)):
                    continue
                p, y = np.load(pp), np.load(lp)
                if p.ndim > 1:
                    p = p.reshape(-1)
                if len(p) != len(y) or y.sum() == 0:
                    continue
                o.append(metrics(p, y))
                mask = contaminated_mask(up, L, len(p), ikey_cache)
                if mask is None:
                    nskip += 1
                    c.append(o[-1])          # order didn't match: conservatively count it as-is
                    continue
                if not mask.any():
                    c.append(o[-1])
                    continue
                keep = ~mask
                ndrop += int(mask.sum())
                p2, y2 = p[keep], y[keep]
                c.append(metrics(p2, y2) if y2.sum() >= 5 else o[-1])
            if not o:
                continue
            f = lambda arr, k: float(np.mean([x[k] for x in arr]))
            summary[(m, L)] = (f(o, "ef1"), f(c, "ef1"))
            print("%-26s %-4s %6d %8.2f %8.2f %8.4f %8.4f %8d%s" %
                  (m, L, len(o), f(o, "ef1"), f(c, "ef1"),
                   f(o, "auroc"), f(c, "auroc"), ndrop,
                   f"  (顺序对不上 {nskip})" if nskip else ""))

    print("\n" + "=" * 92)
    print("L1→L4 衰减：污染剔除前 vs 剔除后")
    print("=" * 92)
    print("%-26s %14s %14s" % ("模型", "原始", "剔除污染后"))
    print("-" * 58)
    for m in args.models:
        a, b = summary.get((m, "L1")), summary.get((m, "L4"))
        if not a or not b:
            continue
        print("%-26s %13.1f%% %13.1f%%" %
              (m, (b[0] - a[0]) / a[0] * 100, (b[1] - a[1]) / a[1] * 100))
    print("\n注：L4 本来就没有污染，所以变化全部来自 L1 被拉低；"
          "剔除后的衰减是下界，原始值是上界。")


TRAIN = None
EVAL = None

if __name__ == "__main__":
    TRAIN = train_pairs()
    EVAL = {}
    for L in ["L1", "L2", "L3", "L4"]:
        p = f"{B}/data/t3/eval/{L}.jsonl"
        EVAL[L] = {}
        if os.path.exists(p):
            for line in open(p):
                r = json.loads(line)
                EVAL[L][r["uniprot"]] = r
    main()
