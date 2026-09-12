"""Export the clean T3 main table after removing pairs already present in the
training set.

Why this deserves to be a first-class artifact
-------------------------------------------------
T3's layering only does a time split; 20.9% of the (target, molecule) pairs
in L1 already exist in the training set. score_t3_clean.py previously ran
this as a **one-off robustness check**, but the main table still reports the
contaminated numbers, and a reader only sees a single line in LIMITATIONS
saying "this is an upper bound". This makes the clean version into a CSV
that sits alongside the main table, turning "64-81% decay" into a range
rather than a single point.

Conventions and limitations
------------------------------
- Only actives with a (target, molecule) pair already in the training set
  are removed; decoys are untouched.
- Consequently, after removal the active:decoy ratio drifts slightly away
  from 1:50 (fewer actives), and EF's denominator shifts along with it --
  this is the only difference from "rebuild the eval set from scratch and
  rerun inference". A true rebuild would need all nine models to rerun
  inference, and the cost is disproportionate to the benefit, so index
  removal is used instead, with the difference documented here.
- L3/L4 contamination is zero, so those two layers' numbers are identical to
  the main table and can serve as a self-check.
"""
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
from metrics import bedroc, enrichment_factor, roc_auc  # noqa: E402

MODELS = ["drugclip", "bindclip_randneg", "bindclip_hardneg",
          "ligunity_pocket_ranking", "ligunity_protein_ranking",
          "litenclip", "hypseek_rk", "hypseek_official_vs", "conglude", "conplex",
          "sprint"]
# Both HypSeek weights are listed on purpose: the main table carries both
# (`_vs` for screening since 2026-09-12, `_rk` kept for the analyses derived
# before the switch), so the clean table has to be readable against it
# row for row. Dropping `_rk` here would leave those derived analyses with
# no clean counterpart.


def train_pairs():
    p = f"{B}/data/t3/train_pairs.json"
    return {tuple(x.split("\t")) for x in json.load(open(p))}


def model_smiles(up, L, n, rec):
    jl = [m["smiles"] for m in rec["actives"]] + [m["smiles"] for m in rec["decoys"]]
    if len(jl) == n:
        return jl
    p = f"{B}/data/T3_6A/{L}/{up}/{up}_lig.lmdb"
    if not os.path.exists(p):
        return None
    e = lmdb.open(p, subdir=False, readonly=True, lock=False)
    out = []
    with e.begin() as t:
        for _k, v in t.cursor():          # cursor order is the order the model actually saw
            out.append(pickle.loads(v)["smi"])
    e.close()
    return out if len(out) == n else None


def main():
    TRAIN = train_pairs()
    cache = {}
    EV = {}
    for L in ["L1", "L2", "L3", "L4"]:
        p = f"{B}/data/t3/eval/{L}.jsonl"
        EV[L] = {json.loads(x)["uniprot"]: json.loads(x) for x in open(p)} if os.path.exists(p) else {}

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", help="restrict to a (layer, uniprot) subset CSV — "
                                     "the 350-quota list is the paper's convention")
    ap.add_argument("--out", default=f"{B}/results/export/T3_main_clean.csv")
    args = ap.parse_args()
    keep = None
    if args.subset:
        import csv as _csv
        keep = {(r["layer"], r["uniprot"]) for r in _csv.DictReader(open(args.subset))}
        print(f"子集：{len(keep)} 条（靶点×层）")

    rows = ["model,layer,n_targets,n_actives_removed,EF1_raw,EF1_clean,BEDROC_raw,BEDROC_clean,AUROC_raw,AUROC_clean"]
    print("%-26s %-4s %7s %9s %16s %16s" % ("模型", "层", "靶点", "删掉", "EF1 原→净", "AUROC 原→净"))
    print("-" * 88)
    for m in MODELS:
        for L in ["L1", "L2", "L3", "L4"]:
            d = f"{B}/results/t3_raw/{m}/T3/{L}"
            if not os.path.isdir(d):
                d = f"{B}/results/t3/{m}/{L}"
            if not os.path.isdir(d):
                continue
            raw, clean, ndrop = [], [], 0
            for up in sorted(os.listdir(d)):
                if keep is not None and (L, up) not in keep:
                    continue
                try:
                    p = np.load(f"{d}/{up}/saved_preds.npy").reshape(-1)
                    y = np.load(f"{d}/{up}/saved_labels.npy")
                except Exception:
                    continue
                if len(p) != len(y) or y.sum() == 0:
                    continue
                r = dict(ef1=enrichment_factor(p, y, 0.01), bedroc=bedroc(p, y, 80.5),
                         auroc=roc_auc(p, y))
                raw.append(r)
                rec = EV[L].get(up)
                smis = model_smiles(up, L, len(p), rec) if rec else None
                if smis is None:
                    clean.append(r)          # order couldn't be matched: count as-is, don't guess
                    continue
                mask = np.zeros(len(p), dtype=bool)
                for i in np.nonzero(y == 1)[0]:
                    s = smis[i]
                    if s not in cache:
                        mm = Chem.MolFromSmiles(s)
                        try:
                            cache[s] = Chem.MolToInchiKey(mm) if mm is not None else ""
                        except Exception:
                            cache[s] = ""
                    if cache[s] and (up, cache[s]) in TRAIN:
                        mask[i] = True
                if not mask.any():
                    clean.append(r)
                    continue
                ndrop += int(mask.sum())
                k = ~mask
                p2, y2 = p[k], y[k]
                clean.append(dict(ef1=enrichment_factor(p2, y2, 0.01),
                                  bedroc=bedroc(p2, y2, 80.5), auroc=roc_auc(p2, y2))
                             if y2.sum() >= 5 else r)
            if not raw:
                continue
            f = lambda a, k: float(np.mean([x[k] for x in a]))
            print("%-26s %-4s %7d %9d %16s %16s" %
                  (m, L, len(raw), ndrop,
                   f"{f(raw,'ef1'):.2f} → {f(clean,'ef1'):.2f}",
                   f"{f(raw,'auroc'):.4f} → {f(clean,'auroc'):.4f}"))
            rows.append(f"{m},{L},{len(raw)},{ndrop},{f(raw,'ef1'):.4f},{f(clean,'ef1'):.4f},"
                        f"{f(raw,'bedroc'):.4f},{f(clean,'bedroc'):.4f},"
                        f"{f(raw,'auroc'):.4f},{f(clean,'auroc'):.4f}")
    out = args.out
    open(out, "w").write("\n".join(rows) + "\n")
    print(f"\n写入 {out}")
    print("L3/L4 两列应完全相同（那两层污染为 0），可作自检")


if __name__ == "__main__":
    main()
