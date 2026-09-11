"""Freeze a per-target, per-molecule intermediate table for T3, so that any
future re-layering is just a re-aggregation.

Why this exists
----------------
"Has this target's model seen it" is not an objective fact, it's a label
that changes with whatever training set the model has. So L1-L4 has to be
redrawn per model, and more than once -- this project has already redrawn it
three times (the fall-through bug, per-model stratification, the union of
the structure half and the affinity half). How long each redraw takes
depends on what was kept:

- Only the summary table kept -> models must be rerun, one GPU-day for ten
  models
- The per-target, per-molecule intermediate table kept -> just a re-group-by,
  a few minutes of pure CPU

Three tables
------------
1. `T3_molecules.csv.gz`  unique molecules: mol_id, inchikey, smiles, two
                          novelty scores
2. `T3_index.csv.gz`      per-target, per-molecule: layer, target,
                          **position under both orderings**, mol_id, label,
                          pAffinity
3. `T3_model_order.csv`   which ordering each model used for each target,
                          and whether it passed strict validation

**The jsonl_pos / lmdb_pos columns in table 2 are the core of this setup.**
When a model reads an lmdb its cursor order is lexicographic
(0, 1, 10, 100, ...), which differs from the eval-set jsonl's
"actives+decoys" order while having the same length; comparing only the
length silently produces mismatches, and this has already wrecked every T2
conclusion once in this project and recurred in two other places
(PATCHES.md). Storing both orderings' positions once and for all means no
future recomputation ever has to re-derive it.
"""
import argparse, csv, gzip, json, os, pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

B = "/data/work/vs-benchmark"
LAYERS = ("L1", "L2", "L3", "L4")


def ikey(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return ""
    try:
        return Chem.MolToInchiKey(m)
    except Exception:
        return ""


def lmdb_order(up, L):
    p = f"{B}/data/T3_6A/{L}/{up}/{up}_lig.lmdb"
    if not os.path.exists(p):
        return None
    try:
        import lmdb
        e = lmdb.open(p, subdir=False, readonly=True, lock=False)
        out = []
        with e.begin() as t:
            for _k, v in t.cursor():
                out.append(pickle.loads(v)["smi"])
        e.close()
        return out
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=f"{B}/results/frozen")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--models", nargs="+", default=[
        "hypseek_rk", "ligunity_pocket_ranking", "ligunity_protein_ranking",
        "litenclip", "drugclip", "bindclip_randneg", "bindclip_hardneg",
        "conglude", "conplex", "sprint"])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    recs = {L: [json.loads(x) for x in open(f"{B}/data/t3/eval/{L}.jsonl")]
            for L in LAYERS}
    nov_p = json.load(open(f"{B}/data/t3/ligand_novelty.json"))
    nov_d = json.load(open(f"{B}/data/t3/ligand_novelty_drugclip.json"))

    smis = sorted({x["smiles"] for L in LAYERS for r in recs[L]
                   for g in ("actives", "decoys") for x in r[g]})
    print(f"唯一分子 {len(smis):,}，算 InChIKey…", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        keys = list(ex.map(ikey, smis, chunksize=500))
    mid = {s: i for i, s in enumerate(smis)}

    with gzip.open(f"{args.out_dir}/T3_molecules.csv.gz", "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mol_id", "inchikey", "smiles",
                    "novelty_pocketaffdb", "novelty_drugclip"])
        for i, s in enumerate(smis):
            w.writerow([i, keys[i], s,
                        f"{nov_p.get(s, -1):.4f}", f"{nov_d.get(s, -1):.4f}"])
    print(f"  写入 T3_molecules.csv.gz")

    # ---- per-target, per-molecule: store positions under both orderings together ----
    n_rows = 0
    n_nolmdb = 0
    with gzip.open(f"{args.out_dir}/T3_index.csv.gz", "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "uniprot", "jsonl_pos", "lmdb_pos",
                    "mol_id", "label", "paff"])
        for L in LAYERS:
            for r in recs[L]:
                up = r["uniprot"]
                jl = [(x["smiles"], 1, x.get("paff")) for x in r["actives"]] + \
                     [(x["smiles"], 0, None) for x in r["decoys"]]
                # Warning: len(lmdb) == len(jsonl) is not assumed. Some
                # molecules get dropped when the lmdb is built (RDKit parsing
                # or conformer generation failures) -- 413 of 546 targets have
                # a jsonl pool 1-9 molecules larger than their lmdb.
                # **The lmdb is the batch the model actually scored**, so
                # align by SMILES; a molecule present in jsonl but absent
                # from lmdb is recorded as -1, which is itself exactly the
                # information this table is meant to preserve.
                lo = lmdb_order(up, L)
                lpos = {}
                if lo is not None:
                    for i, s in enumerate(lo):
                        lpos.setdefault(s, i)
                else:
                    n_nolmdb += 1
                for j, (s, lab, pa) in enumerate(jl):
                    w.writerow([L, up, j, lpos.get(s, -1), mid[s], lab,
                                "" if pa is None else f"{float(pa):.3f}"])
                    n_rows += 1
            print(f"  {L} 完成，累计 {n_rows:,} 行", flush=True)
    print(f"写入 T3_index.csv.gz（{n_rows:,} 行；{n_nolmdb} 个靶点没有可用 lmdb）")

    # ---- the ordering actually used by each model for each target ----
    with open(f"{args.out_dir}/T3_model_order.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "layer", "uniprot", "n_molecules", "order_used"])
        for m in args.models:
            cnt = {"jsonl": 0, "lmdb": 0, "FAIL": 0}
            for L in LAYERS:
                d = f"{B}/results/t3_raw/{m}/T3/{L}"
                if not os.path.isdir(d):
                    d = f"{B}/results/t3/{m}/{L}"
                if not os.path.isdir(d):
                    continue
                for r in recs[L]:
                    up = r["uniprot"]
                    try:
                        y = np.load(f"{d}/{up}/saved_labels.npy")
                    except Exception:
                        continue
                    act = {x["smiles"] for x in r["actives"]}
                    jl = [x["smiles"] for x in r["actives"]] + \
                         [x["smiles"] for x in r["decoys"]]
                    used = "FAIL"
                    if len(jl) == len(y) and \
                       {jl[i] for i in range(len(y)) if y[i] == 1} == act:
                        used = "jsonl"
                    else:
                        lo = lmdb_order(up, L)
                        if lo is not None and len(lo) == len(y) and \
                           {lo[i] for i in range(len(y)) if y[i] == 1} == act:
                            used = "lmdb"
                    cnt[used] += 1
                    w.writerow([m, L, up, len(y), used])
            print(f"  {m}: jsonl {cnt['jsonl']} / lmdb {cnt['lmdb']} / "
                  f"验不过 {cnt['FAIL']}", flush=True)
    print(f"写入 T3_model_order.csv")


if __name__ == "__main__":
    main()
