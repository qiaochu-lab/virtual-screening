"""Exact molecule overlap (InChIKey) between T3 and the training sets.

Why this exists
---------------
`tasks/T3-leakage.md` §3a is the single number finding 5 leads with -- "32.1%
of L1 actives are literally the same molecules as ones in the training set" --
and until now it had **no committed script**. The document pointed at
`ligand_novelty.py`, but that script only computes maximum Tanimoto; nothing
in the repository reproduced the exact-match table. Same reproducibility gap
`finding18_permutation.py` closed for finding 18.

Exact vs. near
--------------
Tanimoto >= 0.7 and "same InChIKey" answer different questions. The Tanimoto
tiers say how *familiar* the chemistry is; this says whether the molecule is
the same compound. The time split guarantees the *record* (molecule, target,
assay) is new -- it does not stop a molecule tested against target A before
the cutoff from reappearing against target B after it. That is exactly what
this measures.

Two references, and they disagree by an order of magnitude
----------------------------------------------------------
The affinity half (`train_label_blend_seq_full.json`, ~428k ligands) trains
HypSeek / LigUnity x2 / LiTENCLIP; the structure half
(`train_no_test_af/train.lmdb`, 13,590 ligands) trains DrugCLIP / BindCLIP x2.
"L1 is nearly a memorisation test" is a statement about the first group only,
so both are reported side by side rather than one being called "the" training
set.

Decoys are the control: they are real actives on *other* targets drawn from
the same chemical universe, so their overlap rate is the background an actives
rate has to beat.
"""
import argparse
import csv
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

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


def keys_of(smis, workers):
    """SMILES -> InChIKey, in parallel. Empty string means RDKit refused the
    molecule; those are counted as "no match" rather than silently colliding
    with each other, which is why the empty key is filtered out of every set."""
    with ProcessPoolExecutor(workers) as ex:
        out = list(ex.map(ikey, smis, chunksize=500))
    return dict(zip(smis, out, strict=True))


def train_smis_affinity(path):
    lab = json.load(open(path))
    return {l["smi"] for a in lab for l in (a.get("ligands") or [])
            if isinstance(l, dict) and l.get("smi")}


def train_smis_structure(path):
    import lmdb
    e = lmdb.open(path, subdir=False, readonly=True, lock=False, readahead=False)
    out = set()
    with e.begin() as t:
        for _k, v in t.cursor():
            d = pickle.loads(v)
            if d.get("smi"):
                out.add(d["smi"])
    e.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-label",
                    default=f"{B}/code/LigUnity/test_datasets/train_label_blend_seq_full.json",
                    help="affinity half")
    ap.add_argument("--train-lmdb",
                    default=f"{B}/data/train_no_test_af/train.lmdb",
                    help="structure half")
    ap.add_argument("--eval-dir", default=f"{B}/data/t3/eval")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--cache", default=f"{B}/data/t3/inchikeys.json",
                    help="SMILES -> InChIKey for every molecule seen, reused across runs")
    ap.add_argument("--subset", help="restrict to a (layer, uniprot) subset CSV — "
                                     "the 350-quota list is the paper's convention")
    ap.add_argument("--out", default=f"{B}/results/export/T3_exact_overlap.csv")
    args = ap.parse_args()

    recs = {L: [json.loads(x) for x in open(f"{args.eval_dir}/{L}.jsonl")]
            for L in LAYERS}

    # ---------- InChIKeys ----------
    # Keyed by molecule, so the cache is built over the *full* T3 set and both
    # training halves regardless of --subset; a subset run then filters records,
    # never the cache. Same discipline as ligand_novelty.py.
    t_aff = train_smis_affinity(args.train_label)
    t_str = train_smis_structure(args.train_lmdb)
    t3 = {x["smiles"] for rs in recs.values() for r in rs
          for g in ("actives", "decoys") for x in r[g]}
    print(f"affinity half {len(t_aff):,} / structure half {len(t_str):,} / "
          f"T3 unique {len(t3):,}", flush=True)

    need = sorted(t_aff | t_str | t3)
    if os.path.exists(args.cache):
        K = json.load(open(args.cache))
        missing = [s for s in need if s not in K]
        print(f"cache hit {args.cache} ({len(K):,}); {len(missing):,} new")
        if missing:
            K.update(keys_of(missing, args.workers))
            json.dump(K, open(args.cache, "w"))
    else:
        print(f"computing {len(need):,} InChIKeys…", flush=True)
        K = keys_of(need, args.workers)
        os.makedirs(os.path.dirname(args.cache), exist_ok=True)
        json.dump(K, open(args.cache, "w"))
        print(f"wrote {args.cache}")

    A = {K[s] for s in t_aff if K.get(s)}
    S = {K[s] for s in t_str if K.get(s)}
    print(f"distinct InChIKeys — affinity {len(A):,}, structure {len(S):,}")

    if args.subset:
        keep = {(r["layer"], r["uniprot"])
                for r in csv.DictReader(open(args.subset))}
        recs = {L: [r for r in rs if (L, r["uniprot"]) in keep]
                for L, rs in recs.items()}
        print(f"subset: {sum(len(v) for v in recs.values())} (target×layer), "
              f"{len({r['uniprot'] for v in recs.values() for r in v})} unique targets")

    # ---------- table ----------
    rows = [["layer", "group", "n", "n_in_affinity_half", "pct_in_affinity_half",
             "n_in_structure_half", "pct_in_structure_half"]]
    print()
    print("%-4s %-8s %10s %22s %22s" % ("layer", "group", "n",
                                        "in affinity half", "in structure half"))
    print("-" * 70)
    for L in LAYERS:
        for g in ("actives", "decoys"):
            smis = [x["smiles"] for r in recs[L] for x in r[g]]
            n = len(smis)
            if not n:
                continue
            ka = sum(1 for s in smis if K.get(s) and K[s] in A)
            ks = sum(1 for s in smis if K.get(s) and K[s] in S)
            print("%-4s %-8s %10d %10d (%5.1f%%) %10d (%5.2f%%)"
                  % (L, g, n, ka, 100 * ka / n, ks, 100 * ks / n))
            rows.append([L, g, n, ka, f"{100*ka/n:.2f}", ks, f"{100*ks/n:.2f}"])
        print()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
