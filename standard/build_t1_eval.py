"""Convert the three standard benchmarks into the same format as the T3
eval set, so the other three models can run T1 directly.

Why convert
----------
ConGLUDe / ConPLex / SPRINT don't consume UniMol's pocket lmdb format --
they need sequences, `.pdb` structures, and SaProt 3Di sequences.
But we already have three runners that can run T3, and the T3 eval-set
format is
    {"uniprot": ..., "actives": [{"smiles":...}], "decoys": [{"smiles":...}]}
so writing DUD-E / DEKOIS / LIT-PCBA into the same format lets the three
runners run T1 just by swapping the `--eval` path, instead of writing a
separate one for each.

Where target identity comes from
--------------
dude.json / dekois.json / PCBA.json in test_datasets are
[UniProt, PDB, name] triples -- 102 / 81 / 15 of them -- which give exactly:
  · UniProt -> sequence (used by ConPLex)
  · PDB     -> structure (ConGLUDe uses the .pdb; SPRINT uses it to build 3Di)
Every DUD-E target directory already ships a receptor.pdb, saving a download.

⚠️ A single LIT-PCBA target can have up to 360k molecules; running
ConPLex/SPRINT on the full set is expensive. By default we subsample via
--max-decoys (actives are never subsampled) and record the sampling ratio
in the output file -- EF must be computed against the actual ratio and
cannot be compared directly to a full-set number.
"""
import argparse
import json
import os
import pickle
import random

import lmdb

B = "/data/work/vs-benchmark"
TD = f"{B}/code/LigUnity/test_datasets"


def read_lmdb(path):
    """Read in cursor order -- matches the order the model side sees
    (keys are strings, lexicographic)."""
    e = lmdb.open(path, subdir=False, readonly=True, lock=False)
    out = []
    with e.begin() as t:
        for _k, v in t.cursor():
            d = pickle.loads(v)
            smi = d.get("smi")
            if smi is None:
                continue
            # some LIT-PCBA smi strings have an ID appended, separated by a space
            out.append((smi.split()[0], int(d.get("label", 0))))
    e.close()
    return out


def targets(bench):
    j = {"DUDE": "dude.json", "DEKOIS": "dekois.json", "PCBA": "PCBA.json"}[bench]
    return json.load(open(f"{TD}/{j}"))


def lig_path(bench, name):
    if bench == "DUDE":
        return f"{TD}/DUD-E/{name.lower()}/mols.lmdb"
    if bench == "DEKOIS":
        return f"{TD}/DEKOIS_2.0x/{name}/{name}_lig.lmdb"
    return f"{TD}/lit_pcba/{name}/mols.lmdb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["DUDE", "DEKOIS", "PCBA"])
    ap.add_argument("--max-decoys", type=int, default=20000,
                    help="每个靶点最多保留多少 decoy，0=不限")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    out = args.out or f"{B}/data/t1/{args.bench}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    rows, miss, meta = [], [], []
    for up, pdb, name in targets(args.bench):
        # DEKOIS directory names are lowercase; DUD-E uses the lowercase of
        # the third column; LIT-PCBA uses the name as-is
        cands = [name, name.lower()]
        p = None
        for c in cands:
            q = lig_path(args.bench, c)
            if os.path.exists(q):
                p = q
                break
        if p is None:
            miss.append((up, pdb, name, "找不到配体 lmdb"))
            continue
        mols = read_lmdb(p)
        act = [{"smiles": s} for s, y in mols if y == 1]
        dec = [{"smiles": s} for s, y in mols if y == 0]
        if len(act) < 5 or len(dec) < 10:
            miss.append((up, pdb, name, f"active {len(act)} decoy {len(dec)} 太少"))
            continue
        n_dec_all = len(dec)
        if args.max_decoys and len(dec) > args.max_decoys:
            dec = rng.sample(dec, args.max_decoys)
        rows.append({"uniprot": up, "pdb": pdb, "name": name,
                     "n_actives": len(act), "n_decoys": len(dec),
                     "n_decoys_full": n_dec_all,
                     "decoy_sampling": len(dec) / n_dec_all,
                     "actives": act, "decoys": dec})
        meta.append((name, len(act), n_dec_all, len(dec)))

    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"{args.bench}: 写出 {len(rows)} 个靶点 -> {out}")
    sampled = [m for m in meta if m[3] < m[2]]
    print(f"  active 合计 {sum(m[1] for m in meta):,}   "
          f"decoy 合计 {sum(m[3] for m in meta):,}（原始 {sum(m[2] for m in meta):,}）")
    if sampled:
        print(f"  ⚠️ {len(sampled)} 个靶点的 decoy 被抽样，"
              f"最低保留比例 {min(m[3]/m[2] for m in sampled):.1%}"
              "——EF 要按实际比例算，不能直接和全量数字比")
    for u, p, n, why in miss:
        print(f"  跳过 {n}: {why}")


if __name__ == "__main__":
    main()
