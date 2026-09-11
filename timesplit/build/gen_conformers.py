"""Generate 3D conformers for every T3 molecule, for the UniMol-family models (DrugCLIP/BindCLIP/LigUnity) to use.

These three models' molecule tower is UniMol, which consumes 3D
coordinates rather than SMILES, so all 146,776 unique molecules in T3
need a conformer computed first. This step is independent of layer and
of model, so it's computed once and reused globally.

Conformer generation follows UniMol's usual preprocessing recipe:
    ETKDGv3 embedding -> MMFF94 optimization (falling back to UFF on
    failure, and to the unoptimized embedded conformer if that also
    fails)
Only 1 conformer is kept per molecule: DrugCLIP's official DUD-E/DEKOIS
data also has just 1 per molecule (the `coordinates` list has length 1),
so this stays consistent with that.

Produces `data/t3/mols/conformers.lmdb`, key = InChIKey,
value = {atoms, coordinates, smi}. Downstream, assembling each target's
`{target}_lig.lmdb` reads straight from this and adds the label.
"""
import argparse
import hashlib
import json
import os
import pickle
from multiprocessing import Pool

import lmdb
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
B = "/data/work/vs-benchmark"


def embed(args):
    ik, smi = args
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return ik, None, "SMILES 解析失败"
        m = Chem.AddHs(m)
        ps = AllChem.ETKDGv3()
        ps.randomSeed = 42                 # fixed seed, for reproducibility
        ps.useSmallRingTorsions = True
        if AllChem.EmbedMolecule(m, ps) != 0:
            ps.useRandomCoords = True      # the usual fallback for macrocycles/flexible molecules
            if AllChem.EmbedMolecule(m, ps) != 0:
                return ik, None, "嵌入失败"
        try:
            if AllChem.MMFFHasAllMoleculeParams(m):
                AllChem.MMFFOptimizeMolecule(m, maxIters=500)
            else:
                AllChem.UFFOptimizeMolecule(m, maxIters=500)
        except Exception:
            pass                           # if optimization fails, use the unoptimized embedded conformer
        m = Chem.RemoveHs(m)
        conf = m.GetConformer()
        coords = np.array([list(conf.GetAtomPosition(i))
                           for i in range(m.GetNumAtoms())], dtype=np.float32)
        atoms = [a.GetSymbol() for a in m.GetAtoms()]
        if len(atoms) == 0 or not np.isfinite(coords).all():
            return ik, None, "坐标异常"
        return ik, {"atoms": atoms, "coordinates": [coords], "smi": smi}, None
    except Exception as e:                 # noqa: BLE001 tolerate failures per-item, don't abort the whole batch
        return ik, None, f"{type(e).__name__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--out", default=f"{B}/data/t3/mols/conformers.lmdb")
    ap.add_argument("--chunk", type=int, default=200)
    args = ap.parse_args()

    # Aggregate the global set of unique molecules
    seen = {}
    for L in ["L1", "L2", "L3", "L4"]:
        p = f"{B}/data/t3/layers/{L}.jsonl"
        if not os.path.exists(p):
            continue
        for line in open(p):
            d = json.loads(line)
            ik = d.get("inchikey")
            # A small number of records have an empty InChIKey: SMILES with a
            # `->` dative bond or a `C:C` notation that RDKit can't parse
            # never generated one at build time. An empty key makes LMDB
            # raise MDB_BAD_VALSIZE outright, so a stable key is derived from
            # the SMILES as a fallback here.
            if not ik:
                ik = "NOKEY_" + hashlib.md5(d["smiles"].encode()).hexdigest()
            seen.setdefault(ik, d["smiles"])
    items = sorted(seen.items())
    print(f"唯一分子: {len(items):,}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if os.path.exists(args.out):
        os.remove(args.out)
    env = lmdb.open(args.out, subdir=False, map_size=1 << 40)

    from collections import Counter
    fails = Counter()
    n_ok = 0
    # Commit in batches: if one giant transaction errors midway, everything computed so far is lost (this is exactly what bit us the first time)
    txn = env.begin(write=True)
    with Pool(args.procs) as pool:
        for i, (ik, rec, err) in enumerate(
                pool.imap_unordered(embed, items, chunksize=args.chunk)):
            if rec is None:
                fails[err] += 1
            else:
                txn.put(ik.encode(), pickle.dumps(rec))
                n_ok += 1
            if (i + 1) % 10000 == 0:
                txn.commit()
                txn = env.begin(write=True)
                print(f"  {i+1:,}/{len(items):,}  成功 {n_ok:,}  失败 {sum(fails.values()):,}",
                      flush=True)
    txn.commit()
    env.close()

    print(f"\n成功 {n_ok:,} / {len(items):,}  ({n_ok/len(items)*100:.1f}%)")
    for k, v in fails.most_common():
        print(f"  {k}: {v:,}")
    print(f"已写入 {args.out}")


if __name__ == "__main__":
    main()
