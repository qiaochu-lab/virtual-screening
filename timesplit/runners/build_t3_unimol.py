"""Assemble T3's evaluation set into the lmdb format the UniMol-family models (DrugCLIP/BindCLIP/LigUnity) consume.

The output directory layout mirrors DEKOIS's convention exactly, so these
three models' existing `test_dekois_target()` code path can be reused
directly, changing only the data path and not the model code:

    data/T3/<layer>/<uniprot>/<uniprot>_lig.lmdb
                             /<uniprot>_pocket.lmdb

Each lig.lmdb record (field-for-field identical to DEKOIS's official format):
    atoms        list[str]    element symbols
    coordinates  list[ndarray] only 1 conformer, matching the official format
    smi          str
    mol          rdkit Mol
    label        int          1=active, 0=decoy

Each pocket.lmdb record:
    pocket, pocket_index, pocket_atoms, pocket_coordinates

Pocket source priority: PDB experimental structure > Boltz-2 predicted
structure. Both are cut with the same residue-level 6 Å logic (see
extract_pocket*.py), and the source is recorded in the manifest for use
in the T5 structure-robustness stratification.
"""
import argparse
import hashlib
import json
import os
import pickle

import lmdb
import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
B = "/data/work/vs-benchmark"


def load_pockets(threshold):
    """PDB source takes priority, Boltz source fills the gaps. Returns uniprot -> (record, source)."""
    out = {}
    for path, src in [(f"{B}/data/t3/pockets/pocket_{threshold:.1f}A.lmdb", "boltz2_pred"),
                      (f"{B}/data/t3/pockets/pdb_pocket_{threshold:.1f}A.lmdb", "pdb_holo")]:
        if not os.path.exists(path):
            continue
        e = lmdb.open(path, subdir=False, readonly=True, lock=False)
        with e.begin() as t:
            for _, v in t.cursor():
                d = pickle.loads(v)
                out[d["pocket"]] = (d, src)          # PDB source is loaded second, so it naturally overrides the Boltz source
        e.close()
    return out


def mol_key(m):
    ik = m.get("inchikey")
    return ik if ik else "NOKEY_" + hashlib.md5(m["smiles"].encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="+", default=["L3", "L4", "L1", "L2"])
    ap.add_argument("--threshold", type=float, default=6.0)
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--force", action="store_true", help="已存在也重建")
    args = ap.parse_args()
    out_root = args.out_root or f"{B}/data/T3_{args.threshold:.0f}A"

    # Idempotent: skip if the data is already built (we run this once manually ahead of time, and the queue calls it again)
    if os.path.exists(f"{out_root}/manifest.json") and not args.force:
        import json as _j
        m = _j.load(open(f"{out_root}/manifest.json"))
        n = sum(v.get("targets", 0) for v in m.values())
        print(f"{out_root} 已存在（{n} 个靶点），跳过。要重建加 --force")
        return

    pockets = load_pockets(args.threshold)
    from collections import Counter
    print(f"口袋库 {len(pockets):,}  {dict(Counter(s for _, s in pockets.values()))}", flush=True)

    conf_env = lmdb.open(f"{B}/data/t3/mols/conformers.lmdb",
                         subdir=False, readonly=True, lock=False)
    with conf_env.begin() as t:
        n_conf = conf_env.stat()["entries"]
    print(f"构象库 {n_conf:,}", flush=True)

    manifest = {}
    for L in args.layers:
        p = f"{B}/data/t3/eval/{L}.jsonl"
        if not os.path.exists(p):
            print(f"[{L}] 评测集不存在，跳过", flush=True)
            continue

        n_ok, skip_pocket, skip_few = 0, 0, 0
        miss_conf_total = 0
        info = {}
        with conf_env.begin() as ct:
            for line in open(p):
                r = json.loads(line)
                up = r["uniprot"]
                if up not in pockets:
                    skip_pocket += 1
                    continue
                prec, psrc = pockets[up]

                mols, labels, miss = [], [], 0
                for kind, lab in (("actives", 1), ("decoys", 0)):
                    for m in r[kind]:
                        raw = ct.get(mol_key(m).encode())
                        if raw is None:
                            miss += 1
                            continue
                        c = pickle.loads(raw)
                        rd = Chem.MolFromSmiles(c["smi"])
                        if rd is None:
                            miss += 1
                            continue
                        mols.append({"atoms": c["atoms"], "coordinates": c["coordinates"],
                                     "smi": c["smi"], "mol": rd, "label": lab})
                        labels.append(lab)
                miss_conf_total += miss
                # Missing conformers shrink both actives and decoys, so the ratio can drift;
                # fewer than 10 actives excludes the target, keeping this consistent with the evaluation set's threshold
                if sum(labels) < 10 or len(labels) - sum(labels) < 10:
                    skip_few += 1
                    continue

                d = f"{out_root}/{L}/{up}"
                os.makedirs(d, exist_ok=True)
                lp = f"{d}/{up}_lig.lmdb"
                pp = f"{d}/{up}_pocket.lmdb"
                for x in (lp, pp):
                    if os.path.exists(x):
                        os.remove(x)

                le = lmdb.open(lp, subdir=False, map_size=1 << 33)
                with le.begin(write=True) as wt:
                    for i, rec in enumerate(mols):
                        wt.put(str(i).encode(), pickle.dumps(rec))
                le.close()

                pe = lmdb.open(pp, subdir=False, map_size=1 << 30)
                with pe.begin(write=True) as wt:
                    wt.put(b"0", pickle.dumps({
                        "pocket": up, "pocket_index": 0,
                        "pocket_atoms": prec["pocket_atoms"],
                        "pocket_coordinates": prec["pocket_coordinates"],
                    }))
                pe.close()

                info[up] = {"n_mols": len(mols), "n_actives": int(sum(labels)),
                            "pocket_source": psrc,
                            "pocket_atoms": len(prec["pocket_atoms"])}
                n_ok += 1

        manifest[L] = {"targets": n_ok, "skipped_no_pocket": skip_pocket,
                       "skipped_too_few": skip_few,
                       "missing_conformers": miss_conf_total, "per_target": info}
        srcs = Counter(v["pocket_source"] for v in info.values())
        print(f"[{L}] 组装 {n_ok:4d} 个靶点  {dict(srcs)}   "
              f"无口袋跳过 {skip_pocket:4d}  数量不足跳过 {skip_few:3d}  "
              f"缺构象分子 {miss_conf_total:,}", flush=True)

    conf_env.close()
    os.makedirs(out_root, exist_ok=True)
    json.dump(manifest, open(f"{out_root}/manifest.json", "w"), indent=1)
    print(f"\n已写入 {out_root}/")


if __name__ == "__main__":
    main()
