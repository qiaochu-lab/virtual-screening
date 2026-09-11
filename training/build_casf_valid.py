"""Split LigUnity's packed casf.lmdb into the three files HypSeek's training
validation set expects.

Why the split is needed
----------
HypSeek's CASF validation branch reads three files — `valid_lig.lmdb` /
`valid_prot.lmdb` / `valid_label_seq.json` — while LigUnity packs CASF into a
single casf.lmdb (each record holds both ligand and pocket fields together).
The two are just different packaging of the same underlying data.

Using CASF as the validation set is required: it is what determines the best
checkpoint by `valid_bedroc`, which is exactly how the "screening weight" _vs
is selected. Using the FEP validation set instead would select the already
public _rk.

When splitting, pocket names are de-duplicated: the pocket lmdb gets one
entry per protein, the ligand lmdb one entry per complex, matching what
HypSeek's load_pockets_dataset / load_mols_dataset expect.
"""
import json
import os
import pickle

import lmdb

B = "/data/work/vs"
SRC = f"{B}/code/LigUnity/test_datasets/casf.lmdb"
OUT = f"{B}/code/LigUnity/test_datasets"


def main():
    e = lmdb.open(SRC, subdir=False, readonly=True, lock=False)
    mols, pockets, seen = [], [], set()
    with e.begin() as t:
        for _k, v in t.cursor():
            d = pickle.loads(v)
            mols.append({k: d[k] for k in ("atoms", "coordinates", "smi", "mol", "label")
                         if k in d} | {"pocket": d["pocket"]})
            if d["pocket"] not in seen:
                seen.add(d["pocket"])
                pockets.append({"pocket": d["pocket"], "pocket_index": d.get("pocket_index", 0),
                                "pocket_atoms": d["pocket_atoms"],
                                "pocket_coordinates": d["pocket_coordinates"]})
    e.close()
    print(f"读到 {len(mols)} 个复合物，{len(pockets)} 个唯一口袋")

    for name, recs in (("valid_lig.lmdb", mols), ("valid_prot.lmdb", pockets)):
        p = f"{OUT}/{name}"
        if os.path.exists(p):
            os.remove(p)
        env = lmdb.open(p, subdir=False, map_size=1 << 34)
        with env.begin(write=True) as w:
            for i, r in enumerate(recs):
                w.put(str(i).encode(), pickle.dumps(r))
        env.close()
        print(f"写入 {p}（{len(recs)} 条）")

    # the label file's content is identical to casf_label_seq.json; symlink it to avoid the two drifting out of sync
    src = f"{OUT}/casf_label_seq.json"
    dst = f"{OUT}/valid_label_seq.json"
    if not os.path.exists(dst):
        os.symlink(src, dst)
    print(f"软链 {dst} -> casf_label_seq.json（内容同一份，避免不同步）")


if __name__ == "__main__":
    main()
