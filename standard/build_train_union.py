"""Build the union of all evaluated models' training sets, for T3's set
difference and re-stratification.

Why only two sets
--------------
The seven pocket-family models actually use only two training sets:
  A  train_no_test_af   ->  DrugCLIP, BindCLIP (both weight variants)
  B  PocketAffDB        ->  LigUnity (both variants), LiTENCLIP
     (LiTENCLIP's test_datasets/ is entirely symlinks into LigUnity's, so
     they share one copy)

So the union = A ∪ B, and there is no need to collect per model.
The lists for ConPLex (BindingDB) / ConGLUDe / SPRINT (MERGED) are not
available yet and will be stated as a limitation.

Two outputs
------------
1. union_uniprots.json  -- the targets covered by the union, used for
                            **re-stratification** (the current L1/L3/L4
                            labels are decided against B alone, which is
                            unfair to models trained on A)
2. union_pairs.json     -- the (UniProt, InChIKey) pairs in the union, used
                            for **content-level decontamination** (the time
                            split alone cannot catch a record that "existed
                            in 2023 and was only re-measured/re-entered in
                            2025" -- measured at 20.9% of L1)
"""
import json
import os
import pickle

import lmdb
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
B = "/data/work/vs-benchmark"


def set_b_pairs():
    """PocketAffDB (LigUnity / LiTENCLIP). The label is UniProt directly --
    use it as-is."""
    lab = json.load(open(f"{B}/data/raw/figshare/train_label_blend_seq_full.json"))
    ups, pairs = set(), set()
    for a in lab:
        up = a.get("uniprot")
        if up:
            ups.add(up)
        for l in a.get("ligands", []):
            smi = l.get("smi") if isinstance(l, dict) else None
            if not smi or not up:
                continue
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            try:
                pairs.add((up, Chem.MolToInchiKey(m)))
            except Exception:
                pass
    print(f"  B (PocketAffDB): {len(ups):,} UniProt, {len(pairs):,} pair")
    return ups, pairs


def set_a_pairs():
    """train_no_test_af (DrugCLIP / BindCLIP). Stored by PDB, needs mapping
    to UniProt."""
    pdb2up = json.load(open(f"{B}/data/t3/drugclip_pdb2uniprot.json"))
    e = lmdb.open(f"{B}/data/train_no_test_af/train.lmdb",
                  subdir=False, readonly=True, lock=False)
    ups, pairs, no_map = set(), set(), 0
    with e.begin() as t:
        for i, (_, v) in enumerate(t.cursor()):
            d = pickle.loads(v)
            smi, pk = d.get("smi"), d.get("pocket")
            if not smi or not pk:
                continue
            key = str(pk).split("_")[0].upper()[:4]
            us = pdb2up.get(key)
            if not us:
                no_map += 1
                continue
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            try:
                ik = Chem.MolToInchiKey(m)
            except Exception:
                continue
            for u in us:
                ups.add(u)
                pairs.add((u, ik))
            if (i + 1) % 20000 == 0:
                print(f"    ...{i+1:,}  pair {len(pairs):,}", flush=True)
    e.close()
    print(f"  A (train_no_test_af): {len(ups):,} UniProt, {len(pairs):,} pair"
          f"（{no_map:,} 条无 PDB→UniProt 映射）")
    return ups, pairs


def main():
    print("收集两套训练集：")
    ub, pb = set_b_pairs()
    ua, pa = set_a_pairs()

    union_up = ua | ub
    union_pair = pa | pb
    print(f"\n并集: {len(union_up):,} UniProt, {len(union_pair):,} pair")
    print(f"  A∩B UniProt: {len(ua & ub):,}   A∩B pair: {len(pa & pb):,}")

    json.dump(sorted(union_up), open(f"{B}/data/t3/union_uniprots.json", "w"))
    json.dump([list(x) for x in sorted(union_pair)],
              open(f"{B}/data/t3/union_pairs.json", "w"))
    print(f"\n已写入 union_uniprots.json / union_pairs.json")


if __name__ == "__main__":
    main()
