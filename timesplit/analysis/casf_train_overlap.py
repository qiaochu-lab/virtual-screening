"""Do CASF-2016's complexes fall directly inside LigUnity/HypSeek's training
set?

Why "directly" rather than "similar"
--------------------------------------
Each PocketAffDB assay carries a `pockets` list whose entries look like
`2q5sA--2q5s_NZA_A_1.lmdb`, where the first four characters are the PDB ID.
CASF-2016's 285 test complexes are also PDB entries. The two can be joined
exactly — if the intersection is non-empty, that isn't "something similar is
in the training set", it is **the same structure sitting in the training
set**, and the leakage audit stops there; no sequence/ligand similarity is
needed.

One specific concern is worth checking: LigUnity's model card only states
that test proteins were removed when training the **screening weight** — it
has never been confirmed whether the **ranking weight** (the one T2's CASF
run actually uses) excludes CASF.

Reported at three levels together, from strictest to loosest:
  1. Exact PDB ID overlap    -- the same structure
  2. Ligand InChIKey overlap -- the same molecule (possibly on a different
     protein)
  3. UniProt overlap         -- the same protein (different structure,
     different ligand)
"""
import collections
import json
import os
import pickle
import sys

B = "/data/work/vs-benchmark"


def casf_entries():
    """CASF's (pdb_ids, uniprot, smiles).

    casf_label_seq.json is a list of 285 entries, each shaped like
    ``{"pockets": ["4eky"], "uniprot": "P00489", "sequence": ..., "ligands": [...]}``.
    The PDB ID lives inside ``pockets``, not as a key -- the first version
    fetched it as a key, got None back, and the overlap count came out as a
    false zero.
    """
    p = f"{B}/code/LigUnity/test_datasets/casf_label_seq.json"
    out = []
    for v in json.load(open(p)):
        pdbs = [str(x)[:4].lower() for x in v.get("pockets", [])]
        out.append((pdbs, v.get("uniprot"),
                    [x.get("smi") for x in v.get("ligands", [])]))
    return out


def main():
    lab = json.load(open(f"{B}/data/raw/figshare/train_label_blend_seq_full.json"))
    train_pdb, train_up = set(), set()
    for a in lab:
        if a.get("uniprot"):
            train_up.add(a["uniprot"])
        for pk in a.get("pockets", []):
            train_pdb.add(str(pk)[:4].lower())
    print(f"PocketAffDB：{len(lab):,} 条 assay，"
          f"{len(train_pdb):,} 个 PDB ID，{len(train_up):,} 个 UniProt")

    ce = casf_entries()
    print(f"CASF：{len(ce)} 个条目")
    print(f"  样例: {ce[0][0]!r}  uniprot={ce[0][1]!r}  配体数={len(ce[0][2])}")

    cp = {p for pdbs, _, _ in ce for p in pdbs}
    cu = {u for _, u, _ in ce if u}
    print(f"  唯一 PDB ID {len(cp)}，唯一 UniProt {len(cu)}")

    ov_pdb = cp & train_pdb
    ov_up = cu & train_up
    print()
    print(f"① PDB ID 精确重合：**{len(ov_pdb)}/{len(cp)}** "
          f"({100*len(ov_pdb)/max(1,len(cp)):.1f}%)")
    if ov_pdb:
        print("   " + " ".join(sorted(ov_pdb)[:40]))
        if len(ov_pdb) > 40:
            print(f"   …共 {len(ov_pdb)} 个")
    # Per-entry: is this CASF complex's whole structure in the training set
    hit_entry = sum(1 for pdbs, _, _ in ce if pdbs and set(pdbs) & train_pdb)
    print(f"   按 285 个条目算：{hit_entry}/{len(ce)} "
          f"({100*hit_entry/len(ce):.1f}%) 的复合物结构在训练集里")
    print(f"③ UniProt 重合：**{len(ov_up)}/{len(cu)}** "
          f"({100*len(ov_up)/max(1,len(cu)):.1f}%)")

    # (2) Ligand InChIKey
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        tk = set()
        for a in lab:
            for l in a.get("ligands", []):
                m = Chem.MolFromSmiles(l.get("smi", ""))
                if m:
                    tk.add(Chem.MolToInchiKey(m))
        ck = collections.Counter()
        n_c = 0
        for _, _, smis in ce:
            for s in smis:
                m = Chem.MolFromSmiles(s or "")
                if m:
                    n_c += 1
                    if Chem.MolToInchiKey(m) in tk:
                        ck["hit"] += 1
        print(f"② 配体 InChIKey 重合：**{ck['hit']}/{n_c}** "
              f"({100*ck['hit']/max(1,n_c):.1f}%)，训练配体 {len(tk):,} 个唯一")
    except ImportError:
        print("② 配体 InChIKey：环境里没有 rdkit，跳过")

    out = f"{B}/results/export/T2_casf_train_overlap.txt"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(f"casf_pdb_total\t{len(cp)}\n")
        f.write(f"casf_pdb_in_pocketaffdb\t{len(ov_pdb)}\n")
        f.write(f"casf_uniprot_total\t{len(cu)}\n")
        f.write(f"casf_uniprot_in_pocketaffdb\t{len(ov_up)}\n")
        f.write("overlapping_pdb_ids\t" + ",".join(sorted(ov_pdb)) + "\n")
    print(f"\n写出 {out}")


if __name__ == "__main__":
    main()
