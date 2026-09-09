"""CASF-2016 的复合物有没有直接落在 LigUnity/HypSeek 的训练集里。

为什么是「直接」而不是「相似」
------------------------------
PocketAffDB 的每条 assay 带一个 `pockets` 列表，元素形如
`2q5sA--2q5s_NZA_A_1.lmdb`，前四位就是 PDB ID。CASF-2016 的 285 个测试复合物
也是 PDB 条目。两者可以精确 join——如果交集非空，那不是「训练集里有相似的」，
而是**同一条结构就在训练集里**，泄漏审计到此为止，不用再做序列/配体相似度。

有个具体的疑点值得查：LigUnity 的模型卡只说过**筛选权重**训练时剔除了测试
蛋白，**排序权重**（T2 的 CASF 用的正是它）有没有剔除 CASF，从没确认过。

三个层级一起给，从严到宽：
  1. PDB ID 精确重合         —— 同一条结构
  2. 配体 InChIKey 重合      —— 同一个分子（可能在别的蛋白上）
  3. UniProt 重合            —— 同一个蛋白（不同结构、不同配体）
"""
import collections
import json
import os
import pickle
import sys

B = "/data/work/vs-benchmark"


def casf_entries():
    """CASF 的 (pdb_ids, uniprot, smiles)。

    casf_label_seq.json 是 285 条的 list，每条形如
    ``{"pockets": ["4eky"], "uniprot": "P00489", "sequence": ..., "ligands": [...]}``。
    PDB ID 在 ``pockets`` 里，不是 key——第一版按 key 取，取到 None，
    重合数假性为 0。
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
    # 逐条目：这条 CASF 复合物的结构是不是整条都在训练集里
    hit_entry = sum(1 for pdbs, _, _ in ce if pdbs and set(pdbs) & train_pdb)
    print(f"   按 285 个条目算：{hit_entry}/{len(ce)} "
          f"({100*hit_entry/len(ce):.1f}%) 的复合物结构在训练集里")
    print(f"③ UniProt 重合：**{len(ov_up)}/{len(cu)}** "
          f"({100*len(ov_up)/max(1,len(cu)):.1f}%)")

    # ② 配体 InChIKey
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
