"""构建所有参评模型训练集的并集，用于 T3 的差集与重新分层。

为什么只有两套
--------------
七个 pocket 系模型实际只用了两套训练数据：
  A  train_no_test_af   →  DrugCLIP、BindCLIP(两档权重)
  B  PocketAffDB        →  LigUnity(两个变体)、LiTENCLIP
     （LiTENCLIP 的 test_datasets/ 全是指向 LigUnity 的软链，共用同一份）

所以并集 = A ∪ B，不需要逐模型收集。
ConPLex(BindingDB) / ConGLUDe / SPRINT(MERGED) 的清单暂时拿不到，
会在 limitation 里写明。

产出两样东西
------------
1. union_uniprots.json  —— 并集覆盖的靶点，用于**重新分层**
                            （现在的 L1/L3/L4 标签只按 B 判定，对用 A 的模型不公平）
2. union_pairs.json     —— 并集里的 (UniProt, InChIKey) 对，用于**内容去污染**
                            （时间切分挡不住「2023 年就有、2025 年重测/才录入」的记录，
                              实测 L1 有 20.9% 是这种）
"""
import json
import os
import pickle

import lmdb
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
B = "/data/work/vs-benchmark"


def set_b_pairs():
    """PocketAffDB（LigUnity / LiTENCLIP）。标的就是 UniProt，直接用。"""
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
    """train_no_test_af（DrugCLIP / BindCLIP）。按 PDB 存，需映射成 UniProt。"""
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
