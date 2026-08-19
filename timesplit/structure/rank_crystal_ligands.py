"""第 2 步（改版）：为每个 T3 新靶点排出一串候选共晶配体，而不是只选一个。

为什么要改成候选列表
--------------------
上一版只输出「最优的一个 (pdb_id, comp_id)」，提取口袋时才发现两类问题：

  1. 47.1% 的 PDB 条目含多个 UniProt（核糖体、蛋白酶体、呼吸链这类复合物）。
     配体结合在 A 亚基上，却会被当成同一条目里 B 亚基的口袋。
  2. 即便限制到该靶点自己的链，配体也可能压根不接触这条链——
     此时这个 (pdb, ligand) 对该靶点无效，必须换下一个候选。

这两件事都只有拿到坐标才能判定，元数据阶段做不到。所以这里改为输出排序候选，
由 extract_pocket_pdb.py 逐个试，取第一个能截出合格口袋的。

排序键（与上一版相同的三层）
----------------------------
  1. Tanimoto 按 0.1 分桶 —— 保证真正相似的配体优先，同时让 0.02 与 0.00
     这种无意义的差异无法压过体积差
  2. 是否落在类药分子量窗口 [250, 700]
  3. 窗口内取大者；超窗口的分子量封顶，不因更大而占优

黑名单排除：离子、缓冲液/冷冻保护剂、脂类与去污剂、聚糖，
以及重原子数过少的片段（单原子的镧系相位离子等）。
"""
import json
import os
from collections import defaultdict

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

B = "/data/work/vs-benchmark"
META = f"{B}/data/t3/pdb_meta.json"
CHAINMAP = f"{B}/data/t3/pdb_chain_map.json"
OUT = f"{B}/data/t3/crystal_ligand_candidates.json"

TOP_N = 8          # 每个靶点最多保留的候选数

# 离子 / 缓冲液 / 冷冻保护剂 / 结晶添加剂
BLOCK = {
    "HOH", "DOD", "SO4", "PO4", "CL", "NA", "MG", "ZN", "CA", "K", "MN", "FE", "FE2",
    "CU", "CU1", "NI", "CD", "HG", "IOD", "BR", "F", "ACT", "EDO", "GOL", "PEG", "PG4",
    "PGE", "1PE", "P6G", "MPD", "DMS", "TRS", "EPE", "MES", "BME", "IMD", "FMT", "CIT",
    "TAR", "ACY", "NO3", "AZI", "SCN", "CO3", "NH4", "OXY", "PER", "UNX", "UNL", "MLI",
    "SIN", "BCT", "CAC", "PIN", "HEZ", "12P", "15P", "2PE", "MLA", "MRD", "BU3", "PDO",
    "SRT", "MAE", "FLC", "ARS", "VO4", "WO4", "MOO", "PI", "PPV", "POP",
    # 相位用重原子/镧系
    "PR", "EU", "GD", "SM", "YB", "LU", "TB", "HO", "ER", "DY", "LA", "CE", "ND", "TM",
    "PT", "AU", "PB", "OS", "IR", "TA", "TL", "BA", "SR", "CS", "RB", "AG", "MO", "W",
    # 结晶/结构常见但通常不是药物位点
    "IHP",
}
# 脂类 / 去污剂 / 甾醇 —— 标记的是跨膜面或疏水沟槽，不是药物口袋
LIPID = {
    "CDL", "POV", "PGV", "PGT", "LHG", "PEE", "PEF", "PC1", "PCF", "PSC", "3PE",
    "6PL", "PX4", "PLM", "MYR", "STE", "OLA", "OLC", "PEV", "PIO", "PLC", "DGA",
    "LDA", "C8E", "BOG", "BNG", "SDS", "OCT", "LMT", "LMN", "DDQ", "UND", "D10",
    "D12", "TWT", "C10", "HTG", "HP6", "F09", "JEF", "TRD", "P15", "PE4", "XPE",
    "7PE", "DPO", "ETE", "CLR", "CHD", "Y01", "HC3", "SOG", "LI1", "3PH", "SQD",
}
# 聚糖 —— 翻译后修饰，不是配体位点
GLYCAN = {
    "NAG", "NDG", "BMA", "MAN", "BGC", "GLC", "GAL", "GLA", "FUC", "FUL",
    "XYS", "XYP", "SIA", "NGA", "A2G", "RAM", "GCU", "IDS", "SGN", "MBG",
}
BLOCK |= LIPID | GLYCAN

MIN_MW, MIN_HEAVY = 120.0, 8
MW_LO, MW_HI = 250.0, 700.0


def _key(c):
    return (round(c["tanimoto"], 1),
            1 if MW_LO <= c["mw"] <= MW_HI else 0,
            min(c["mw"], MW_HI))


def fp_and_heavy(smi):
    """返回 (指纹, 重原子数)；无法解析时返回 (None, 0)。"""
    if not smi:
        return None, 0
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None, 0
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048), m.GetNumHeavyAtoms()


def main():
    meta = json.load(open(META))
    up2pdb, pdb_lig = meta["up2pdb"], meta["pdb_lig"]
    cm = json.load(open(CHAINMAP))
    chain_map = cm["chain_map"]

    # T3 里每个靶点的参照配体（取亲和力最高的若干）
    ref_lig = defaultdict(list)
    for L in ["L1", "L2", "L3", "L4"]:   # 已知靶点(L1/L2)也要参照配体
        p = f"{B}/data/t3/layers/{L}.jsonl"
        if not os.path.exists(p):
            continue
        for line in open(p):
            d = json.loads(line)
            try:
                ref_lig[d["uniprot"]].append((float(d["paff"]), d["smiles"]))
            except (TypeError, ValueError):
                pass
    for u in ref_lig:
        ref_lig[u] = sorted(ref_lig[u], reverse=True)[:20]
    print(f"T3 参照配体覆盖靶点: {len(ref_lig):,}", flush=True)

    fp_cache = {}
    cands_all, need_boltz = {}, []
    n_no_chain = 0
    for n, (up, pdbs) in enumerate(sorted(up2pdb.items())):
        refs = []
        for _, smi in ref_lig.get(up, []):
            f, _ = fp_and_heavy(smi)
            if f is not None:
                refs.append(f)

        cands = []
        for pid in pdbs:
            # 该靶点在这个条目里占哪些链；拿不到映射就不排除（宁可后面按坐标判定）
            own = sorted(set(chain_map.get(pid, {}).get(up, [])))
            for lg in pdb_lig.get(pid, []):
                cid = lg.get("comp_id")
                mw = lg.get("mw")
                if cid in BLOCK or mw is None or float(mw) < MIN_MW:
                    continue
                smi = lg.get("smiles")
                if smi not in fp_cache:
                    fp_cache[smi] = fp_and_heavy(smi)
                f, heavy = fp_cache[smi]
                if f is None or heavy < MIN_HEAVY:
                    continue
                sim = max((DataStructs.TanimotoSimilarity(f, r) for r in refs), default=0.0)
                cands.append({"pdb_id": pid, "comp_id": cid, "smiles": smi,
                              "mw": float(mw), "tanimoto": round(sim, 4),
                              "target_chains": own,
                              "ligand_chains": lg.get("chains") or []})

        if not cands:
            need_boltz.append(up)
        else:
            cands.sort(key=_key, reverse=True)
            # 同一个 (pdb, comp) 只留一次
            seen, uniq = set(), []
            for c in cands:
                k = (c["pdb_id"], c["comp_id"])
                if k in seen:
                    continue
                seen.add(k)
                uniq.append(c)
                if len(uniq) >= TOP_N:
                    break
            cands_all[up] = uniq
            if not any(c["target_chains"] for c in uniq):
                n_no_chain += 1
        if (n + 1) % 200 == 0:
            print(f"  ...{n+1}/{len(up2pdb):,}", flush=True)

    json.dump({"candidates": cands_all, "need_boltz": sorted(need_boltz)},
              open(OUT, "w"), indent=1)

    print()
    print("=" * 64)
    print(f"有候选配体的靶点   : {len(cands_all):,}")
    print(f"无候选需回退 Boltz : {len(need_boltz):,}")
    print(f"  其中所有候选都拿不到链归属: {n_no_chain:,}（提取时按坐标兜底）")
    print("=" * 64)
    nc = np.array([len(v) for v in cands_all.values()])
    print(f"\n每靶点候选数: 中位 {int(np.median(nc))}  仅1个 {(nc == 1).sum():,}  满 {TOP_N} 个 {(nc == TOP_N).sum():,}")
    top = np.array([v[0]["tanimoto"] for v in cands_all.values()])
    print(f"首选候选的 Tanimoto: 中位 {np.median(top):.3f}  ≥0.8 的 {(top >= 0.8).sum():,}")
    print(f"\n已写入 {OUT}")


if __name__ == "__main__":
    main()
