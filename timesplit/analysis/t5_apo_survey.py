"""T5 的 apo 对照：先查有多少靶点找得到无配体（apo）结构。

要回答什么
----------
T5 已经证明「用预测结构替代实验结构没有显著差异」，但**都是 holo 结构**——
口袋是被配体撑开的构象。真实虚筛拿到的常常是 apo（未结合）构象，
侧链没有为配体让位。这是这类模型在实践中最可能吃亏的地方，也是 T5 原计划里
一直没做的一项。

怎么判 apo
----------
pdb_meta.json 里 `pdb_lig` 给出每个 PDB 条目的非聚合物配体列表。
去掉水、离子、缓冲剂、冷冻保护剂这些结晶添加物之后，
**一个配体都不剩的条目**就是 apo 候选。
只统计「同一个 UniProt 既有 holo（我们已经用过的）又有 apo」的靶点——
只有这样才能做同靶点的配对比较，避免把靶点难度差混进来。
"""
import json
from collections import Counter

B = "/data/work/vs-benchmark"

# 结晶添加物：不算真正的结合配体
JUNK = {
    "HOH", "DOD", "SO4", "PO4", "CL", "NA", "K", "MG", "CA", "ZN", "MN", "FE",
    "FE2", "CU", "NI", "CO", "CD", "HG", "IOD", "BR", "F", "ACT", "EDO", "GOL",
    "PEG", "PGE", "PG4", "MPD", "DMS", "TRS", "MES", "EPE", "IMD", "FMT", "ACY",
    "NO3", "CO3", "BME", "DTT", "TCE", "CIT", "MLI", "OXL", "TAR", "SIN",
    "AZI", "CYN", "SCN", "NH4", "LI", "CS", "RB", "SR", "BA", "PB", "PT", "AU",
    "AG", "GD", "SM", "EU", "YB", "LU", "HO", "TB", "ER",
}


def main():
    meta = json.load(open(f"{B}/data/t3/pdb_meta.json"))
    up2pdb, pdb_lig = meta["up2pdb"], meta["pdb_lig"]
    choice = json.load(open(f"{B}/data/t3/crystal_ligand_choice.json"))
    holo_up = set(choice["choice"])   # 文件顶层是 choice/need_boltz/no_ref_ligand 三段
    print(f"有共晶配体选择的靶点（holo 可用）: {len(holo_up):,}")

    stat = Counter()
    pairs = {}
    for up, pdbs in up2pdb.items():
        if up not in holo_up:
            continue
        apo = []
        for pid in pdbs:
            ligs = [l for l in pdb_lig.get(pid, []) if l["comp_id"].upper() not in JUNK]  # 记录是 dict，不是字符串
            if not ligs:
                apo.append(pid)
        if apo:
            pairs[up] = apo
            stat["有 apo"] += 1
        else:
            stat["只有 holo"] += 1

    print(f"\n{dict(stat)}")
    print(f"可做 apo↔holo 配对比较的靶点: {len(pairs):,}")
    n_apo = sum(len(v) for v in pairs.values())
    print(f"apo 结构条目合计: {n_apo:,}（中位每靶点 "
          f"{sorted(len(v) for v in pairs.values())[len(pairs)//2] if pairs else 0} 个）")

    json.dump(pairs, open(f"{B}/data/t3/apo_candidates.json", "w"), indent=1)
    print(f"\n名单写入 {B}/data/t3/apo_candidates.json")
    print("\n下一步：把 apo 结构叠合到 holo 上，用 holo 配体的坐标在 apo 里划口袋，")
    print("      这样两边口袋位置一致，差别只来自侧链构象——这正是要测的东西。")


if __name__ == "__main__":
    main()
