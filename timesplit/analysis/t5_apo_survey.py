"""T5's apo control: first survey how many targets have an available
ligand-free (apo) structure.

What this is answering
------------
T5 has already shown "substituting predicted structures for experimental
ones makes no significant difference", but **both were holo structures** —
pockets in a conformation opened up by the ligand. Real screening campaigns
often only have an apo (unbound) conformation, where side chains haven't
made room for a ligand. This is where these models are most likely to
suffer in practice, and it's the one item T5's original plan never covered.

How apo is determined
------------
pdb_meta.json's `pdb_lig` gives each PDB entry's list of non-polymer
ligands. After removing crystallisation additives — water, ions, buffers,
cryoprotectants — **an entry with no ligand left at all** is an apo
candidate.
Only targets where the same UniProt has both a holo structure (already used)
and an apo structure are counted — that's the only way to do a same-target
paired comparison without mixing in differences in target difficulty.
"""
import json
from collections import Counter

B = "/data/work/vs-benchmark"

# Crystallisation additives: not counted as genuine bound ligands
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
    holo_up = set(choice["choice"])   # the file's top level has three sections: choice/need_boltz/no_ref_ligand
    print(f"有共晶配体选择的靶点（holo 可用）: {len(holo_up):,}")

    stat = Counter()
    pairs = {}
    for up, pdbs in up2pdb.items():
        if up not in holo_up:
            continue
        apo = []
        for pid in pdbs:
            ligs = [l for l in pdb_lig.get(pid, []) if l["comp_id"].upper() not in JUNK]  # each record is a dict, not a string
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
