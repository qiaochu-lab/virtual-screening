"""Step 2 (revised): rank a list of candidate co-crystallized ligands for each T3 new target, instead of picking just one.

Why switch to a candidate list
-------------------------------
The previous version output only the "single best (pdb_id, comp_id)", and
pocket extraction later revealed two problems:

  1. 47.1% of PDB entries contain multiple UniProt accessions (complexes
     like the ribosome, proteasome, respiratory chain). A ligand bound to
     subunit A would get misattributed as a pocket on subunit B in the
     same entry.
  2. Even restricted to the target's own chain, the ligand may simply not
     contact that chain at all -- in which case this (pdb, ligand) pair is
     invalid for the target and the next candidate must be tried.

Both of these can only be determined once coordinates are available, not
at the metadata stage. So this step now outputs a ranked candidate list,
which extract_pocket_pdb.py tries one by one, taking the first that yields
a valid pocket.

Ranking key (same three tiers as the previous version)
--------------------------------------------------------
  1. Tanimoto binned to 0.1 -- ensures genuinely similar ligands rank
     first, while preventing a meaningless difference like 0.02 vs 0.00
     from being overridden by a size difference
  2. Whether it falls in the drug-like molecular-weight window [250, 700]
  3. Within the window, prefer the larger one; above the window, molecular
     weight is capped so being larger confers no further advantage

Blacklist exclusions: ions, buffers/cryoprotectants, lipids and
detergents, glycans, and fragments with too few heavy atoms (e.g.
single-atom lanthanide phasing ions).
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

TOP_N = 8          # max candidates kept per target

# Ions / buffers / cryoprotectants / crystallization additives
BLOCK = {
    "HOH", "DOD", "SO4", "PO4", "CL", "NA", "MG", "ZN", "CA", "K", "MN", "FE", "FE2",
    "CU", "CU1", "NI", "CD", "HG", "IOD", "BR", "F", "ACT", "EDO", "GOL", "PEG", "PG4",
    "PGE", "1PE", "P6G", "MPD", "DMS", "TRS", "EPE", "MES", "BME", "IMD", "FMT", "CIT",
    "TAR", "ACY", "NO3", "AZI", "SCN", "CO3", "NH4", "OXY", "PER", "UNX", "UNL", "MLI",
    "SIN", "BCT", "CAC", "PIN", "HEZ", "12P", "15P", "2PE", "MLA", "MRD", "BU3", "PDO",
    "SRT", "MAE", "FLC", "ARS", "VO4", "WO4", "MOO", "PI", "PPV", "POP",
    # Heavy-atom/lanthanide phasing agents
    "PR", "EU", "GD", "SM", "YB", "LU", "TB", "HO", "ER", "DY", "LA", "CE", "ND", "TM",
    "PT", "AU", "PB", "OS", "IR", "TA", "TL", "BA", "SR", "CS", "RB", "AG", "MO", "W",
    # Common in crystallography/structures but usually not a drug site
    "IHP",
}
# Lipids / detergents / sterols -- mark a transmembrane face or hydrophobic groove, not a drug pocket
LIPID = {
    "CDL", "POV", "PGV", "PGT", "LHG", "PEE", "PEF", "PC1", "PCF", "PSC", "3PE",
    "6PL", "PX4", "PLM", "MYR", "STE", "OLA", "OLC", "PEV", "PIO", "PLC", "DGA",
    "LDA", "C8E", "BOG", "BNG", "SDS", "OCT", "LMT", "LMN", "DDQ", "UND", "D10",
    "D12", "TWT", "C10", "HTG", "HP6", "F09", "JEF", "TRD", "P15", "PE4", "XPE",
    "7PE", "DPO", "ETE", "CLR", "CHD", "Y01", "HC3", "SOG", "LI1", "3PH", "SQD",
}
# Glycans -- post-translational modifications, not a ligand site
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
    """Return (fingerprint, heavy-atom count); return (None, 0) when it can't be parsed."""
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

    # Reference ligands for each target in T3 (the top several by affinity)
    ref_lig = defaultdict(list)
    for L in ["L1", "L2", "L3", "L4"]:   # known targets (L1/L2) also need reference ligands
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
            # Which chains this target occupies in this entry; if the mapping is unavailable, don't exclude it (better to decide by coordinates later)
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
            # Keep only one entry per (pdb, comp)
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
