"""Pack the T3 time-split dataset into a publishable form.

The raw eval jsonl is 955 MB, because cross-target decoys are drawn from a
shared pool, so the same molecule's SMILES + InChIKey get stored once for
each of hundreds of targets. This splits it into three normalized tables:

  molecules.csv.gz   global unique-molecule table: mol_id, smiles, inchikey
  targets.csv.gz     target table: uniprot, layer, n_actives, n_decoys, ratio
  actives.csv.gz     uniprot, layer, mol_id, paff
  decoys.csv.gz      uniprot, layer, mol_id

3D conformers are not packed (17 GB, and can be regenerated from SMILES with
RDKit). Pockets are packed separately.
"""
import csv, gzip, json, os, sys

B = "/data/work/vs"
OUT = sys.argv[1] if len(sys.argv) > 1 else f"{B}/data/release"
os.makedirs(OUT, exist_ok=True)
LAYERS = ("L1", "L2", "L3", "L4")

mol_id = {}          # inchikey -> id
mol_rows = []        # id, smiles, inchikey
tgt_rows = []
act_rows = []
dec_rows = []


def intern(smi, ikey):
    """Store each molecule only once. Molecules without an inchikey fall back to using smiles as the key."""
    key = ikey or f"SMI:{smi}"
    i = mol_id.get(key)
    if i is None:
        i = len(mol_id)
        mol_id[key] = i
        mol_rows.append((i, smi, ikey or ""))
    return i


for L in LAYERS:
    p = f"{B}/data/t3/eval/{L}.jsonl"
    if not os.path.exists(p):
        print(f"跳过 {L}（文件不存在）")
        continue
    n = 0
    for line in open(p):
        r = json.loads(line)
        up = r["uniprot"]
        acts = r.get("actives", [])
        decs = r.get("decoys", [])
        for a in acts:
            act_rows.append((up, L, intern(a["smiles"], a.get("inchikey")),
                             a.get("paff", "")))
        for d in decs:
            smi = d["smiles"] if isinstance(d, dict) else d
            ik = d.get("inchikey") if isinstance(d, dict) else None
            dec_rows.append((up, L, intern(smi, ik)))
        tgt_rows.append((up, L, len(acts), len(decs), r.get("ratio", "")))
        n += 1
    print(f"{L}: {n} 个靶点")


def dump(name, header, rows):
    fp = f"{OUT}/{name}.csv.gz"
    with gzip.open(fp, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    mb = os.path.getsize(fp) / 1e6
    print(f"  {name}.csv.gz  {len(rows):>9,} 行  {mb:7.1f} MB")
    return mb


print("\n写出：")
tot = 0
tot += dump("molecules", ["mol_id", "smiles", "inchikey"], mol_rows)
tot += dump("targets", ["uniprot", "layer", "n_actives", "n_decoys", "ratio"], tgt_rows)
tot += dump("actives", ["uniprot", "layer", "mol_id", "paff"], act_rows)
tot += dump("decoys", ["uniprot", "layer", "mol_id"], dec_rows)

man = {
    "name": "T3 time-split virtual screening benchmark",
    "cutoff": "2024-12",
    "sources": ["ChEMBL 37", "BindingDB 202608"],
    "active_threshold": "pAff >= 6",
    "decoys": "cross-target: actives of dissimilar targets, 1:50",
    "n_targets": len(tgt_rows),
    "n_unique_molecules": len(mol_rows),
    "n_active_pairs": len(act_rows),
    "n_decoy_pairs": len(dec_rows),
    "layers": {L: sum(1 for t in tgt_rows if t[1] == L) for L in LAYERS},
    "note": "3D conformers are not shipped; regenerate from SMILES with RDKit "
            "ETKDG. Pockets ship separately as pockets_6A.tar.gz.",
}
json.dump(man, open(f"{OUT}/manifest.json", "w"), indent=1)

print(f"\n唯一分子 {len(mol_rows):,}")
print(f"active 配对 {len(act_rows):,}   decoy 配对 {len(dec_rows):,}")
print(f"合计 {tot:.1f} MB  ->  {OUT}")
