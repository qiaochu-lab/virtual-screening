"""Second round of Boltz-2 inputs: long targets after domain truncation, plus first-round failures retried with a different ligand.

Two batches
-----------
A. 123 targets >1170aa, using the truncated sequence from truncate_domains2.py's output.
B. 9 targets that failed in round 1, retried with a different ligand:
   6 had a peptide ligand over 128 atoms (Boltz-2's affinity module hard limit),
   1 had a ligand RDKit couldn't generate a 3D conformer for, 2 had MSA/file
   errors (unaffected by the ligand swap, retried along with the rest)
   -- the pocket only needs one representative ligand, not necessarily the
   highest-affinity one.
"""
import json
import os

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

B = "/data/work/vs-benchmark"
OUT = f"{B}/boltz_r2"
SHARDS = 4
MAX_LIG_ATOMS = 128          # Boltz-2 affinity module's ceiling

os.makedirs(OUT, exist_ok=True)
for i in range(SHARDS):
    os.makedirs(f"{OUT}/shard_{i}", exist_ok=True)

trunc = json.load(open(f"{B}/data/t3/domain_truncation.json"))["truncation"]
seqs = json.load(open(f"{B}/data/t3/sequences.json"))
br = json.load(open(f"{B}/data/t3/missing_breakdown.json"))
retry = set(br["跑了但失败"])

# Collect ligands per target sorted by affinity, high to low, for later selection
ligs = {}
need = set(trunc) | retry
for L in ["L3", "L4"]:
    for line in open(f"{B}/data/t3/layers/{L}.jsonl"):
        d = json.loads(line)
        u = d["uniprot"]
        if u not in need:
            continue
        try:
            ligs.setdefault(u, []).append((float(d["paff"]), d["smiles"]))
        except (TypeError, ValueError):
            pass
for u in ligs:
    ligs[u].sort(reverse=True)


def pick_ligand(u):
    """Pick the highest-affinity ligand that also passes the atom-count and conformer checks."""
    for paff, smi in ligs.get(u, []):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        if m.GetNumAtoms() > MAX_LIG_ATOMS:          # heavy-atom count (before adding H) is already enough to filter out peptides
            continue
        mh = Chem.AddHs(m)
        if mh.GetNumAtoms() > MAX_LIG_ATOMS * 2:
            continue
        return smi
    return None


rows, no_lig = [], []
for u in sorted(trunc):
    smi = pick_ligand(u)
    (rows.append((u, trunc[u]["seq"], smi, "truncated")) if smi else no_lig.append(u))
for u in sorted(retry):
    s = (seqs.get(u) or {}).get("seq")
    smi = pick_ligand(u)
    if s and smi:
        rows.append((u, s, smi, "retry"))
    else:
        no_lig.append(u)

rows.sort(key=lambda r: -len(r[1]))          # longest first, to balance load across the four shards
for n, (u, seq, smi, kind) in enumerate(rows):
    y = ("version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: %s\n"
         "  - ligand:\n      id: B\n      smiles: '%s'\n"
         "properties:\n  - affinity:\n      binder: B\n" % (seq, smi))
    open(f"{OUT}/shard_{n % SHARDS}/{u}.yaml", "w").write(y)

json.dump({"targets": [{"uniprot": u, "kind": k, "seq_len": len(s)} for u, s, _, k in rows],
           "no_usable_ligand": no_lig},
          open(f"{B}/data/t3/boltz_r2_manifest.json", "w"), indent=1)

from collections import Counter
print(f"生成输入 {len(rows)}  （{dict(Counter(k for *_, k in rows))}）")
print(f"无可用配体 {len(no_lig)}: {no_lig[:10]}")
ls = sorted(len(s) for _, s, _, _ in rows)
print(f"序列长度 中位 {ls[len(ls)//2]}  最长 {ls[-1]}")
for i in range(SHARDS):
    print(f"  shard_{i}: {len(os.listdir(f'{OUT}/shard_{i}'))}")
