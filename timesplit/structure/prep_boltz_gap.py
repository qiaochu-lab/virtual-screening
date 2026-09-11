"""Generate Boltz-2 inputs for targets that "have a PDB entry but no usable pocket could be cut".

Where this batch of targets comes from: they do have a structure in the
PDB, but none of the candidate co-crystallized ligands contact the
target's own chain -- mostly because they are subunits of large complexes
(ribosome, proteasome, respiratory chain) where the ligand is bound to a
different subunit. In this situation the PDB structure cannot supply a
pocket for this target, so a prediction is the only option.

Follows the same convention as prep_boltz.py: take the highest-affinity
ligand as the representative for each target, and list sequences
>1170aa separately (the empirically-tested ceiling for Boltz-2 on our
machine, and mostly multimeric proteins that, per the hivpr lesson,
should first be truncated by domain).
"""
import json
import os

B = "/data/work/vs-benchmark"
OUT = f"{B}/boltz_gap"
LIMIT = 1170
SHARDS = 4                      # use at most 4 GPUs

os.makedirs(OUT, exist_ok=True)

need = set(json.load(open(f"{B}/data/t3/need_boltz2.json")))
seqs = json.load(open(f"{B}/data/t3/sequences.json"))
print(f"待补靶点: {len(need):,}")

best = {}
for L in ["L3", "L4"]:
    for line in open(f"{B}/data/t3/layers/{L}.jsonl"):
        d = json.loads(line)
        u = d["uniprot"]
        if u not in need:
            continue
        try:
            paff = float(d["paff"])
        except (TypeError, ValueError):
            continue
        if u not in best or paff > best[u][0]:
            best[u] = (paff, d["smiles"])

ok, too_long, no_seq, no_lig = [], [], [], []
for u in sorted(need):
    s = (seqs.get(u) or {}).get("seq")
    if not s:
        no_seq.append(u)
        continue
    if u not in best:
        no_lig.append(u)
        continue
    if len(s) > LIMIT:
        too_long.append((u, len(s)))
        continue
    ok.append((u, s, best[u][1]))

# Round-robin shard by sequence length to balance load across GPUs (Boltz-2 runtime grows steeply with length)
ok.sort(key=lambda x: -len(x[1]))
for i in range(SHARDS):
    os.makedirs(f"{OUT}/shard_{i}", exist_ok=True)
for n, (u, seq, smi) in enumerate(ok):
    y = ("version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: %s\n"
         "  - ligand:\n      id: B\n      smiles: '%s'\n"
         "properties:\n  - affinity:\n      binder: B\n" % (seq, smi))
    open(f"{OUT}/shard_{n % SHARDS}/{u}.yaml", "w").write(y)

json.dump({"ok": [u for u, _, _ in ok],
           "too_long": [{"uniprot": u, "length": l} for u, l in too_long],
           "no_seq": no_seq, "no_ligand": no_lig},
          open(f"{B}/data/t3/boltz_gap_manifest.json", "w"), indent=1)

print(f"  生成输入   : {len(ok):,}  （分 {SHARDS} 片）")
print(f"  序列过长   : {len(too_long):,}  (>{LIMIT}aa，需先按结构域截取)")
print(f"  无序列     : {len(no_seq):,}")
print(f"  无代表配体 : {len(no_lig):,}")
for i in range(SHARDS):
    print(f"    shard_{i}: {len(os.listdir(f'{OUT}/shard_{i}')):,}")
