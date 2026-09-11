"""Prepare Boltz-2 inputs for every ligand across the 16 FEP systems.

Why per-ligand
--------------
T6's existing 929 Boltz-2 predictions have **only one representative ligand
per target**, which can only support "absolute affinity correlation across
targets" (already measured at Spearman +0.404). The FEP benchmark instead
tests **ranking by binding strength within the same target** -- every ligand
has to be scored individually.

Why use the FEP systems
------------------------
There are already three numbers, but on different footings and not strictly
comparable:
  . retrieval models within an FEP congeneric series   rho ~ 0.4
  . Boltz-2 across targets                             rho = +0.404
  . FEP+ physics method (literature)                   r ~ 0.6-0.8
Running Boltz-2 on the same batch of 16 systems / 461 ligands puts all three
method families on a genuinely aligned comparison.

Warning: known limitation
--------------------------
Boltz-2's affinity module does not support ligands with >128 atoms; the FEP
set is all drug-like small molecules so this should not be an issue, but any
skipped cases are still checked and recorded.
"""
import json
import os

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

B = "/data/work/vs-benchmark"
FEP = f"{B}/code/LigUnity/test_datasets/FEP"
OUT = f"{B}/boltz_fep"
SHARDS = 3          # use the free GPUs 4/6/7
MAX_ATOMS = 128


def main():
    labels = json.load(open(f"{FEP}/fep_labels.json"))
    # cmet(1390aa) and tyk2(1187aa) exceed Boltz-2's 1170 limit,
    # replaced with the kinase-domain-truncated sequence (covers 3/3 binding sites, see fep_truncate.py)
    trunc = {}
    tp = f"{B}/data/t3/fep_truncation.json"
    if os.path.exists(tp):
        trunc = {k: v["seq"] for k, v in json.load(open(tp)).items()}
    os.makedirs(OUT, exist_ok=True)
    for i in range(SHARDS):
        os.makedirs(f"{OUT}/shard_{i}", exist_ok=True)

    rows, skipped = [], []
    for e in labels:
        pocket = e["pockets"][0]
        seq = trunc.get(e["uniprot"], e["sequence"])
        for j, lig in enumerate(e["ligands"]):
            smi = lig["smi"]
            m = Chem.MolFromSmiles(smi)
            if m is None:
                skipped.append((pocket, j, "SMILES 解析失败")); continue
            if m.GetNumAtoms() > MAX_ATOMS:
                skipped.append((pocket, j, f"{m.GetNumAtoms()} 原子超限")); continue
            rows.append({"pocket": pocket, "uniprot": e["uniprot"], "idx": j,
                         "smi": smi, "act": lig["act"], "seq": seq})

    # round-robin shard by sequence length, to balance load across GPUs (Boltz-2 runtime rises steeply with length)
    rows.sort(key=lambda r: -len(r["seq"]))
    manifest = []
    for n, r in enumerate(rows):
        name = f"{r['pocket']}__{r['idx']:03d}"
        y = ("version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: %s\n"
             "  - ligand:\n      id: B\n      smiles: '%s'\n"
             "properties:\n  - affinity:\n      binder: B\n" % (r["seq"], r["smi"]))
        open(f"{OUT}/shard_{n % SHARDS}/{name}.yaml", "w").write(y)
        manifest.append({"name": name, "pocket": r["pocket"], "uniprot": r["uniprot"],
                         "idx": r["idx"], "act": r["act"], "smi": r["smi"]})

    json.dump({"entries": manifest,
               "skipped": [{"pocket": p, "idx": i, "why": w} for p, i, w in skipped]},
              open(f"{B}/data/t3/boltz_fep_manifest.json", "w"), indent=1)

    from collections import Counter
    print(f"生成输入 {len(rows):,} 个（{len(labels)} 个体系）")
    print(f"跳过 {len(skipped)}: {Counter(w for _, _, w in skipped).most_common(3)}")
    ls = sorted(len(r["seq"]) for r in rows)
    print(f"序列长度 中位 {ls[len(ls)//2]}  最长 {ls[-1]}")
    for i in range(SHARDS):
        print(f"  shard_{i}: {len(os.listdir(f'{OUT}/shard_{i}')):,}")


if __name__ == "__main__":
    main()
