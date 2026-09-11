"""Quantify the impact of the "set difference computed only against
LigUnity's training set" convention.

Problem
----
T3's layer labels (L1 = target seen / L3, L4 = target unseen) are decided
against **LigUnity's training set** (PocketAffDB, 2,196 UniProt). But
DrugCLIP / BindCLIP use a different training set (16,744 PDB pockets).

If a target is unseen by LigUnity but seen by DrugCLIP, we label it L4
(novel target) -- but it is actually familiar to DrugCLIP, so DrugCLIP's L4
decay is **understated**.

This script maps DrugCLIP's training-set PDB entries to UniProt, then
counts how many of our L3/L4 targets are actually in DrugCLIP's training
set.

Note: the time split itself already blocks data leakage (every model was
trained before 2024-12, and none has seen post-2025 affinity data). What is
checked here is **the fairness of the layer labels**, not leakage.
"""
import json
import time
import urllib.request

B = "/data/work/vs-benchmark"

GQL = """
query($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    polymer_entities {
      uniprots { rcsb_id }
      entity_poly { rcsb_entity_polymer_type }
    }
  }
}
"""


def gql(ids):
    body = json.dumps({"query": GQL, "variables": {"ids": ids}}).encode()
    req = urllib.request.Request("https://data.rcsb.org/graphql", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def main():
    pdbs = json.load(open(f"{B}/data/t3/drugclip_train_pdbs.json"))
    print(f"DrugCLIP 训练集 PDB 条目: {len(pdbs):,}", flush=True)

    dc_up = set()
    GB = 50
    for i in range(0, len(pdbs), GB):
        d = None
        for _ in range(3):
            try:
                d = gql(pdbs[i:i + GB])
                break
            except Exception:
                time.sleep(3)
        if not d or "data" not in d:
            continue
        for e in (d["data"].get("entries") or []):
            for pe in (e.get("polymer_entities") or []):
                if (pe.get("entity_poly") or {}).get("rcsb_entity_polymer_type") != "Protein":
                    continue
                for u in (pe.get("uniprots") or []):
                    if u.get("rcsb_id"):
                        dc_up.add(u["rcsb_id"])
        if (i // GB) % 40 == 0:
            print(f"  {min(i + GB, len(pdbs)):,}/{len(pdbs):,}  已得 UniProt {len(dc_up):,}",
                  flush=True)

    json.dump(sorted(dc_up), open(f"{B}/data/t3/drugclip_train_uniprots.json", "w"))
    print(f"\nDrugCLIP 训练集覆盖的 UniProt: {len(dc_up):,}")

    # LigUnity's training targets
    TD = f"{B}/code/LigUnity/test_datasets"
    lab = json.load(open(f"{TD}/train_label_blend_seq_full.json"))
    lig_up = {a["uniprot"] for a in lab if a.get("uniprot")}
    print(f"LigUnity 训练集覆盖的 UniProt: {len(lig_up):,}")
    print(f"两者交集: {len(dc_up & lig_up):,}")
    print(f"并集    : {len(dc_up | lig_up):,}")

    print("\n" + "=" * 74)
    print("影响：我们按 LigUnity 标为「新靶点」的，有多少其实在 DrugCLIP 训练集里")
    print("=" * 74)
    print("%-4s %10s %22s %14s" % ("层", "评测靶点", "在 DrugCLIP 训练集里", "占比"))
    print("-" * 74)
    for L in ["L1", "L2", "L3", "L4"]:
        ups = sorted({json.loads(x)["uniprot"]
                      for x in open(f"{B}/data/t3/eval/{L}.jsonl")})
        n_dc = len([u for u in ups if u in dc_up])
        print("%-4s %10d %22d %13.1f%%" % (L, len(ups), n_dc, n_dc / len(ups) * 100))

    print("\n判读：")
    print("· L3/L4 里「在 DrugCLIP 训练集里」的比例越高，")
    print("  说明按 LigUnity 定的层标签对 DrugCLIP 越不适用，其衰减被低估得越多")
    print("· L1/L2 的比例高是正常的（本来就是已知靶点）")


if __name__ == "__main__":
    main()
