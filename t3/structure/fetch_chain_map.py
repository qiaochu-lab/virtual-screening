"""补取每个 PDB 条目的「链 → UniProt」映射。

为什么需要：T3 的靶点里有不少属于大型复合物（核糖体、蛋白酶体、呼吸链），
同一个 PDB 条目会被十几个不同 UniProt 同时命中。若提取口袋时用文件里
**所有**链的蛋白原子，配体明明结合在 A 亚基上，却会被当成 B 亚基的口袋。
有了这张映射，就能把口袋限制在该靶点自己的链上，并在配体压根不接触
该靶点时判定这个 PDB 不可用、换下一个候选。
"""
import json
import os
import time
import urllib.request

B = "/data/yicheng/xqc/vs-benchmark"
META = f"{B}/data/t3/pdb_meta.json"
OUT = f"{B}/data/t3/pdb_chain_map.json"

GQL = """
query($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    polymer_entities {
      rcsb_polymer_entity_container_identifiers { auth_asym_ids }
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
    meta = json.load(open(META))
    # 只需要含配体的条目——apo 结构不会被选中
    ids = sorted(p for p, ligs in meta["pdb_lig"].items() if ligs)
    print(f"待查 PDB 条目: {len(ids):,}", flush=True)

    chain_map = {}          # pdb_id -> {uniprot: [chains]}
    prot_chains = {}        # pdb_id -> [所有蛋白链]
    GB = 50
    for i in range(0, len(ids), GB):
        chunk = ids[i:i + GB]
        d = None
        for _ in range(3):
            try:
                d = gql(chunk)
                break
            except Exception:
                time.sleep(4)
        if not d or "data" not in d:
            continue
        for e in (d["data"].get("entries") or []):
            m, prots = {}, []
            for pe in (e.get("polymer_entities") or []):
                ch = ((pe.get("rcsb_polymer_entity_container_identifiers") or {})
                      .get("auth_asym_ids") or [])
                ptype = (pe.get("entity_poly") or {}).get("rcsb_entity_polymer_type")
                if ptype == "Protein":
                    prots.extend(ch)
                for u in (pe.get("uniprots") or []):
                    acc = u.get("rcsb_id")
                    if acc:
                        m.setdefault(acc, []).extend(ch)
            chain_map[e["rcsb_id"]] = m
            prot_chains[e["rcsb_id"]] = sorted(set(prots))
        if (i // GB) % 20 == 0:
            print(f"  {min(i + GB, len(ids)):,}/{len(ids):,}", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"chain_map": chain_map, "protein_chains": prot_chains}, open(OUT, "w"))

    n_multi = sum(1 for m in chain_map.values() if len(m) > 1)
    print(f"\n拿到映射的条目: {len(chain_map):,}")
    print(f"  含多个 UniProt 的（复合物）: {n_multi:,} ({n_multi / max(1, len(chain_map)) * 100:.1f}%)")
    print(f"已写入 {OUT}")


if __name__ == "__main__":
    main()
