"""第 1 步：为 972 个有 PDB 结构的 T3 新靶点收集元数据。

产出 uniprot -> [ {pdb_id, ligands:[{comp_id, smiles, mw, chains}]} ]，
供第 2 步按「与 T3 配体的相似度」挑选共晶配体。
"""
import json, os, time, urllib.parse, urllib.request

B = "/data/yicheng/xqc/vs-benchmark"
OUT = f"{B}/data/t3/pdb_meta.json"

cov = json.load(open(f"{B}/data/t3/new_target_structure_coverage.json"))
targets = sorted(cov["have_pdb"].keys())
print(f"待处理靶点: {len(targets):,}", flush=True)

# ---------- 1. UniProt -> PDB IDs ----------
up2pdb = {}
BATCH = 80
for i in range(0, len(targets), BATCH):
    chunk = targets[i:i + BATCH]
    q = " OR ".join("accession:%s" % a for a in chunk)
    url = ("https://rest.uniprot.org/uniprotkb/stream?format=tsv&fields=accession,xref_pdb&query=%s"
           % urllib.parse.quote(q))
    txt = ""
    for _ in range(3):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                txt = r.read().decode()
            break
        except Exception:
            time.sleep(3)
    for ln in txt.split("\n")[1:]:
        p = ln.split("\t")
        if len(p) >= 2 and p[1].strip():
            ids = [x.strip() for x in p[1].split(";") if x.strip()]
            if ids:
                up2pdb[p[0]] = ids
    print(f"  UniProt {min(i+BATCH,len(targets)):,}/{len(targets):,}", flush=True)

all_pdb = sorted({p for v in up2pdb.values() for p in v})
print(f"\n涉及 PDB 条目: {len(all_pdb):,}", flush=True)

# ---------- 2. PDB -> 配体 ----------
GQL = """
query($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    nonpolymer_entities {
      nonpolymer_comp {
        chem_comp { id formula_weight }
        rcsb_chem_comp_descriptor { SMILES }
      }
      rcsb_nonpolymer_entity_container_identifiers { auth_asym_ids }
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

pdb_lig = {}
GB = 50
for i in range(0, len(all_pdb), GB):
    chunk = all_pdb[i:i + GB]
    d = None
    for _ in range(3):
        try:
            d = gql(chunk); break
        except Exception:
            time.sleep(4)
    if not d or "data" not in d:
        continue
    for e in (d["data"].get("entries") or []):
        ligs = []
        for ne in (e.get("nonpolymer_entities") or []):
            c = ne["nonpolymer_comp"]["chem_comp"]
            smi = (ne["nonpolymer_comp"].get("rcsb_chem_comp_descriptor") or {}).get("SMILES")
            ch = ne["rcsb_nonpolymer_entity_container_identifiers"].get("auth_asym_ids") or []
            ligs.append({"comp_id": c["id"], "smiles": smi,
                         "mw": c.get("formula_weight"), "chains": ch})
        pdb_lig[e["rcsb_id"]] = ligs
    if (i // GB) % 10 == 0:
        print(f"  RCSB {min(i+GB,len(all_pdb)):,}/{len(all_pdb):,}", flush=True)

print(f"\n拿到配体信息的 PDB: {len(pdb_lig):,}", flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump({"up2pdb": up2pdb, "pdb_lig": pdb_lig}, open(OUT, "w"))

n_with_lig = sum(1 for v in pdb_lig.values() if v)
print(f"  其中含非聚合物配体的: {n_with_lig:,}")
cover = sum(1 for u in targets if any(pdb_lig.get(p) for p in up2pdb.get(u, [])))
print(f"  至少有一个含配体 PDB 的靶点: {cover:,}/{len(targets):,}")
print(f"\n已写入 {OUT}")
