"""ChEMBL 37 时间切分（切分点 ≤2024-12）。

口径
----
- 时间依据用 `docs.year`（文献发表年份）。ChEMBL 的 assay 随文献入库，
  发表年份 ≥2025 即为切分点之后的新数据。
- 差集：排除 assay_id 已出现在 LigUnity 训练集（PocketAffDB）中的记录。
- 只保留有 UniProt 映射、有 SMILES、有可用活性值的记录。
- 活性值统一转 pAffinity = -log10(M)，取 Ki/Kd/IC50/EC50（标准单位 nM）。
"""
import json, math, os, sqlite3

DB = "/data/yicheng/xqc/vs-benchmark/data/raw/chembl37/chembl_37/chembl_37_sqlite/chembl_37.db"
TD = "/data/yicheng/xqc/vs-benchmark/code/LigUnity/test_datasets"
OUT = "/data/yicheng/xqc/vs-benchmark/data/t3/chembl37_2025plus.jsonl"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

lab = json.load(open(f"{TD}/train_label_blend_seq_full.json"))
train_assay = {int(a["assay_id"]) for a in lab if str(a["assay_id"]).isdigit()}
train_up = {a["uniprot"] for a in lab if a.get("uniprot")}
print(f"训练集 ChEMBL assay: {len(train_assay):,}   UniProt: {len(train_up):,}", flush=True)

c = sqlite3.connect(DB)
c.execute("PRAGMA temp_store=MEMORY")

SQL = """
SELECT a.assay_id, d.year, cs.accession, ms.canonical_smiles,
       act.standard_type, act.standard_value, act.standard_units, act.standard_relation
FROM activities act
JOIN assays a               ON act.assay_id = a.assay_id
JOIN docs d                 ON a.doc_id = d.doc_id
JOIN target_dictionary td   ON a.tid = td.tid
JOIN target_components tc   ON td.tid = tc.tid
JOIN component_sequences cs ON tc.component_id = cs.component_id
JOIN compound_structures ms ON act.molregno = ms.molregno
WHERE d.year >= 2025
  AND cs.accession IS NOT NULL
  AND act.standard_value IS NOT NULL
  AND act.standard_units = 'nM'
  AND act.standard_type IN ('Ki','Kd','IC50','EC50')
  AND act.standard_relation = '='
"""

n_raw = n_kept = 0
from collections import Counter
by_up = Counter()
new_up = set()
seen_assay_in_train = 0

with open(OUT, "w") as fo:
    for row in c.execute(SQL):
        n_raw += 1
        assay_id, year, up, smi, stype, val, unit, rel = row
        if assay_id in train_assay:
            seen_assay_in_train += 1
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        if v <= 0:
            continue
        paff = -math.log10(v * 1e-9)
        n_kept += 1
        by_up[up] += 1
        if up not in train_up:
            new_up.add(up)
        fo.write(json.dumps({
            "source": "chembl37", "assay_id": assay_id, "year": year,
            "uniprot": up, "smiles": smi, "paff": round(paff, 4),
            "std_type": stype, "in_train_uniprot": up in train_up,
        }) + "\n")
        if n_kept % 20000 == 0:
            print(f"  ...已保留 {n_kept:,}", flush=True)

print()
print(f"发表年份 ≥2025 的活性记录  : {n_raw:,}")
print(f"  其中 assay 已在训练集里   : {seen_assay_in_train:,}   ← 差集扣掉")
print(f"  保留                     : {n_kept:,}")
print(f"涉及 UniProt              : {len(by_up):,}")
print(f"  训练集里没有的（新靶点）   : {len(new_up):,}")
print(f"  训练集里已有的            : {len(by_up)-len(new_up):,}")
print(f"\n输出: {OUT}")
