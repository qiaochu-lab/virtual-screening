"""从 BindingDB 抽取时间切分后的候选数据（切分点 ≤2024-12，代哥 2026-08-12 确认）。

口径
----
- 用 `Date in BindingDB`（入库日期）而非 `Date of publication`：
  模型能看到某条数据的前提是它已入库。LigUnity 用的是 BindingDB v2024m5，
  故入库日期 ≥2025-01 的记录对所有被评测模型都是「未来数据」。
- 同时记录发表日期，供后续按 PPT slide 12 做更严格的筛选。
- 亲和力取 Ki 优先，其次 IC50，转为 pAffinity = -log10(M)。
"""
import csv, json, math, sys
from collections import defaultdict

SRC = "/data/yicheng/xqc/vs-benchmark/data/raw/bindingdb/BindingDB_All.tsv"
OUT = "/data/yicheng/xqc/vs-benchmark/data/t3/bindingdb_2025plus.jsonl"
TRAIN = "/data/yicheng/xqc/vs-benchmark/code/LigUnity/test_datasets/train_label_blend_seq_full.json"

train_up = {a["uniprot"] for a in json.load(open(TRAIN)) if a.get("uniprot")}
print(f"LigUnity 训练集 UniProt: {len(train_up):,}", flush=True)

# 列索引（1-based → 0-based）
C_SMILES, C_KI, C_IC50, C_PUB, C_INDB, C_UP = 1, 8, 9, 23, 24, 44

def year(s):
    if not s:
        return None
    p = s.split("/")
    if len(p) == 3 and p[2].isdigit():
        return int(p[2])
    return None

def to_p(v):
    """nM -> pAffinity。去掉 > < 等修饰符。"""
    if not v:
        return None
    v = v.strip().lstrip("><=~ ")
    try:
        x = float(v)
    except ValueError:
        return None
    return -math.log10(x * 1e-9) if x > 0 else None

import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)

n_tot = n_new = n_kept = 0
by_up = defaultdict(int)
new_up = set()
csv.field_size_limit(10**9)
with open(SRC, newline="", encoding="utf-8", errors="replace") as f, open(OUT, "w") as fo:
    r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
    next(r, None)
    for row in r:
        n_tot += 1
        if len(row) <= C_UP:
            continue
        y = year(row[C_INDB])
        if y is None or y < 2025:
            continue
        n_new += 1
        up = row[C_UP].strip()
        smi = row[C_SMILES].strip()
        if not up or not smi:
            continue
        aff = to_p(row[C_KI]) or to_p(row[C_IC50])
        if aff is None:
            continue
        n_kept += 1
        by_up[up] += 1
        if up not in train_up:
            new_up.add(up)
        fo.write(json.dumps({
            "uniprot": up, "smiles": smi, "paff": round(aff, 4),
            "in_train_uniprot": up in train_up,
            "year_indb": y, "year_pub": year(row[C_PUB]),
        }) + "\n")
        if n_kept % 20000 == 0:
            print(f"  ...已保留 {n_kept:,}", flush=True)

print()
print(f"BindingDB 总行数        : {n_tot:,}")
print(f"入库年份 ≥2025          : {n_new:,}")
print(f"有 UniProt+SMILES+活性值 : {n_kept:,}")
print(f"涉及 UniProt            : {len(by_up):,}")
print(f"  其中训练集里没有的     : {len(new_up):,}   ← 全新靶点")
print(f"  训练集里已有的         : {len(by_up)-len(new_up):,}   ← 旧靶点新数据")
print(f"\n输出: {OUT}")
top = sorted(by_up.items(), key=lambda kv: -kv[1])[:8]
print("\n数据最多的 8 个靶点:")
for u, c in top:
    print(f"  {u:10s} {c:6,}  {'(训练集已有)' if u in train_up else '(全新)'}")
