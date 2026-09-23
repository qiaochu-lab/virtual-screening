#!/usr/bin/env python3
"""生成 strict 口径的子集清单：L1/L2 全留，L3/L4 只留未被任一训练集见过的。

这样其余分析脚本只要传 --subset 就能得到 strict 版，不必逐个改脚本。
"""
import csv
B = "/data/yicheng/xqc/vs-benchmark/results/export"
exp = {r["uniprot"]: r for r in csv.DictReader(open(f"{B}/T3_target_exposure.csv"))}
rows = list(csv.DictReader(open(f"{B}/T3_vsds_matched.csv")))
hdr = list(rows[0])
out, drop = [], []
for r in rows:
    e = exp.get(r["uniprot"])
    lay = r["layer"]
    # corrected：同源 ≥0.4 的 L4 视作 L3
    if e and lay == "L4" and e["layer_corrected"] == "L3":
        lay = "L3"
    if lay in ("L3", "L4") and e and e["seen_any"] == "1":
        drop.append((r["uniprot"], r["layer"], lay))
        continue
    out.append(r)
p = f"{B}/T3_vsds_matched_strict.csv"
with open(p, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=hdr); w.writeheader(); w.writerows(out)
from collections import Counter
c = Counter(r["layer"] for r in out)
print(f"写出 {p}")
print(f"  保留 {len(out)} 条（原 {len(rows)}），剔除 {len(drop)} 条")
print(f"  按原始 layer 列：", dict(c))
print(f"  剔除的按 corrected 层：", dict(Counter(d[2] for d in drop)))
