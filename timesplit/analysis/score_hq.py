"""在「高质量结构」子集上重算主表，检验结论是否依赖低置信预测结构。

高质量 = 有 RCSB 实验结构，或 Boltz-2 预测且 complex_plddt≥0.70 且 iptm≥0.60。
"""
import json
import numpy as np
B = "/data/work/vs-benchmark"
s = json.load(open(f"{B}/results/t3/summary.json"))
hq = set(json.load(open(f"{B}/data/t3/target_quality.json"))["high_quality"])

MODELS = ["ligunity_protein_ranking", "ligunity_pocket_ranking", "drugclip",
          "bindclip_randneg", "bindclip_hardneg", "conglude", "conplex"]
print("高质量子集 vs 全体（EF1%）")
print("=" * 84)
print("%-26s %-4s %8s %8s %8s %10s" % ("模型","层","全体n","全体","高质量n","高质量"))
print("-" * 84)
drops = {}
for m in MODELS:
    if m not in s: continue
    row = {}
    for L in ["L1","L2","L3","L4"]:
        if L not in s[m]: continue
        r = s[m][L]["per_target"]
        a = np.array([x["ef1"] for x in r])
        h = np.array([x["ef1"] for x in r if x["uniprot"] in hq])
        if len(h) < 5: continue
        row[L] = (a.mean(), h.mean())
        print("%-26s %-4s %8d %8.2f %8d %10.2f" % (m, L, len(a), a.mean(), len(h), h.mean()))
    if "L1" in row and "L4" in row:
        drops[m] = ((row["L4"][0]-row["L1"][0])/row["L1"][0]*100,
                    (row["L4"][1]-row["L1"][1])/row["L1"][1]*100)

print("\n" + "=" * 84)
print("L1→L4 衰减：全体 vs 高质量子集")
print("=" * 84)
print("%-26s %14s %14s %10s" % ("模型","全体","高质量","差异"))
print("-" * 68)
for m,(a,h) in drops.items():
    print("%-26s %13.1f%% %13.1f%% %9.1f pt" % (m, a, h, h-a))
print("\n若两列接近 → 结论不依赖低置信预测结构，稳健")
