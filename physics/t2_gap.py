"""T2 的空洞：同一批模型 CASF 靶点内排序 ρ≈0.42–0.55，T3 L1 只有 0.09–0.26。

候选解释
  A 取值范围受限：T3 的 active 过了 pAff>=6 的门，真实强弱差距被压窄，
    相关系数被机械衰减（attenuation due to restriction of range）。
  B 每靶点配体数：CASF 少、T3 多 —— 影响方差不影响均值，不构成系统偏差。
  C 标签异质：CASF 是 PDBbind 精选 Kd/Ki，T3 是 ChEMBL 混合 IC50/Ki/EC50。

本脚本量 A：两套数据靶点内 pAff 展布，并给出衰减校正后的估计。
校正公式（Thorndike case II）：ρ_true ≈ ρ_obs·(S/s) / sqrt(1 + ρ_obs²(S²/s² - 1))
"""
import json, os, numpy as np
B = "/data/yicheng/xqc/vs-benchmark"

by = {}
for e in json.load(open(f"{B}/code/LigUnity/test_datasets/casf_label_seq.json")):
    for l in e["ligands"]:
        by.setdefault(e["uniprot"], []).append(float(l["act"]))
casf = [np.array(v) for v in by.values() if len(v) >= 3]

t3 = []
for x in open(f"{B}/data/t3/eval/L1.jsonl"):
    r = json.loads(x)
    v = [float(a["paff"]) for a in r["actives"]]
    if len(v) >= 3:
        t3.append(np.array(v))

def rep(name, arrs):
    sd  = np.array([a.std(ddof=1) for a in arrs])
    rng = np.array([a.max() - a.min() for a in arrs])
    n   = np.array([len(a) for a in arrs])
    print(f"{name}")
    print(f"   靶点数 {len(arrs)}   每靶点配体数中位 {np.median(n):.0f}")
    print(f"   靶点内 pAff 标准差  中位 {np.median(sd):.3f}")
    print(f"   靶点内 pAff 极差    中位 {np.median(rng):.3f}")
    return float(np.median(sd))

print("=" * 64)
S = rep("CASF-2016（跨复合物、同 uniprot 分组）", casf); print()
s = rep("T3 L1（pAff>=6 的 active）", t3)
k = S / s
print(f"\n展布比 S/s = {k:.2f}")
print("\n若差距全部来自范围受限，把 T3 观测值校正回 CASF 的展布应得：")
print("%-26s %10s %12s %10s" % ("模型", "T3 实测ρ", "范围校正后", "CASF 实测"))
print("-" * 62)
casf_obs = {"hypseek_rk": 0.549, "ligunity_pocket_ranking": 0.424}
d = json.load(open(f"{B}/results/t3/summary_t2_v2.json"))
for m, c in casf_obs.items():
    r = d[m]["L1"]["spearman"]
    corr = r * k / np.sqrt(1 + r * r * (k * k - 1))
    print("%-26s %10.3f %12.3f %10.3f" % (m, r, corr, c))
print("-" * 62)
