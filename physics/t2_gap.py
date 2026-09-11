"""T2's gap: the same models rank within-target rho ~ 0.42-0.55 on CASF targets, but only 0.09-0.26 on T3 L1.

Candidate explanations
  A Restriction of range: T3's actives pass a pAff>=6 gate, which narrows
    the true spread of binding strength and mechanically attenuates the
    correlation coefficient (attenuation due to restriction of range).
  B Ligands per target: CASF has fewer, T3 has more -- affects variance, not
    the mean, so it is not a systematic bias.
  C Label heterogeneity: CASF is PDBbind's curated Kd/Ki, T3 is ChEMBL's mix
    of IC50/Ki/EC50.

This script quantifies A: the within-target pAff spread on both datasets,
and gives an attenuation-corrected estimate.
Correction formula (Thorndike case II):
rho_true ~ rho_obs*(S/s) / sqrt(1 + rho_obs^2*(S^2/s^2 - 1))
"""
import json, os, numpy as np
B = "/data/work/vs"

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
