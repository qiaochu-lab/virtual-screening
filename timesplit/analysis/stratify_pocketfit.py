"""Stratified control: how much of the pocket-family models' L1 advantage
comes from "the pocket being induced by the test ligand itself"?

The problem
----
Well-studied targets (L1/L2) are so heavily characterised that the
co-crystallised ligand in the PDB is often the very active molecule we're
testing (median Tanimoto 0.748, 235 targets >= 0.8); novel targets (L3/L4)
only reach 0.12-0.28. So the L1/L2 pockets are conformations induced
specifically by the test ligand.

For structure/pocket-family models, this means the L1->L4 decay is partly
mixed with a "pocket fit" effect, not purely "target unseen during
training". Sequence-family models (ConPLex) don't use a pocket and aren't
affected by this, making them a natural **negative control**: if this effect
is real, it should show up only in structure models.

Approach
----
Split L1 targets into low/high groups by the Tanimoto between the
co-crystallised ligand and the test active ligand, then compare the same
model's performance across the two groups. Then check whether the
structure-model / sequence-model difference is consistent with the
hypothesis.
"""
import argparse
import json
import pickle

import lmdb
import numpy as np
from scipy import stats

B = "/data/work/vs-benchmark"


def load_sim():
    """uniprot -> Tanimoto between the co-crystallised ligand and this target's test ligand."""
    sim = {}
    e = lmdb.open(f"{B}/data/t3/pockets/pdb_pocket_6.0A.lmdb",
                  subdir=False, readonly=True, lock=False)
    with e.begin() as t:
        for _, v in t.cursor():
            d = pickle.loads(v)
            sim[d["pocket"]] = d.get("ligand_tanimoto_to_t3")
    e.close()
    return sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["conglude", "conplex"])
    ap.add_argument("--cut", type=float, default=0.5)
    ap.add_argument("--metric", default="ef1")
    args = ap.parse_args()

    s = json.load(open(f"{B}/results/t3/summary.json"))
    sim = load_sim()

    print(f"分层依据：L1 靶点的共晶配体 vs 测试配体 Tanimoto，切点 {args.cut}")
    print(f"（序列模型 ConPLex 不用口袋，是本对照的阴性对照）\n")
    print("%-10s %-6s %6s %14s %6s %14s %10s" %
          ("模型", "层", "n低", f"低相似(<{args.cut})", "n高", f"高相似(≥{args.cut})", "p"))
    print("-" * 76)

    summary = {}
    for m in args.models:
        if m not in s:
            continue
        for L in ["L1", "L2"]:
            if L not in s[m]:
                continue
            rows = s[m][L]["per_target"]
            lo = [r for r in rows if sim.get(r["uniprot"]) is not None
                  and sim[r["uniprot"]] < args.cut]
            hi = [r for r in rows if sim.get(r["uniprot"]) is not None
                  and sim[r["uniprot"]] >= args.cut]
            if len(lo) < 8 or len(hi) < 8:
                print("%-10s %-6s  低组或高组样本不足（%d / %d）" % (m, L, len(lo), len(hi)))
                continue
            a = np.array([r[args.metric] for r in lo])
            b = np.array([r[args.metric] for r in hi])
            p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
            print("%-10s %-6s %6d %14s %6d %14s %10.4f%s" %
                  (m, L, len(a), f"{a.mean():.2f}±{a.std(ddof=1)/np.sqrt(len(a)):.2f}",
                   len(b), f"{b.mean():.2f}±{b.std(ddof=1)/np.sqrt(len(b)):.2f}",
                   p, "  *" if p < 0.05 else ""))
            summary[(m, L)] = (a.mean(), b.mean(), p)

    # Key comparison: treat L1's low-similarity group as "L1 with the induced-fit advantage removed", then compare against L4
    print("\n" + "=" * 76)
    print("关键检验：用 L1 的低相似组（口袋没有诱导优势）替代整个 L1，衰减还剩多少")
    print("=" * 76)
    print("%-10s %14s %14s %14s %12s" %
          ("模型", "L1 全体", f"L1 低相似组", "L4", "衰减(低相似→L4)"))
    print("-" * 76)
    for m in args.models:
        if m not in s or "L1" not in s[m] or "L4" not in s[m]:
            continue
        rows = s[m]["L1"]["per_target"]
        lo = [r[args.metric] for r in rows
              if sim.get(r["uniprot"]) is not None and sim[r["uniprot"]] < args.cut]
        if len(lo) < 8:
            continue
        allL1 = np.mean([r[args.metric] for r in rows])
        l4 = np.mean([r[args.metric] for r in s[m]["L4"]["per_target"]])
        drop_all = (l4 - allL1) / allL1 * 100
        drop_lo = (l4 - np.mean(lo)) / np.mean(lo) * 100
        print("%-10s %14.2f %14.2f %14.2f %11.0f%%  (整体口径 %.0f%%)" %
              (m, allL1, np.mean(lo), l4, drop_lo, drop_all))
    print("\n若结构模型的「低相似组衰减」明显小于「整体衰减」，说明原来的衰减")
    print("确实被口袋契合度放大了；序列模型两者应当接近（它不用口袋）。")


if __name__ == "__main__":
    main()
