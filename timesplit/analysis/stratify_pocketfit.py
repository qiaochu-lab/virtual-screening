"""分层对照：口袋类模型的 L1 优势里，有多少来自「口袋按测试配体诱导」？

问题
----
已知靶点（L1/L2）被研究得透，PDB 里的共晶配体常常就是我们要测的活性分子
（中位 Tanimoto 0.748，235 个靶点 ≥0.8）；新靶点（L3/L4）只有 0.12–0.28。
所以 L1/L2 的口袋是按测试配体「量身诱导」出来的构象。

对结构/口袋类模型，L1→L4 的衰减里因此掺了一部分「口袋契合度」差异，
不纯是「靶点没见过」。序列类模型（ConPLex）不用口袋，不受影响，
正好可以当**阴性对照**：如果这个效应是真的，它应该只出现在结构模型身上。

做法
----
把 L1 靶点按共晶配体与测试活性配体的 Tanimoto 切成低/高两组，
比较同一模型在两组上的表现。再看结构模型与序列模型的差异是否一致。
"""
import argparse
import json
import pickle

import lmdb
import numpy as np
from scipy import stats

B = "/data/work/vs-benchmark"


def load_sim():
    """uniprot -> 共晶配体与该靶点测试配体的 Tanimoto。"""
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

    # 关键比较：把 L1 的低相似组当作「去掉诱导优势的 L1」，再与 L4 比
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
