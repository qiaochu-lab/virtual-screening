"""Stratify the 8Å results by "whether the pocket exceeds the 511-atom cap".

Why this is needed
------------
Models cap the number of pocket atoms (511 for DrugCLIP/BindCLIP). When a
pocket exceeds the cap, the code keeps 511 atoms by **random sampling**
weighted by distance to the pocket's geometric centre — meaning the model
sees, for over-cap targets, not the full pocket but a randomly sampled
subset of it.

We measured 8Å performing 39-75% worse than 6Å. But 10.8% of 8Å's pockets
exceed 511 atoms and get truncated, so this degradation might be mixed with
a "truncation artefact", not purely "the pocket is too large".

Here, 8Å targets are split into two groups and compared directly:
  * capped group (>511 atoms, randomly truncated)
  * uncapped group (<=511 atoms, full pocket)

How to read it
----
* only the capped group drops sharply -> 8Å's degradation is mainly caused
  by truncation, and the conclusion needs to be rewritten
* both groups drop -> it's a genuine pocket-scale effect, the conclusion
  holds
* the uncapped group also drops noticeably -> even stronger evidence, since
  this group has no truncation factor at all

Control: 4Å has a 0% cap rate, so its 31-62% degradation contains no
truncation factor to begin with, and can serve as the reference for the
"pure pocket effect".
"""
import json
import os
import pickle

import lmdb
import numpy as np
from scipy import stats

B = "/data/work/vs-benchmark"
CAP = 511


def pocket_sizes(threshold):
    """uniprot -> pocket atom count at this threshold. PDB source takes priority (consistent with how it was assembled)."""
    out = {}
    for pref in ["pocket", "pdb_pocket"]:      # pdb source loaded last, overrides the boltz source
        p = f"{B}/data/t3/pockets/{pref}_{threshold:.1f}A.lmdb"
        if not os.path.exists(p):
            continue
        e = lmdb.open(p, subdir=False, readonly=True, lock=False)
        with e.begin() as t:
            for _, v in t.cursor():
                d = pickle.loads(v)
                out[d["pocket"]] = len(d["pocket_atoms"])
        e.close()
    return out


def main():
    sizes8 = pocket_sizes(8.0)
    s6 = json.load(open(f"{B}/results/t3/summary.json"))
    s8p = f"{B}/results/t3/summary_8a.json"
    if not os.path.exists(s8p):
        print("8Å 结果不存在"); return
    s8 = json.load(open(s8p))

    n_over = sum(1 for v in sizes8.values() if v > CAP)
    print(f"8Å 口袋: {len(sizes8):,} 个，其中 >{CAP} 原子的 {n_over}"
          f"（{n_over/len(sizes8)*100:.1f}%）\n")

    print("=" * 86)
    print("8Å 结果按「是否触顶」分层，并与同靶点的 6Å 对比（EF1%）")
    print("=" * 86)
    print("%-20s %-4s %26s %26s" % ("模型", "层", "未触顶组（完整口袋）", "触顶组（被随机截断）"))
    print("%-20s %-4s %8s %8s %8s %8s %8s %8s" %
          ("", "", "n", "6Å", "8Å", "n", "6Å", "8Å"))
    print("-" * 86)

    agg = {"under": {"6": [], "8": []}, "over": {"6": [], "8": []}}
    for m in ["drugclip", "bindclip_randneg", "bindclip_hardneg"]:
        for L in ["L1", "L2", "L3", "L4"]:
            a = s6.get(m, {}).get(L)
            b = s8.get(m + "_8a", {}).get(L)
            if not a or not b:
                continue
            r6 = {x["uniprot"]: x["ef1"] for x in a["per_target"]}
            r8 = {x["uniprot"]: x["ef1"] for x in b["per_target"]}
            common = [u for u in r6 if u in r8 and u in sizes8]
            under = [u for u in common if sizes8[u] <= CAP]
            over = [u for u in common if sizes8[u] > CAP]
            if len(under) < 5 or len(over) < 5:
                continue
            for grp, us in (("under", under), ("over", over)):
                agg[grp]["6"] += [r6[u] for u in us]
                agg[grp]["8"] += [r8[u] for u in us]
            print("%-20s %-4s %8d %8.2f %8.2f %8d %8.2f %8.2f" %
                  (m, L, len(under), np.mean([r6[u] for u in under]),
                   np.mean([r8[u] for u in under]),
                   len(over), np.mean([r6[u] for u in over]),
                   np.mean([r8[u] for u in over])))

    print("-" * 86)
    print("\n" + "=" * 86)
    print("汇总：两组各自从 6Å 到 8Å 掉了多少")
    print("=" * 86)
    for grp, label in [("under", "未触顶（完整口袋，无截断）"),
                       ("over", "触顶（被随机截断）")]:
        x = np.array(agg[grp]["6"]); y = np.array(agg[grp]["8"])
        if len(x) < 10:
            continue
        drop = (y.mean() - x.mean()) / x.mean() * 100
        # Paired test on the same set of targets
        p = stats.wilcoxon(x, y).pvalue if len(x) > 10 else float("nan")
        print(f"  {label}")
        print(f"    n={len(x):4d}   6Å {x.mean():6.2f} → 8Å {y.mean():6.2f}"
              f"   衰减 {drop:+.1f}%   配对 p={p:.2e}")

    print("\n" + "=" * 86)
    print("怎么读")
    print("=" * 86)
    print("· 若未触顶组也明显下降 → 8Å 变差是真的口袋尺度效应，不是截断伪影")
    print("· 若只有触顶组下降     → 结论要改写为「截断导致」")
    print("· 参照：4Å 触顶率 0%，其 31–62% 的退化完全不含截断因素")


if __name__ == "__main__":
    main()
