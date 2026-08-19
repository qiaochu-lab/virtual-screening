"""把 8Å 的结果按「口袋是否超过 511 原子上限」分层。

为什么需要
----------
模型对口袋原子数有上限（DrugCLIP/BindCLIP 是 511）。超限时代码会按
到口袋几何中心的距离加权**随机抽样**保留 511 个——也就是说，超限的靶点
模型看到的不是完整口袋，而是随机采样过的一部分。

我们测出 8Å 比 6Å 差 39–75%。但 8Å 有 10.8% 的口袋超过 511 被截断了，
所以这个退化里可能掺了「截断伪影」，不纯是「口袋太大」。

这里把 8Å 的靶点分成两组直接比：
  · 触顶组（>511 原子，被随机截断过）
  · 未触顶组（≤511 原子，完整口袋）

判读
----
· 只有触顶组掉得厉害  → 8Å 的退化主要是截断造成的，结论要改写
· 两组都掉            → 是真的口袋尺度效应，结论成立
· 未触顶组也明显掉    → 更强的证据，因为这组完全没有截断因素

对照：4Å 触顶率为 0%，它的退化（31–62%）本来就不含截断因素，
可以作为「纯口袋效应」的参照。
"""
import json
import os
import pickle

import lmdb
import numpy as np
from scipy import stats

B = "/data/yicheng/xqc/vs-benchmark"
CAP = 511


def pocket_sizes(threshold):
    """uniprot -> 该阈值下的口袋原子数。PDB 源优先（与组装时一致）。"""
    out = {}
    for pref in ["pocket", "pdb_pocket"]:      # pdb 源后加载，覆盖 boltz 源
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
        # 同一批靶点的配对检验
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
