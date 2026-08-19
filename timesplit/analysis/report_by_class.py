"""分类别报表：新靶点层（L3+L4 合并）按靶点类别拆开报指标。

为什么合并 L3+L4
----------------
L3 只有 53 个靶点，任何单一类别都凑不到能报的样本量。而 L3（新靶点·同家族）
和 L4（新靶点·新家族）都属于「训练时没见过的靶点」，合并在科学上也说得通——
要区分同家族/新家族时，另有 L3 vs L4 的对比表。

为什么按类别拆
--------------
不同类别靶点难度差异极大。自家数据实证：ConPLex 在 L1 上
GPCR AUROC 0.777、表观 0.518，同模型同口径差 0.26。
只报总平均会把这种分化完全掩盖。

样本量不足的类别照样列出来并标注 n，不藏起来——读者需要知道
哪些类别是「测了但不够」，哪些是「压根没有」。
"""
import argparse
import json

import numpy as np
from scipy import stats

B = "/data/work/vs-benchmark"
MIN_REPORT = 8          # 低于此只列 n，不报指标


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--metric", default="ef1", choices=["auroc", "bedroc", "ef1", "ef5"])
    args = ap.parse_args()

    s = json.load(open(f"{B}/results/t3/summary.json"))
    cls = json.load(open(f"{B}/data/t3/target_class.json"))["class"]

    # 收集：模型 -> 类别 -> [每靶点指标]
    data = {}
    for m in args.models:
        if m not in s:
            continue
        rows = []
        for L in ["L3", "L4"]:
            if L in s[m]:
                rows += s[m][L]["per_target"]
        by = {}
        for r in rows:
            by.setdefault(cls.get(r["uniprot"], "其他/未分类"), []).append(r)
        data[m] = by

    if not data:
        print("没有可用结果")
        return

    order = ["GPCR", "激酶", "离子通道", "表观", "蛋白酶", "转运体",
             "核受体", "P450", "其他酶", "其他/未分类"]
    models = list(data)

    print(f"新靶点层（L3+L4 合并）分类别 {args.metric.upper()}")
    print("=" * (16 + 22 * len(models)))
    print("%-12s %5s %s" % ("类别", "n", " ".join(f"{m:>20s}" for m in models)))
    print("-" * (16 + 22 * len(models)))
    for k in order:
        ns = [len(data[m].get(k, [])) for m in models]
        n = max(ns) if ns else 0
        if n == 0:
            print("%-12s %5s %s" % (k, "0", "  ".join(f"{'—':>20s}" for _ in models)))
            continue
        cells = []
        for m in models:
            v = data[m].get(k, [])
            if len(v) < MIN_REPORT:
                cells.append(f"n={len(v)} 样本不足".rjust(20))
            else:
                a = np.array([x[args.metric] for x in v])
                cells.append(f"{a.mean():.2f}±{a.std(ddof=1)/np.sqrt(len(a)):.2f} (n={len(a)})".rjust(20))
        print("%-12s %5d %s" % (k, n, " ".join(cells)))
    print("-" * (16 + 22 * len(models)))
    for m in models:
        allv = [x[args.metric] for v in data[m].values() for x in v]
        print(f"  {m} 全体: {np.mean(allv):.2f}  (n={len(allv)})")

    # 两模型时给配对检验
    if len(models) == 2:
        ma, mb = models
        print(f"\n{ma} vs {mb} 分类别配对检验（Wilcoxon，只对 n≥{MIN_REPORT} 的类）:")
        for k in order:
            A = {x["uniprot"]: x for x in data[ma].get(k, [])}
            Bd = {x["uniprot"]: x for x in data[mb].get(k, [])}
            common = sorted(set(A) & set(Bd))
            if len(common) < MIN_REPORT:
                continue
            a = np.array([A[u][args.metric] for u in common])
            b = np.array([Bd[u][args.metric] for u in common])
            try:
                p = stats.wilcoxon(a, b).pvalue
            except ValueError:
                p = float("nan")
            print(f"  {k:10s} n={len(common):3d}  {ma} {a.mean():6.2f}  {mb} {b.mean():6.2f}"
                  f"  差 {a.mean()-b.mean():+6.2f}  p={p:.4f}{'  *' if p < 0.05 else ''}")


if __name__ == "__main__":
    main()
