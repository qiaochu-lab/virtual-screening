"""提高 active 数门槛，会不会把某些蛋白类别整个删掉？

背景
----
合作者建议「去掉 50 个 active 以下的靶点看看」。按 active 数筛选不是中性操作：
研究得多的靶点文献里活性化合物多，冷门靶点少。所以抬高门槛可能系统性地保留
热门类别、删掉冷门类别——那样得到的子集就不能代表全集。

这个脚本量化两件事：
  1. 各门槛下每个蛋白类别还剩多少靶点、占比变了多少
  2. 与三个公开基准的 active 数分布对照，看「多少算够」有没有行业标准

用法
----
    python actives_floor_classes.py [--targets CSV] [--out CSV]
"""
import argparse
import collections
import csv

THRESHOLDS = (10, 20, 30, 50, 100)
LAYERS = ("L1", "L2", "L3", "L4")

# 三个公开基准的 active 数分布，从各自的 saved_labels.npy 数出来（见 standard/）
REFERENCE = {
    "DUD-E":    dict(n=102, lo=40, q25=102, med=158, q75=338, hi=592, ge50=95, ge100=82),
    "DEKOIS":   dict(n=81,  lo=37, q25=40,  med=40,  q75=40,  hi=40,  ge50=0,  ge100=0),
    "LIT-PCBA": dict(n=15,  lo=13, q25=27,  med=102, q75=369, hi=7166, ge50=67, ge100=53),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="results/T3_targets.csv")
    ap.add_argument("--out", default="results/T3_actives_floor_classes.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.targets)))
    print(f"T3 共 {len(rows)} 个靶点\n")

    # --- 每层还剩多少
    by_layer = collections.defaultdict(list)
    for r in rows:
        by_layer[r["layer"]].append(int(r["n_actives"]))
    total = {t: 0 for t in THRESHOLDS}
    print("=== 抬高门槛后各层剩余靶点 ===")
    print("%-6s" % "层" + "".join("%9s" % f"≥{t}" for t in THRESHOLDS))
    print("-" * 51)
    for L in LAYERS:
        c = [sum(1 for x in by_layer[L] if x >= t) for t in THRESHOLDS]
        for t, v in zip(THRESHOLDS, c):
            total[t] += v
        print("%-6s" % L + "".join("%9d" % v for v in c))
    print("-" * 51)
    print("%-6s" % "合计" + "".join("%9d" % total[t] for t in THRESHOLDS))
    print("%-6s" % "" + "".join("%9s" % f"{100*total[t]/total[10]:.0f}%" for t in THRESHOLDS))

    # --- 类别构成
    cls = collections.defaultdict(lambda: collections.defaultdict(int))
    for r in rows:
        c = r.get("protein_class") or "(未标注)"
        n = int(r["n_actives"])
        for t in THRESHOLDS:
            if n >= t:
                cls[t][c] += 1
    order = sorted(cls[10], key=lambda k: -cls[10][k])

    print("\n=== 类别构成：括号是占当前子集的比例 ===")
    print("%-16s" % "类别" + "".join("%12s" % f"≥{t}" for t in THRESHOLDS))
    print("-" * 78)
    out_rows = [["protein_class"] + [f"n_ge{t}" for t in THRESHOLDS]
                + [f"pct_ge{t}" for t in THRESHOLDS]]
    for c in order:
        cells, ns, pcts = [], [], []
        for t in THRESHOLDS:
            n = cls[t][c]
            p = 100 * n / max(total[t], 1)
            cells.append(f"{n}({p:.0f}%)")
            ns.append(n)
            pcts.append(f"{p:.1f}")
        print("%-16s" % c + "".join("%12s" % x for x in cells))
        out_rows.append([c] + ns + pcts)
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(out_rows)

    # --- 哪些类别在 ≥50 时已经不够用
    print("\n=== ≥50 时样本不足的类别（少于 10 个靶点就撑不起分层结论）===")
    dead = [(c, cls[10][c], cls[50][c]) for c in order if cls[50][c] < 10]
    for c, a, b in dead:
        print(f"  {c:<14} {a:>4} → {b:>3}")
    if not dead:
        print("  （无）")

    # --- 与公开基准对照
    print("\n=== 「多少算够」有行业标准吗：三个公开基准的 active 数 ===")
    print("%-10s %6s %6s %6s %7s %7s %8s %9s %9s"
          % ("基准", "靶点", "最小", "25%", "中位", "75%", "最大", "≥50", "≥100"))
    print("-" * 76)
    for k, v in REFERENCE.items():
        print("%-10s %6d %6d %6d %7d %7d %8d %8d%% %8d%%"
              % (k, v["n"], v["lo"], v["q25"], v["med"], v["q75"], v["hi"],
                 v["ge50"], v["ge100"]))
    med_all = sorted(int(r["n_actives"]) for r in rows)
    n = len(med_all)
    ge50 = 100 * sum(1 for x in med_all if x >= 50) / n
    ge100 = 100 * sum(1 for x in med_all if x >= 100) / n
    print("%-10s %6d %6d %6d %7d %7d %8d %8.0f%% %8.0f%%"
          % ("T3(我们)", n, med_all[0], med_all[n//4], med_all[n//2],
             med_all[3*n//4], med_all[-1], ge50, ge100))

    print("\n怎么读")
    print("· DEKOIS 每个靶点固定 40 个 active，一个都不到 50，仍是十几年的正规基准，")
    print("  所以「必须 ≥50」不是行业硬标准。")
    print("· 真正的问题不是绝对数量而是同一基准内部的一致性：DEKOIS 全部 40（完全齐平），")
    print(f"  DUD-E 跨 15 倍，我们跨 {med_all[-1]//med_all[0]} 倍。")
    print(f"\n写入 {args.out}")


if __name__ == "__main__":
    main()
