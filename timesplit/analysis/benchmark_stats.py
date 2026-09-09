"""T3 数据集的描述性统计——论文 Part 1 要的那张表。

写清楚这个 benchmark 长什么样：多少靶点、多少分子、活性怎么分布、
蛋白类别怎么构成、结构从哪来、各层的规模。所有数字都从数据本身数出来，
不引用任何文档里的旧值。

两栏并排：全量（1,144 条）和实际用于报告的 350 配额子集（328 条）。
子集是按「活性 ≥50 且类别构成对齐参照数据集」选出来的，两栏放一起
能看出这个筛选把什么样的靶点筛掉了。
"""
import argparse
import collections
import csv
import os
import statistics as st

LAYERS = ["L1", "L2", "L3", "L4"]
LAYER_DESC = {
    "L1": "靶点见过 · 骨架见过",
    "L2": "靶点见过 · 骨架新",
    "L3": "靶点新 · 家族见过",
    "L4": "靶点新 · 家族也新",
}


def q(v, p):
    v = sorted(v)
    if not v:
        return float("nan")
    i = (len(v) - 1) * p
    lo, hi = int(i), min(int(i) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (i - lo)


def block(rows, title, out):
    n_t = len({r["uniprot"] for r in rows})
    act = [int(r["n_actives"]) for r in rows]
    dec = [int(r["n_decoys"]) for r in rows]
    print(f"\n{title}")
    print("=" * 84)
    print(f"  条目 {len(rows):,}（唯一靶点 {n_t:,}）  "
          f"活性合计 {sum(act):,}  诱饵合计 {sum(dec):,}  "
          f"分子合计 {sum(act)+sum(dec):,}")
    print(f"  每靶点活性：中位 {st.median(act):.0f}  均值 {st.mean(act):.0f}  "
          f"四分位 {q(act,.25):.0f}–{q(act,.75):.0f}  "
          f"范围 {min(act)}–{max(act)}")
    print(f"  活性:诱饵 = 1:{sum(dec)/sum(act):.0f}")

    print("\n  分层")
    print("  %-4s %-22s %6s %8s %10s %10s" %
          ("层", "含义", "条目", "靶点", "活性", "中位活性"))
    print("  " + "-" * 66)
    for L in LAYERS:
        g = [r for r in rows if r["layer"] == L]
        if not g:
            continue
        a = [int(r["n_actives"]) for r in g]
        print("  %-4s %-22s %6d %8d %10s %10.0f" %
              (L, LAYER_DESC[L], len(g), len({r["uniprot"] for r in g}),
               f"{sum(a):,}", st.median(a)))
        out.append([title, "layer", L, len(g), len({r["uniprot"] for r in g}),
                    sum(a), f"{st.median(a):.0f}"])

    print("\n  蛋白类别")
    c = collections.Counter(r["protein_class"] for r in rows)
    print("  %-14s %6s %7s %10s" % ("类别", "条目", "占比", "中位活性"))
    print("  " + "-" * 42)
    for k, v in c.most_common():
        a = [int(r["n_actives"]) for r in rows if r["protein_class"] == k]
        print("  %-14s %6d %6.1f%% %10.0f" % (k, v, 100 * v / len(rows), st.median(a)))
        out.append([title, "class", k, v, "", sum(a), f"{st.median(a):.0f}"])

    print("\n  结构来源")
    c = collections.Counter(r["structure_grade"] for r in rows)
    for k, v in c.most_common():
        print("  %-22s %6d %6.1f%%" % (k, v, 100 * v / len(rows)))
        out.append([title, "structure", k, v, "", "", ""])

    thr = [50, 100, 200, 500]
    print("\n  活性数达标情况")
    for t in thr:
        n = sum(1 for r in rows if int(r["n_actives"]) >= t)
        print("  ≥%-5d %6d 条 %6.1f%%" % (t, n, 100 * n / len(rows)))
        out.append([title, "actives_ge", t, n, "", "", f"{100*n/len(rows):.1f}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default="results/T3_targets.csv")
    ap.add_argument("--subset", default="results/T3_vsds_matched.csv")
    ap.add_argument("--out", default="results/T3_benchmark_stats.csv")
    args = ap.parse_args()

    full = list(csv.DictReader(open(args.full)))
    sub = list(csv.DictReader(open(args.subset)))
    out = [["dataset", "kind", "key", "n_entries", "n_targets", "n_actives",
            "value"]]
    block(full, "全量 T3", out)
    block(sub, "350 配额子集（实际用于报告）", out)

    print("\n\n子集相对全量筛掉了什么")
    print("=" * 84)
    fa = [int(r["n_actives"]) for r in full]
    sa = [int(r["n_actives"]) for r in sub]
    print(f"  条目 {len(full):,} → {len(sub):,}（保留 {100*len(sub)/len(full):.1f}%）")
    print(f"  中位活性 {st.median(fa):.0f} → {st.median(sa):.0f}")
    print(f"  活性合计 {sum(fa):,} → {sum(sa):,}"
          f"（保留 {100*sum(sa)/sum(fa):.1f}%）")
    fc = collections.Counter(r["protein_class"] for r in full)
    sc = collections.Counter(r["protein_class"] for r in sub)
    print("\n  %-14s %10s %10s %10s" % ("类别", "全量占比", "子集占比", "变化"))
    print("  " + "-" * 48)
    for k in sorted(set(fc) | set(sc), key=lambda x: -sc.get(x, 0)):
        pf, ps = 100 * fc.get(k, 0) / len(full), 100 * sc.get(k, 0) / len(sub)
        print("  %-14s %9.1f%% %9.1f%% %+9.1f pt" % (k, pf, ps, ps - pf))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(out)
    print(f"\n写入 {args.out}")


if __name__ == "__main__":
    main()
