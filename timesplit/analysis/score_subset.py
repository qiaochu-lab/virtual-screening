"""把 T3 的指标重算到某个靶点子集上——不重新推理，只换一批靶点做汇总。

用途
----
导师 2026-09-04 定的新口径（活性 ≥50、类别构成对齐 VSDS-vd）产出了一个
242 条（222 个唯一靶点）的子集。每个靶点的 EF/AUROC 都是在它**自己的**
候选池里算的，靶点之间互不影响，所以换子集只是换一批数求平均，
不需要重跑任何模型。

同时输出两种分层
----------------
build_t3.py 判 L3/L4 时用 `fam.get(up)`，靶点不在 uniport40.clstr 里就
返回 None，直接落到 L4——「没查到家族」被当成了「没有同源家族」。
254 个 L4 里有 61 个（24%）根本不在那个文件里，其中 30 个用 mmseqs 对
训练集一查就有 ≥40% 的同源物（含 100% 全长一致的跨物种直系同源）。
所以这里并排给出原分层和修正分层，差多少一眼可见。

衰减一律按「超出随机」算：((L1-1)-(L4-1))/(L1-1)，因为 EF 的随机底是 1.0。
"""
import argparse
import collections
import csv
import json
import os

import numpy as np

B = "/data/work/vs-benchmark"
METRICS = ["ef1", "ef5", "bedroc", "auroc"]
LAYERS = ["L1", "L2", "L3", "L4"]


def decay(l1, l4, floor):
    """超出随机基线的损失比例。floor 是该指标的随机值。"""
    base = l1 - floor
    return float("nan") if base <= 0 else (base - (l4 - floor)) / base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=f"{B}/results/t3/summary.json")
    ap.add_argument("--subset", default=f"{B}/results/export/T3_vsds_matched.csv")
    ap.add_argument("--mirroring", default=f"{B}/results/export/T3_target_mirroring.csv",
                    help="mmseqs 查出的对训练集最高同源；用来做修正分层")
    ap.add_argument("--relabel-above", type=float, default=0.40,
                    help="L4 靶点对训练集同源 ≥该值时改判 L3")
    ap.add_argument("--out", default=f"{B}/results/export/T3_main_vsds_subset.csv")
    args = ap.parse_args()

    s = json.load(open(args.summary))
    keep = {(r["layer"], r["uniprot"]) for r in csv.DictReader(open(args.subset))}
    print(f"子集：{len(keep)} 条（靶点×层），"
          f"{len({u for _, u in keep})} 个唯一靶点")

    relabel = {}
    if os.path.exists(args.mirroring):
        for r in csv.DictReader(open(args.mirroring)):
            v = r.get("identity")
            if v and float(v) >= args.relabel_above:
                relabel[r["uniprot"]] = float(v)
        n = len({u for L, u in keep if L == "L4"} & set(relabel))
        print(f"修正分层：子集里 {n} 个 L4 靶点对训练集同源 ≥{args.relabel_above:.0%}，"
              f"改判 L3")

    rows = [["model", "layer", "n_targets", "ef1", "ef5", "bedroc", "auroc",
             "layering"]]
    out = {}
    for mode in ("original", "corrected"):
        for m in sorted(s):
            agg = collections.defaultdict(list)
            for L in LAYERS:
                for t in s[m].get(L, {}).get("per_target", []):
                    if (L, t["uniprot"]) not in keep:
                        continue
                    lay = L
                    if mode == "corrected" and L == "L4" and t["uniprot"] in relabel:
                        lay = "L3"
                    agg[lay].append([t[k] for k in METRICS])
            for L in LAYERS:
                if not agg[L]:
                    continue
                v = np.array(agg[L]).mean(axis=0)
                out[(mode, m, L)] = (len(agg[L]), v)
                rows.append([m, L, len(agg[L])] + [f"{x:.4f}" for x in v] + [mode])

    for mode in ("original", "corrected"):
        title = "原分层" if mode == "original" else \
                f"修正分层（L4 中对训练集同源 ≥{args.relabel_above:.0%} 的改判 L3）"
        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
        print("%-26s %8s %8s %8s %8s %10s %10s" %
              ("模型", "L1", "L2", "L3", "L4", "EF衰减", "AUROC衰减"))
        print("-" * 78)
        order = sorted(s, key=lambda m: -out.get((mode, m, "L1"), (0, [0]))[1][0])
        for m in order:
            cell = {}
            for L in LAYERS:
                r = out.get((mode, m, L))
                cell[L] = r[1] if r is not None else None
            if cell["L1"] is None or cell["L4"] is None:
                continue
            d_ef = decay(cell["L1"][0], cell["L4"][0], 1.0)
            d_au = decay(cell["L1"][3], cell["L4"][3], 0.5)
            print("%-26s %8.2f %8.2f %8.2f %8.2f %9.0f%% %9.0f%%" %
                  (m, cell["L1"][0], cell["L2"][0],
                   cell["L3"][0] if cell["L3"] is not None else float("nan"),
                   cell["L4"][0], -100 * d_ef, -100 * d_au))
        n = {L: out.get((mode, order[0], L), (0,))[0] for L in LAYERS}
        print("-" * 78)
        print("靶点数： " + "  ".join(f"{L} {n[L]}" for L in LAYERS))

    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\n写入 {args.out}")


if __name__ == "__main__":
    main()
