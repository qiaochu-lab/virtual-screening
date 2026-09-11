"""Recompute T3's metrics on a given target subset — no re-inference, just
re-aggregating over a different set of targets.

Purpose
----
The new convention set by the advisor on 2026-09-04 (actives >= 50, class
composition matched to VSDS-vd) produced a subset (final quota 350 →
**328 entries / 293 unique targets**; the earlier 250-quota version, 242
entries / 222 targets, is no longer used). Each target's EF/AUROC is
computed within its **own** candidate pool, targets don't affect each
other, so switching subsets is just averaging over a different set of
numbers — no model needs to be re-run.

Also outputs two stratifications
----------------
build_t3.py decides L3/L4 with `fam.get(up)`; if a target isn't in
uniport40.clstr, it returns None and falls straight through to L4 —
"family not found" was treated as "no homologous family". 61 of 254 L4
targets (24%) aren't in that file at all, and an mmseqs search against the
training set finds >=40% homologues for 30 of them (including cross-species
orthologs at 100% full-length identity). So this script reports both the
original and the corrected stratification side by side, so the difference
is visible at a glance.

Decay is always computed as "excess over random": ((L1-1)-(L4-1))/(L1-1),
since EF's random floor is 1.0.
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
    """Fraction of loss relative to the random baseline. floor is the metric's random-chance value."""
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
