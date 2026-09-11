"""Recompute T2's ranking metrics on a target subset.

score_t2_v2.py originally only stored per-layer summary values, which made
it impossible to recompute on a different target subset, so per_target
output (uniprot / spearman / kendall / n_actives) was added to it. This
script only filters and averages — it does not re-run inference.
"""
import argparse
import collections
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _subset import add_subset_arg, load_subset

B = "/data/work/vs-benchmark"
LAYERS = ["L1", "L2", "L3", "L4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=f"{B}/results/t3/summary_t2_v2.json")
    add_subset_arg(ap)
    ap.add_argument("--out", default=f"{B}/results/export/T2_on_T3_subset.csv")
    args = ap.parse_args()

    s = json.load(open(args.summary))
    keep = load_subset(args.subset)

    rows = [["model", "layer", "n_targets", "spearman", "spearman_sem",
             "kendall", "frac_positive"]]
    print("%-26s %-4s %8s %16s %9s %10s" %
          ("模型", "层", "靶点", "Spearman", "Kendall", "正向靶点"))
    print("-" * 78)
    for m in sorted(s, key=lambda x: -(s[x].get("L1", {}).get("spearman") or 0)):
        for L in LAYERS:
            b = s[m].get(L)
            if not b:
                continue
            pt = b.get("per_target")
            if pt is None:
                print(f"  ⚠️ {m}/{L} 没有 per_target，请先重跑 score_t2_v2.py")
                continue
            v = [x for x in pt if keep is None or (L, x["uniprot"]) in keep]
            if len(v) < 3:
                continue
            r = np.array([x["spearman"] for x in v])
            k = np.array([x["kendall"] for x in v])
            sem = r.std(ddof=1) / np.sqrt(len(r))
            print("%-26s %-4s %8d %16s %9.3f %9.0f%%" %
                  (m, L, len(r), f"{r.mean():+.3f}±{sem:.3f}",
                   k.mean(), 100 * (r > 0).mean()))
            rows.append([m, L, len(r), f"{r.mean():.4f}", f"{sem:.4f}",
                         f"{k.mean():.4f}", f"{(r > 0).mean():.4f}"])
        print()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"写入 {args.out}")


if __name__ == "__main__":
    main()
