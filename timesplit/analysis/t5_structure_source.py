"""T5-a: on the same set of targets, how much do results differ between
experimental holo structures and Boltz-2 predicted structures.

This is one of T5's "structure robustness" core questions: how sensitive is
the model to the structure source? The material is already at hand — which
structure type was used for each target when T3 data was assembled is
already recorded per-target in data/T3_6A/manifest.json, so it's a matter
of slicing by source directly.

Note that this is **not** a random control: whether a target has an
experimental structure is itself non-random (only well-studied targets have
one), so the two groups' target difficulty may differ to begin with. The
negative control is two **pure sequence models** (ConPLex, LigUnity-protein)
— they don't use structure at all, so if they show the same gap between the
two groups, that indicates the gap comes from the targets themselves rather
than the structure source.

⚠️ --models must be passed explicitly
------------
An earlier version iterated the models in summary.json with
`for m in sorted(s)`. That file gets overwritten with whichever model set
was run each time score_t3.py runs, so which models this table reported
depended on what the previous command happened to run — the docs once
reported only two BindCLIP models this way and concluded "no significant
difference", while running all ten models shows four are significant at L4.
Model names must now be passed explicitly; whichever is missing gets
reported as missing, instead of silently dropped.
"""
import argparse
import csv
import json
import os

import numpy as np
from scipy import stats

from _subset import add_subset_arg, load_subset

B = "/data/work/vs"
ALL = ["drugclip", "bindclip_randneg", "bindclip_hardneg",
       "ligunity_pocket_ranking", "ligunity_protein_ranking", "litenclip",
       "hypseek_official_vs", "hypseek_rk", "conglude", "conplex", "sprint"]
# Models that don't use structure, used as a negative control
SEQ_ONLY = {"conplex", "ligunity_protein_ranking"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=ALL,
                    help="要报的模型；默认全部十个。缺失的会明确报出来。")
    ap.add_argument("--summary", default=f"{B}/results/t3/summary.json")
    add_subset_arg(ap)
    ap.add_argument("--out", default=f"{B}/results/export/T5_structure_source.csv")
    args = ap.parse_args()

    keep = load_subset(args.subset)
    man = json.load(open(f"{B}/data/T3_6A/manifest.json"))
    src = {}
    for L, d in man.items():
        for up, info in (d.get("per_target") or {}).items():
            src[(L, up)] = info["pocket_source"]

    s = json.load(open(args.summary))
    missing = [m for m in args.models if m not in s]
    if missing:
        print(f"⚠️ summary.json 里缺这些模型，本次未报: {', '.join(missing)}")
        print(f"   先跑: python score_t3.py --models {' '.join(args.models)} "
              f"--layers L1 L2 L3 L4\n")

    rows = [["model", "layer", "n_holo", "ef1_holo", "sem_holo",
             "n_pred", "ef1_pred", "sem_pred", "p", "seq_only"]]
    print("T5-a  结构来源对照（EF1%，只有新靶点层两种来源都有）")
    print("=" * 84)
    print("%-24s %-4s %7s %14s %7s %14s %9s %s" %
          ("模型", "层", "n(holo)", "实验 holo", "n(pred)", "Boltz-2 预测", "p", ""))
    print("-" * 84)
    sig, direction = [], []
    for m in args.models:
        if m not in s:
            continue
        for L in ["L3", "L4"]:
            if L not in s[m]:
                continue
            r = [x for x in s[m][L]["per_target"]
                 if keep is None or (L, x["uniprot"]) in keep]
            h = [x["ef1"] for x in r if src.get((L, x["uniprot"])) == "pdb_holo"]
            p_ = [x["ef1"] for x in r if src.get((L, x["uniprot"])) == "boltz2_pred"]
            if len(h) < 8 or len(p_) < 8:
                continue
            pv = stats.mannwhitneyu(h, p_, alternative="two-sided").pvalue
            mark = " *" if pv < 0.05 else ""
            tag = " (纯序列)" if m in SEQ_ONLY else ""
            print("%-24s %-4s %7d %8.2f±%-5.2f %7d %8.2f±%-5.2f %9.4f%s%s" %
                  (m, L, len(h), np.mean(h), stats.sem(h),
                   len(p_), np.mean(p_), stats.sem(p_), pv, mark, tag))
            rows.append([m, L, len(h), f"{np.mean(h):.4f}", f"{stats.sem(h):.4f}",
                         len(p_), f"{np.mean(p_):.4f}", f"{stats.sem(p_):.4f}",
                         f"{pv:.4f}", int(m in SEQ_ONLY)])
            if pv < 0.05:
                sig.append((m, L, pv))
            if L == "L4":
                direction.append(np.mean(h) > np.mean(p_))
    print("-" * 84)

    # Multiple comparisons: BH step-up procedure
    pvals = sorted(float(r[8]) for r in rows[1:])
    n = len(pvals)
    k_max = max((i for i, p in enumerate(pvals, 1) if p <= 0.05 * i / n), default=0)
    print(f"\n共 {n} 次比较，单独 p<0.05 的 {len(sig)} 个："
          f"{', '.join(f'{m}/{L}' for m, L, _ in sig) or '无'}")
    print(f"BH-FDR (α=0.05) 后存活 {k_max} 个"
          + (f"（{', '.join(f'{m}/{L}' for m, L, _ in sorted(sig, key=lambda x: x[2])[:k_max])}）"
             if k_max else ""))

    if direction:
        from math import comb
        k = sum(direction)
        # ⚠️ Must be two-sided. The original code used 2*P(X>=k), which only
        # detects the single direction "experimental structure is better": if it
        # went the other way (k < n/2), that formula returns a number >1 and can't
        # detect "predicted structure is significantly better".
        # This time k=8/10 and 10/10 both fall on the k>n/2 side, so the two
        # formulas happen to be equal and **the already-published numbers are
        # unaffected** — but it would fail silently on a different dataset. A
        # teammate hit the same pitfall in paired_vs_mw.py, and that version also
        # clamped to 1, turning "significantly worse" into "no difference".
        _n = len(direction)
        _pk = [comb(_n, i) for i in range(_n + 1)]
        sign = min(1.0, sum(x for x in _pk if x <= _pk[k]) / 2 ** _n)
        print(f"L4 方向性：{k}/{len(direction)} 个模型实验结构更好，符号检验 p={sign:.3f}")

    seq = [r for r in rows[1:] if r[9] == 1 and r[1] == "L4"]
    if seq:
        print("\n阴性对照（纯序列模型，完全不用结构）：")
        for r in seq:
            ok = "✓ 无差距" if float(r[8]) > 0.1 else "✗ 也有差距 → 差距可能来自靶点本身"
            print(f"  {r[0]:<26} {r[3]} vs {r[6]}   p={r[8]}  {ok}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\n写入 {args.out}")


if __name__ == "__main__":
    main()
