"""Effect of per-target actives count on the conclusions: a gradient analysis
at >=10 / >=20 / >=30 / >=50.

Problem
-------
EF@fraction is pinned at a cutoff position, and the hit count can only take
integer values, so its value is quantized by the actives count into steps of
roughly 100/A. At only 10 actives per target the step is about 8.5, while the
per-layer EF1% means are only 8-39 — one molecule changing rank can move EF by
an amount comparable to the whole mean. When averaging across targets, this
coarse measurement is given equal weight to a target with 665 actives, whose
step is 0.15.

Approach
--------
Progressively raise the actives-count floor and check whether two things
change:
  1. The absolute level per layer and the L1->L4 decay
  2. The ranking among models
Also report PR-AUC — it uses the whole ranking, has no cutoff, carries none of
EF's quantization issue, and is more sensitive than ROC-AUC to the 1:50 class
imbalance here. If EF's conclusion drifts with the floor while PR-AUC does
not, the drift comes from the metric's coarseness, not from the models.

Usage
-----
    python actives_gradient.py [--raw raw-scores dir] [--out output CSV prefix]
"""
import argparse
import collections
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
try:
    from metrics import enrichment_factor, roc_auc, bedroc, pr_auc
except ImportError:
    sys.path.insert(0, "/data/work/vs/eval")
    from metrics import enrichment_factor, roc_auc, bedroc, pr_auc

LAYERS = ("L1", "L2", "L3", "L4")
THRESHOLDS = (10, 20, 30, 50)
METRICS = (("EF1%", lambda s, y: enrichment_factor(s, y, 0.01)),
           ("BEDROC", lambda s, y: bedroc(s, y, 80.5)),
           ("PR-AUC", pr_auc),
           ("AUROC", roc_auc))
NICE = {"hypseek_rk": "HypSeek", "ligunity_protein_ranking": "LigUnity-protein",
        "ligunity_pocket_ranking": "LigUnity-pocket", "litenclip": "LiTENCLIP",
        "drugclip": "DrugCLIP", "bindclip_randneg": "BindCLIP-randneg",
        "bindclip_hardneg": "BindCLIP-hardneg", "conglude": "ConGLUDe",
        "conplex": "ConPLex", "sprint": "SPRINT"}


def load(raw_dir, model):
    p = os.path.join(raw_dir, f"T3_{model}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    out = {}
    for t in {k.rsplit("/", 1)[0] for k in z.files}:
        parts = t.split("/")
        if len(parts) < 3:
            continue
        out[(parts[1], parts[-1])] = (z[f"{t}/preds"], z[f"{t}/labels"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/data/work/vs/results/raw_release")
    ap.add_argument("--out", default="T3_actives_gradient")
    args = ap.parse_args()

    models = [m for m in NICE if os.path.exists(os.path.join(args.raw, f"T3_{m}.npz"))]
    print(f"模型 {len(models)} 个: {', '.join(models)}\n")

    # Compute once per model per target, then filter by floor afterward, to
    # avoid recomputing
    per = {}
    counts = collections.Counter()
    for m in models:
        d = load(args.raw, m)
        per[m] = {}
        for (layer, up), (s, y) in d.items():
            n_act = int(y.sum())
            if n_act < 10 or (y == 0).sum() < 1:
                continue
            vals = {}
            for nm, fn in METRICS:
                try:
                    v = fn(s, y)
                except Exception:
                    v = float("nan")
                vals[nm] = v
            per[m][(layer, up)] = (n_act, vals)
        counts[m] = len(per[m])

    rows = [["model", "layer", "min_actives", "n_targets"] + [nm for nm, _ in METRICS]]
    summary = collections.defaultdict(dict)
    for m in models:
        for L in LAYERS:
            for th in THRESHOLDS:
                sel = [v for (lay, _), (na, v) in per[m].items()
                       if lay == L and na >= th]
                if len(sel) < 5:
                    continue
                row = [m, L, th, len(sel)]
                for nm, _ in METRICS:
                    a = np.array([x[nm] for x in sel], dtype=float)
                    a = a[np.isfinite(a)]
                    row.append(f"{a.mean():.4f}" if len(a) else "")
                    summary[(m, L, th)][nm] = a.mean() if len(a) else float("nan")
                rows.append(row)

    with open(f"{args.out}.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)

    # --- Table 1: how the per-layer level changes with the floor (using the
    # best model as an example)
    ref = "ligunity_protein_ranking" if "ligunity_protein_ranking" in models else models[0]
    print(f"== 门槛对绝对水平的影响（{NICE.get(ref, ref)}）==")
    for nm, _ in METRICS:
        print(f"\n{nm}")
        print("%-4s %10s %10s %10s %10s" % ("层", "≥10", "≥20", "≥30", "≥50"))
        print("-" * 48)
        for L in LAYERS:
            cells = []
            for th in THRESHOLDS:
                v = summary.get((ref, L, th), {}).get(nm)
                n = next((r[3] for r in rows[1:] if r[0] == ref and r[1] == L and r[2] == th), "-")
                cells.append(f"{v:.3f}({n})" if v == v else "  —")
            print("%-4s %10s %10s %10s %10s" % (L, *cells))

    # --- Table 2: how the L1->L4 decay changes with the floor
    print("\n\n== L1→L4 衰减随门槛的变化（所有模型）==")
    print("衰减按超出随机的部分算：(L1−base)−(L4−base) 相对 (L1−base)")
    for nm, base in (("EF1%", 1.0), ("BEDROC", 0.0), ("PR-AUC", None), ("AUROC", 0.5)):
        print(f"\n{nm}")
        print("%-20s %9s %9s %9s %9s" % ("模型", "≥10", "≥20", "≥30", "≥50"))
        print("-" * 60)
        for m in models:
            cells = []
            for th in THRESHOLDS:
                a = summary.get((m, "L1", th), {}).get(nm)
                b = summary.get((m, "L4", th), {}).get(nm)
                if a is None or b is None or a != a or b != b:
                    cells.append("    —"); continue
                bb = base
                if bb is None:            # PR-AUC's random baseline is the actives fraction, which differs per layer
                    bb = 0.0
                num, den = (a - bb) - (b - bb), (a - bb)
                cells.append(f"{100*num/den:+8.0f}%" if den > 1e-9 else "    —")
            print("%-20s %9s %9s %9s %9s" % (NICE.get(m, m), *cells))

    # --- Table 3: whether the model ranking changes with the floor
    print("\n\n== 模型排名随门槛的变化 ==")
    for nm, _ in METRICS:
        for L in ("L1", "L4"):
            order = {}
            for th in THRESHOLDS:
                v = [(summary.get((m, L, th), {}).get(nm), m) for m in models]
                v = [(x, m) for x, m in v if x is not None and x == x]
                order[th] = [m for _, m in sorted(v, reverse=True)]
            if not order[10]:
                continue
            same = all(order[th][:3] == order[10][:3] for th in THRESHOLDS if order[th])
            top3 = " > ".join(NICE.get(m, m) for m in order[10][:3])
            print(f"{nm:>8} {L}: 前三名 {'不变' if same else '**变了**'}   ≥10 时为 {top3}")
            if not same:
                for th in THRESHOLDS[1:]:
                    if order[th][:3] != order[10][:3]:
                        print(f"{'':>13}≥{th} 时为 " +
                              " > ".join(NICE.get(m, m) for m in order[th][:3]))

    print(f"\n逐格结果写入 {args.out}.csv")


if __name__ == "__main__":
    main()
