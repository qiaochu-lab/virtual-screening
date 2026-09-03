"""每靶点 active 数对结论的影响：≥10 / ≥20 / ≥30 / ≥50 的梯度分析。

问题
----
EF@fraction 卡在一个截断位置上，命中数只能取整数，所以它的取值被 active 数
量化成台阶：步长约 100/A。每靶点只有 10 个 active 时步长约 8.5，而各层 EF1%
均值才 8–39——一个分子的位置变化就能让 EF 跳掉整个均值的量级。跨靶点平均时，
这种粗糙测量和 665 个 active 那种精细测量（步长 0.15）被等权对待。

做法
----
逐步抬高 active 数门槛，看两件事变不变：
  1. 各层的绝对水平和 L1→L4 衰减
  2. 模型之间的排名
同时报 PR-AUC——它用整个排序、不卡截断，没有 EF 的量化问题，且对 1:50
这种不平衡比 ROC-AUC 敏感。如果 EF 的结论随门槛漂移而 PR-AUC 不漂，
那说明漂移来自指标的粗糙度而不是模型。

用法
----
    python actives_gradient.py [--raw 原始打分目录] [--out 输出CSV前缀]
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

    # 逐模型逐靶点算一次，之后按门槛过滤，避免重复计算
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

    # --- 表一：各层水平随门槛怎么变（拿最好的模型举例）
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

    # --- 表二：L1→L4 衰减随门槛怎么变
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
                if bb is None:            # PR-AUC 的随机基线是 active 占比，逐层不同
                    bb = 0.0
                num, den = (a - bb) - (b - bb), (a - bb)
                cells.append(f"{100*num/den:+8.0f}%" if den > 1e-9 else "    —")
            print("%-20s %9s %9s %9s %9s" % (NICE.get(m, m), *cells))

    # --- 表三：模型排名随门槛变不变
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
