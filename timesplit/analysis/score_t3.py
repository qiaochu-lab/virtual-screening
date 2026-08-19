"""汇总 T3 评测结果：读各模型落盘的原始分数，用统一评测层算指标。

输入布局（各模型 runner 统一产出）：
    results/t3/<model>/<layer>/<uniprot>/saved_preds.npy
                                        /saved_labels.npy

为什么每个模型都要落原始分数：各家论文自带的指标实现不一致（同一个 DrugCLIP
在自己论文和 BindCLIP 论文里差近 5%），只有统一重算才能保证横评表里的差异
只来自模型本身。详见 eval/README.md。

跨层比较用 AUROC 而不是 EF —— EF 依赖库大小与活性比例，虽然这里各层都是
1:50，但每层靶点数与活性配体数分布不同，AUROC 更稳。EF 仍然报，
因为它是虚筛领域的通用语言。
"""
import argparse
import json
import os
import sys

import numpy as np

B = "/data/work/vs-benchmark"
sys.path.insert(0, B)
from eval.metrics import enrichment_factor, roc_auc, bedroc  # noqa: E402


def score_layer(d):
    rows = []
    for up in sorted(os.listdir(d)):
        p = f"{d}/{up}"
        if not os.path.isdir(p):
            continue
        try:
            s = np.load(f"{p}/saved_preds.npy")
            l = np.load(f"{p}/saved_labels.npy")
        except (FileNotFoundError, OSError):
            continue
        if l.sum() == 0 or l.sum() == len(l):
            continue
        rows.append({
            "uniprot": up, "n": int(len(l)), "n_act": int(l.sum()),
            "auroc": roc_auc(s, l),
            "bedroc": bedroc(s, l, alpha=80.5),
            "ef1": enrichment_factor(s, l, 0.01),
            "ef5": enrichment_factor(s, l, 0.05),
            "ef01": enrichment_factor(s, l, 0.001),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--layers", nargs="+", default=["L1", "L2", "L3", "L4"])
    ap.add_argument("--root", default=f"{B}/results/t3")
    ap.add_argument("--out", default=f"{B}/results/t3/summary.json")
    args = ap.parse_args()

    summary = {}
    print("%-12s %-4s %6s %7s %8s %8s %8s %8s" %
          ("模型", "层", "靶点", "AUROC", "BEDROC", "EF1%", "EF5%", "EF0.1%"))
    print("-" * 72)
    for m in args.models:
        summary[m] = {}
        for L in args.layers:
            d = f"{args.root}/{m}/{L}"
            if not os.path.isdir(d):
                continue
            rows = score_layer(d)
            if not rows:
                continue
            agg = {k: float(np.mean([r[k] for r in rows]))
                   for k in ["auroc", "bedroc", "ef1", "ef5", "ef01"]}
            # 每靶点自成一个统计单元，用靶点间标准误做不确定度
            agg["auroc_sem"] = float(np.std([r["auroc"] for r in rows], ddof=1)
                                     / np.sqrt(len(rows)))
            agg["n_targets"] = len(rows)
            agg["per_target"] = rows
            summary[m][L] = agg
            print("%-12s %-4s %6d %7.3f %8.3f %8.2f %8.2f %8.2f" %
                  (m, L, len(rows), agg["auroc"], agg["bedroc"],
                   agg["ef1"], agg["ef5"], agg["ef01"]))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=1)

    # L1→L4 的衰减：本 benchmark 的核心量
    print("\n泛化衰减（相对 L1 的 AUROC）:")
    for m, d in summary.items():
        if "L1" not in d:
            continue
        base = d["L1"]["auroc"]
        cells = []
        for L in ["L2", "L3", "L4"]:
            if L in d:
                drop = (d[L]["auroc"] - base) / (base - 0.5) * 100 if base > 0.5 else float("nan")
                cells.append(f"{L} {d[L]['auroc']:.3f} ({drop:+.0f}%)")
        print(f"  {m:12s} L1 {base:.3f}  ->  " + "   ".join(cells))
    print("\n（衰减按超出随机的部分算：(AUROC-0.5) 的相对变化，避免 0.5 基线稀释差异）")
    print(f"\n已写入 {args.out}")


if __name__ == "__main__":
    main()
