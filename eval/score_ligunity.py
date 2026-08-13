"""读 LigUnity 推理的原始输出，用我们的统一指标层算分。

LigUnity 的输出布局（见 unimol/tasks/test_task.py）：

    {results_path}/{DUDE|PCBA|DEKOIS}/{target}/
        saved_labels.npy         1=active, 0=decoy
        saved_mols_embed.npy     (n_mol, dim)
        saved_target_embed.npy   (n_pocket, dim)
        saved_preds.npy          回归架构才有；有则直接用

打分方式与官方 ensemble_result.py 一致：
``(pocket_reps @ mol_reps.T).max(axis=0)``——同一靶点有多个口袋时取最大相似度。

用法：
    python score_ligunity.py <results_path> [--bootstrap]

输出逐靶点指标 + 汇总平均，并与 docs/paper-reference-values.md 的基准值对比。
"""
import argparse
import json
import os
import sys
from functools import partial

import numpy as np

from metrics import bedroc, bootstrap_ci, enrichment_factor, roc_auc, top_k_recall

# 来自 LigUnity 仓库 results/VS_results/*.csv 的作者基准值（逐靶点平均）。
#
# 重要：CSV 里有两列不能混用——
#   "LigUnity"       = 论文最终数值，是 transformer + H-GNN 的 **ensemble** 结果
#   "LigUnity(seq)"  = 序列塔单模型，对应 arch=protein_ranking
# 因此 protein_ranking 的原始输出应对标 LigUnity(seq)；
# 只有跑完 HGNN + ensemble_result.py 之后才能对标 LigUnity。
REFERENCE = {
    "ensemble": {   # "LigUnity" 列
        "DUDE":   {"n": 102, "EF1": 52.0421, "BEDROC": 0.7886, "AUROC": 0.9310},
        "DEKOIS": {"n": 81,  "EF1": 28.2123, "BEDROC": 0.8487, "AUROC": 0.9409},
        "PCBA":   {"n": 15,  "EF1": 7.3592,  "BEDROC": 0.0889, "AUROC": 0.5895},
    },
    "seq": {        # "LigUnity(seq)" 列
        "DUDE":   {"n": 102, "EF1": 36.8838, "BEDROC": 0.5746, "AUROC": 0.8872},
        "DEKOIS": {"n": 81,  "EF1": 27.1297, "BEDROC": 0.7848, "AUROC": 0.9246},
        "PCBA":   {"n": 15,  "EF1": 6.2208,  "BEDROC": 0.0746, "AUROC": 0.5630},
    },
    "drugclip": {   # "DrugCLIP" 列
        "DUDE":   {"n": 102, "EF1": 31.9905, "BEDROC": 0.4997, "AUROC": 0.8070},
        "DEKOIS": {"n": 81,  "EF1": 17.8579, "BEDROC": 0.5040, "AUROC": 0.7906},
        "PCBA":   {"n": 15,  "EF1": 5.5481,  "BEDROC": 0.0624, "AUROC": 0.5717},
    },
    # BindCLIP 论文 (arXiv 2602.15236) Table 1/2，原文以百分数给出，此处换算为小数。
    # 注意：同一篇论文里重跑的 DrugCLIP 基线是 EF1=30.52 / AUROC=0.7929，
    # 与 DrugCLIP 自己论文的 31.99 / 0.8070 差 4.8%——正是"抄论文数字不可比"的实例。
    "bindclip": {
        "DUDE": {"n": 102, "EF1": 32.16, "BEDROC": 0.4973, "AUROC": 0.8014},
        "PCBA": {"n": 15,  "EF1": 6.26,  "BEDROC": 0.0788, "AUROC": 0.5915},
    },
}


def load_target(target_dir):
    """读一个靶点的打分与标签。返回 (scores, labels)，读不到则返回 None。"""
    labels_p = os.path.join(target_dir, "saved_labels.npy")
    if not os.path.exists(labels_p):
        return None
    labels = np.load(labels_p)

    preds_p = os.path.join(target_dir, "saved_preds.npy")
    if os.path.exists(preds_p):
        scores = np.load(preds_p)
    else:
        mol_p = os.path.join(target_dir, "saved_mols_embed.npy")
        poc_p = os.path.join(target_dir, "saved_target_embed.npy")
        if not (os.path.exists(mol_p) and os.path.exists(poc_p)):
            return None
        mol_reps = np.load(mol_p)
        pocket_reps = np.load(poc_p)
        # 与官方一致：多口袋取最大
        scores = (pocket_reps @ mol_reps.T).max(axis=0)

    return np.asarray(scores, dtype=float).ravel(), np.asarray(labels).ravel()


def score_benchmark(bench_dir, with_ci=False):
    """算一个 benchmark 下所有靶点的指标。"""
    targets = sorted(
        d for d in os.listdir(bench_dir) if os.path.isdir(os.path.join(bench_dir, d))
    )
    rows = []
    for t in targets:
        loaded = load_target(os.path.join(bench_dir, t))
        if loaded is None:
            print("  [skip] %s: 输出文件不全" % t, file=sys.stderr)
            continue
        scores, labels = loaded
        if len(scores) != len(labels):
            print("  [skip] %s: 长度不匹配 %d vs %d" % (t, len(scores), len(labels)),
                  file=sys.stderr)
            continue

        row = {
            "target": t,
            "n_mol": int(len(labels)),
            "n_active": int(labels.sum()),
            # EF0.1% 是官方实现没有、PPT 要求的那一档
            "EF0.1": enrichment_factor(scores, labels, 0.001),
            "EF0.5": enrichment_factor(scores, labels, 0.005),
            "EF1": enrichment_factor(scores, labels, 0.01),
            "EF5": enrichment_factor(scores, labels, 0.05),
            "BEDROC": bedroc(scores, labels, 80.5),
            "AUROC": roc_auc(scores, labels),
            "Recall@100": top_k_recall(scores, labels, 100),
        }
        if with_ci:
            lo, hi = bootstrap_ci(partial(enrichment_factor, fraction=0.01),
                                  scores, labels, n=500, seed=0)
            row["EF1_CI"] = (lo, hi)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_path", help="如 results/pocket_ranking")
    ap.add_argument("--bootstrap", action="store_true",
                    help="附带 EF1%% 的 95%% 置信区间（慢）")
    ap.add_argument("--out", default=None, help="逐靶点结果写入 json")
    ap.add_argument("--ref", choices=["ensemble", "seq", "drugclip", "bindclip", "auto"], default="auto",
                    help="对标哪一列：ensemble=LigUnity, seq=LigUnity(seq), drugclip=DrugCLIP。"
                         "auto 按 results_path 名字推断")
    args = ap.parse_args()

    ref_key = args.ref
    base = os.path.basename(args.results_path.rstrip("/"))
    if ref_key == "auto":
        if "bindclip" in base:
            ref_key = "bindclip"
        elif "drugclip" in base:
            ref_key = "drugclip"
        elif "protein" in base:
            ref_key = "seq"
        else:
            ref_key = "ensemble"
    ref_table = REFERENCE[ref_key]
    ref_label = {"seq": "LigUnity(seq)", "drugclip": "DrugCLIP",
                 "bindclip": "BindCLIP(论文)", "ensemble": "LigUnity(ensemble)"}[ref_key]
    print("对标列：%s" % ref_label)

    all_rows = {}
    for bench in ["DUDE", "DEKOIS", "PCBA"]:
        bench_dir = os.path.join(args.results_path, bench)
        if not os.path.isdir(bench_dir):
            print("\n== %s: 目录不存在，跳过 ==" % bench)
            continue

        rows = score_benchmark(bench_dir, args.bootstrap)
        all_rows[bench] = rows
        if not rows:
            print("\n== %s: 没有可用结果 ==" % bench)
            continue

        ref = ref_table.get(bench, {})
        print("\n%s" % ("=" * 72))
        print("%s  —  %d 个靶点（作者基准 n=%s，对标 %s）"
              % (bench, len(rows), ref.get("n", "?"), ref_label))
        print("=" * 72)
        print("%-10s %10s %10s %10s" % ("指标", "我们复现", "作者基准", "相对偏差"))
        print("-" * 72)

        for key in ["EF0.1", "EF0.5", "EF1", "EF5", "BEDROC", "AUROC", "Recall@100"]:
            vals = [r[key] for r in rows if not np.isnan(r[key])]
            if not vals:
                continue
            mean = float(np.mean(vals))
            if key in ref:
                dev = (mean - ref[key]) / ref[key] * 100
                print("%-10s %10.4f %10.4f %9.1f%%" % (key, mean, ref[key], dev))
            else:
                mark = "  <- 官方无此档" if key == "EF0.1" else ""
                print("%-10s %10.4f %10s %10s%s" % (key, mean, "-", "-", mark))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(all_rows, f, indent=2, default=float)
        print("\n逐靶点结果已写入 %s" % args.out)


if __name__ == "__main__":
    main()
