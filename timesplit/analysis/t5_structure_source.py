"""T5-a：同一批靶点上，用实验 holo 结构 vs Boltz-2 预测结构，结果差多少。

这是 T5「结构鲁棒性」的核心问题之一：模型对结构来源有多敏感？
材料是现成的——组装 T3 数据时每个靶点用的是哪种结构已逐靶点记录在
data/T3_6A/manifest.json 里，直接按来源切一刀即可。

注意这**不是**随机对照：一个靶点有没有实验结构本身就不随机
（研究得多的靶点才有），所以两组的靶点难度本来就可能不同。
阴性对照是两个**纯序列模型**（ConPLex、LigUnity-protein）——它们完全不用结构，
若它们在两组间也有同样的差距，说明差距来自靶点本身而非结构来源。

⚠️ 必须显式传 --models
--------------------
早期版本对 summary.json 里的模型做 `for m in sorted(s)` 遍历。那个文件每跑一次
score_t3.py 就被覆盖成当次的模型集，于是这张表报了哪些模型取决于上一条命令跑了
什么——文档里曾因此只报了 BindCLIP 两个模型、得出「无显著差异」，而跑全十个模型
时有四个在 L4 上显著。现在必须显式传模型名，缺谁就报缺谁，不再静默漏报。
"""
import argparse
import csv
import json
import os

import numpy as np
from scipy import stats

B = "/data/work/vs"
ALL = ["drugclip", "bindclip_randneg", "bindclip_hardneg",
       "ligunity_pocket_ranking", "ligunity_protein_ranking", "litenclip",
       "hypseek_rk", "conglude", "conplex", "sprint"]
# 不使用结构的模型，用作阴性对照
SEQ_ONLY = {"conplex", "ligunity_protein_ranking"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=ALL,
                    help="要报的模型；默认全部十个。缺失的会明确报出来。")
    ap.add_argument("--summary", default=f"{B}/results/t3/summary.json")
    ap.add_argument("--out", default=f"{B}/results/export/T5_structure_source.csv")
    args = ap.parse_args()

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
            r = s[m][L]["per_target"]
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

    # 多重比较：BH 步进
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
        sign = 2 * sum(comb(len(direction), i)
                       for i in range(k, len(direction) + 1)) / 2 ** len(direction)
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
