"""LigUnity 序列版 vs 口袋版：逐靶点配对比较。

为什么这是个受控对照
--------------------
LigUnity 一次发布里给了两条平行分支：

    LigUnity_VS/pocket_ranking_vs/checkpoint_avg_41-50.pt    蛋白侧 = 3D 口袋
    LigUnity_VS/protein_ranking_vs/checkpoint_avg_41-50.pt   蛋白侧 = 氨基酸序列

训练数据（PocketAffDB）、分子侧编码器（3D 构象）、存档点挑法（41–50 轮平均）
都相同，只有蛋白侧的表示不同。两个权重在同一批靶点、同一批候选分子上评测，
所以可以逐靶点配对。

为什么必须做配对检验而不是只比均值
----------------------------------
EF@1% 在每靶点上是个很粗的量（候选池中位约 1,200，前 1% 只有 12 个位置），
大量靶点上两个模型取值完全相同。只看均值会被少数差异大的靶点主导，
掩盖掉「多数靶点上其实没差别」。本脚本把胜/负/平分开报，并对去掉平局后的
胜率和 Wilcoxon 配对检验都给出结果。

用法
----
    python seq_vs_pocket.py [--raw 原始打分目录] [--out 输出CSV]

需要 T3_ligunity_pocket_ranking.npz 和 T3_ligunity_protein_ranking.npz。
"""
import argparse
import collections
import csv
import math
import os

import numpy as np
from scipy import stats

POCKET = "T3_ligunity_pocket_ranking"
SEQ = "T3_ligunity_protein_ranking"


def load(raw_dir, name):
    z = np.load(os.path.join(raw_dir, f"{name}.npz"))
    keys = {k.rsplit("/", 1)[0] for k in z.files}
    return {t: (z[f"{t}/preds"], z[f"{t}/labels"]) for t in keys}


def enrichment_factor(y, s, frac):
    """EF@frac。截断用 math.ceil，与 RDKit 的 CalcEnrichment 一致。"""
    n = len(y)
    k = math.ceil(n * frac)
    if k < 1 or y.sum() == 0:
        return None
    return (y[np.argsort(-s)][:k].sum() / k) / (y.sum() / n)


def auroc(y, s):
    if y.sum() < 1 or (y == 0).sum() < 1:
        return None
    u = stats.mannwhitneyu(s[y == 1], s[y == 0], alternative="greater").statistic
    return u / ((y == 1).sum() * (y == 0).sum())


METRICS = (("EF1%", lambda y, s: enrichment_factor(y, s, 0.01)),
           ("EF5%", lambda y, s: enrichment_factor(y, s, 0.05)),
           ("AUROC", auroc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw_scores", help="存放 npz 的目录")
    ap.add_argument("--out", default="seq_vs_pocket_per_target.csv")
    args = ap.parse_args()

    P = load(args.raw, POCKET)
    S = load(args.raw, SEQ)
    common = sorted(set(P) & set(S))
    print(f"两个权重都跑过的靶点: {len(common)}")

    rows = [["target", "layer", "n_molecules", "n_actives", "metric",
             "pocket", "sequence", "diff"]]
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    mismatch = 0
    for t in common:
        parts = t.split("/")
        layer = parts[1] if len(parts) > 2 else "?"
        up = parts[-1]
        yp, ys = P[t][1], S[t][1]
        if not np.array_equal(yp, ys):
            mismatch += 1        # 标签不一致说明不是同一批候选，不能配对
            continue
        for nm, fn in METRICS:
            a, b = fn(yp, P[t][0]), fn(ys, S[t][0])
            if a is None or b is None:
                continue
            by[layer][nm].append((a, b))
            rows.append([up, layer, len(yp), int(yp.sum()), nm,
                         f"{a:.4f}", f"{b:.4f}", f"{b - a:+.4f}"])
    if mismatch:
        print(f"（{mismatch} 个靶点两边标签不一致，已剔除）")

    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    print(f"\n{'层':<4} {'指标':<7} {'靶点':>5} {'序列赢':>7} {'口袋赢':>7} {'平局':>6} "
          f"{'去平局胜率':>10} {'均值差':>9} {'中位差':>9} {'p(配对)':>10}")
    print("-" * 84)
    for L in ("L1", "L2", "L3", "L4"):
        for nm, _ in METRICS:
            v = by[L].get(nm)
            if not v:
                continue
            a = np.array([x[0] for x in v])
            b = np.array([x[1] for x in v])
            d = b - a
            w = int((d > 1e-9).sum())
            l = int((d < -1e-9).sum())
            tie = len(d) - w - l
            try:
                p = stats.wilcoxon(a, b).pvalue
            except ValueError:
                p = float("nan")
            print(f"{L:<4} {nm:<7} {len(d):>5} {w:>7} {l:>7} {tie:>6} "
                  f"{100*w/max(w+l,1):>9.1f}% {d.mean():>+9.4f} "
                  f"{np.median(d):>+9.4f} {p:>10.2g}")
        print()

    print("怎么读")
    print("· 平局多是因为 EF@1% 很粗：候选池中位约 1,200，前 1% 只有 12 个位置，")
    print("  两个模型经常取到相同值。所以「去平局胜率」比「序列赢的绝对个数」更有意义。")
    print("· Wilcoxon 同时考虑差值的符号和大小，均值差为正但 p 不显著，")
    print("  意味着优势由少数靶点贡献，不是普遍现象。")
    print(f"\n逐靶点结果写入 {args.out}")


if __name__ == "__main__":
    main()
