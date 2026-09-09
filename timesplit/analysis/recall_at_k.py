"""Recall@K：前 K 名里捞回了这个靶点多少比例的活性。

为什么要它
----------
EF@1% 是**比例**指标——它回答「前 1% 里活性的浓度是随机的多少倍」，
不回答「一次筛选能拿到多少个真阳性」。而实际做湿实验时买得起的化合物数
是**绝对数**（几十到几百个），不是候选池的百分比。两者在候选池大小差异很大时
会给出完全不同的排序：一个 1,784 分子的靶点，top-1% 只有 18 个位置；
一个 16 万分子的靶点，top-1% 有 1,600 个位置。

Recall@100 = 前 100 名里的活性数 / 该靶点活性总数。
Hit@100 = 前 100 名里的活性**个数**——就是订 100 个化合物能拿到几个真阳性，
这是三个数里最接近湿实验决策的一个。

⚠️ **固定 K 不是「跨靶点可比」的万能解。** 前 100 名占池子的比例，
在 1,784 个分子的靶点上是 5.6%，在 16 万个分子的靶点上是 0.06%——
池子小的靶点天然占便宜。各层的池子中位数本来就不同（L1 5,070、L4 7,743），
所以 Recall@100 的层间差距里混了一部分池子大小效应。
因此同时给出 **EF@100**：

    EF@100 = Hit@100 / (100 × 活性占比)

它把基线除掉了，1.0 = 和随机一样，可以和 EF@1% 直接对照。
Hit@100 回答「能拿到几个」，EF@100 回答「比瞎猜好多少倍」，两个都要看。
"""
import argparse
import collections
import csv
import json
import os

import numpy as np

B = "/data/work/vs-benchmark"
KS = [100, 500, 1000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=f"{B}/results/t3_raw")
    ap.add_argument("--models", nargs="+",
                    default=["ligunity_protein_ranking", "ligunity_pocket_ranking",
                             "hypseek_rk", "litenclip", "drugclip",
                             "bindclip_randneg", "bindclip_hardneg", "conglude",
                             "conplex", "sprint"])
    ap.add_argument("--layers", nargs="+", default=["L1", "L2", "L3", "L4"])
    ap.add_argument("--subset", default=f"{B}/results/export/T3_vsds_matched.csv")
    ap.add_argument("--out", default=f"{B}/results/export/T3_recall_at_k.csv")
    args = ap.parse_args()

    keep = None
    if args.subset:
        keep = {(r["layer"], r["uniprot"]) for r in csv.DictReader(open(args.subset))}
        print(f"子集过滤：{len(keep)} 条")

    rows = [["model", "layer", "n_targets", "median_pool", "median_actives"]
            + [f"recall_at_{k}" for k in KS] + ["hit_at_100", "ef_at_100"]]
    print("\nRecall@K（前 K 名里捞回该靶点多少比例的活性）")
    print("=" * 108)
    print("%-26s %-4s %6s %9s %8s %9s %9s %9s %9s %9s"
          % ("模型", "层", "靶点", "池中位", "活性中位",
             "R@100", "R@500", "R@1000", "命中@100", "EF@100"))
    print("-" * 108)
    for m in args.models:
        for L in args.layers:
            d = f"{args.raw}/{m}/T3/{L}"
            if not os.path.isdir(d):
                continue
            rec = collections.defaultdict(list)
            pools, nacts, hits, efs = [], [], [], []
            for up in sorted(os.listdir(d)):
                if keep is not None and (L, up) not in keep:
                    continue
                try:
                    p = np.load(f"{d}/{up}/saved_preds.npy").reshape(-1)
                    y = np.load(f"{d}/{up}/saved_labels.npy")
                except Exception:
                    continue
                if len(p) != len(y) or y.sum() == 0:
                    continue
                # Recall@K 只依赖打分和标签的配对，不依赖分子身份，
                # 所以不需要还原 SMILES 顺序——lmdb 游标序的坑在这里不存在。
                order = np.argsort(-p)
                ys = y[order]
                na = int(y.sum())
                pools.append(len(y))
                nacts.append(na)
                for k in KS:
                    kk = min(k, len(y))
                    rec[k].append(float(ys[:kk].sum()) / na)
                kk = min(100, len(y))
                h = float(ys[:kk].sum())
                hits.append(h)
                efs.append((h / kk) / (na / len(y)))     # EF@100，基线已除掉
            if len(pools) < 5:
                continue
            vals = [float(np.mean(rec[k])) for k in KS]
            print("%-26s %-4s %6d %9.0f %8.0f %8.1f%% %8.1f%% %8.1f%% %9.1f %9.1f"
                  % (m, L, len(pools), np.median(pools), np.median(nacts),
                     100 * vals[0], 100 * vals[1], 100 * vals[2],
                     np.mean(hits), np.mean(efs)))
            rows.append([m, L, len(pools), f"{np.median(pools):.0f}",
                         f"{np.median(nacts):.0f}"]
                        + [f"{v:.4f}" for v in vals]
                        + [f"{np.mean(hits):.2f}", f"{np.mean(efs):.2f}"])
        print("-" * 108)
    print("\n命中@100 = 订 100 个化合物平均能拿到几个真阳性——最接近湿实验决策。")
    print("EF@100 把活性基线除掉了，1.0 = 和随机一样；它和 EF@1% 可以直接对照，")
    print("两者的差别只在截断线取「固定 100 个」还是「池子的 1%」。")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\n写入 {args.out}")


if __name__ == "__main__":
    main()
