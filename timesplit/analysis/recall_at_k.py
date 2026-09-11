"""Recall@K: what fraction of a target's actives are recovered in the top K.

Why we need it
----------
EF@1% is a **proportional** metric — it answers "how many times more
concentrated are the actives in the top 1% than random", not "how many true
positives does one screening campaign actually retrieve". In real wet-lab
work, the number of compounds you can afford to buy is an **absolute count**
(tens to a few hundred), not a percentage of the candidate pool. The two give
completely different rankings when pool sizes vary widely: a target with
1,784 molecules has only 18 slots in its top 1%, while a target with 160,000
molecules has 1,600.

Recall@100 = number of actives in the top 100 / total actives for that target.
Hit@100 = the **count** of actives in the top 100 — i.e. how many true
positives you get for ordering 100 compounds. Of the three numbers, this is
the one closest to an actual wet-lab decision.

⚠️ **A fixed K is not a universal fix for cross-target comparability.** The
top 100 is 5.6% of the pool for a 1,784-molecule target but only 0.06% for a
160,000-molecule target — small-pool targets have a built-in advantage. The
layers already have different median pool sizes (L1 5,070, L4 7,743), so
some of the layer-to-layer gap in Recall@100 is really a pool-size effect.
Hence also reporting **EF@100**:

    EF@100 = Hit@100 / (100 × active fraction)

This divides out the baseline, so 1.0 = same as random, and it can be
compared directly against EF@1%. Hit@100 answers "how many can I get",
EF@100 answers "how many times better than guessing" — both matter.
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
                # Recall@K depends only on the (score, label) pairing, not molecule
                # identity, so there's no need to restore SMILES order — the lmdb
                # cursor-order pitfall doesn't apply here.
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
                efs.append((h / kk) / (na / len(y)))     # EF@100, baseline already divided out
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
