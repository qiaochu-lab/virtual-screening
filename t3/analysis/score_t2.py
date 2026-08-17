"""T2 亲和力排序：模型能不能排出**结合强弱**，而不只是分开活性/非活性。

为什么这个任务几乎零成本
------------------------
T3 评测集里每个活性配体都带实测 pAffinity，而各模型对每个分子的打分
我们已经全存下来了。把两者对上算相关性，就是 T2——不用再跑任何模型。

为什么它测的是**质的不同**的能力
--------------------------------
T1/T3 问的是「能不能把活性分子挑出来」，T2 问的是「能不能排出谁强谁弱」。
这类模型的训练目标是对比学习（把结合对拉近、非结合对推远），
**目标函数里根本没有约束亲和力的顺序**。所以完全可能出现
「富集很好但排序接近随机」——这本身就是个值得报告的结论。

只用 active 算（decoy 没有实测值）。每个靶点至少要 10 个 active
才算，少于这个数 Spearman 方差太大。
"""
import argparse
import json
import os

import numpy as np
from scipy import stats

B = "/data/yicheng/xqc/vs-benchmark"
MIN_ACT = 10


def load_truth(layer):
    """uniprot -> {smiles/inchikey 顺序对应的 pAff 列表}。

    评测集写入时的顺序是：先所有 actives 再所有 decoys，
    各模型的 runner 也按同样顺序打分，所以前 n_actives 个分数
    就对应 actives 的 pAff。
    """
    out = {}
    p = f"{B}/data/t3/eval/{layer}.jsonl"
    if not os.path.exists(p):
        return out
    for line in open(p):
        d = json.loads(line)
        out[d["uniprot"]] = [a["paff"] for a in d["actives"]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--layers", nargs="+", default=["L1", "L2", "L3", "L4"])
    ap.add_argument("--out", default=f"{B}/results/t3/summary_t2.json")
    args = ap.parse_args()

    truth = {L: load_truth(L) for L in args.layers}
    summary = {}

    print("T2 亲和力排序（只用 active，模型打分 vs 实测 pAffinity）")
    print("=" * 74)
    print("%-20s %-4s %7s %16s %16s" % ("模型", "层", "靶点", "Spearman ρ", "Kendall τ"))
    print("-" * 74)
    for m in args.models:
        summary[m] = {}
        for L in args.layers:
            d = f"{B}/results/t3/{m}/{L}"
            if not os.path.isdir(d):
                continue
            rho, tau, ns = [], [], []
            for up in sorted(os.listdir(d)):
                pa = truth[L].get(up)
                if not pa or len(pa) < MIN_ACT:
                    continue
                try:
                    s = np.load(f"{d}/{up}/saved_preds.npy")
                    lab = np.load(f"{d}/{up}/saved_labels.npy")
                except Exception:
                    continue
                act = np.nonzero(lab == 1)[0]
                # 长度必须对得上，否则说明顺序假设不成立，宁可跳过
                if len(act) != len(pa):
                    continue
                sc = s[act]
                if np.std(sc) == 0:
                    continue
                r = stats.spearmanr(sc, pa)
                t = stats.kendalltau(sc, pa)
                if np.isfinite(r.statistic):
                    rho.append(r.statistic)
                    tau.append(t.statistic)
                    ns.append(len(pa))
            if not rho:
                continue
            rho, tau = np.array(rho), np.array(tau)
            summary[m][L] = {
                "n_targets": len(rho),
                "spearman": float(rho.mean()),
                "spearman_sem": float(rho.std(ddof=1) / np.sqrt(len(rho))),
                "kendall": float(tau.mean()),
                "frac_positive": float((rho > 0).mean()),
                "median_n_actives": int(np.median(ns)),
            }
            print("%-20s %-4s %7d %16s %16s"
                  % (m, L, len(rho),
                     f"{rho.mean():+.3f}±{rho.std(ddof=1)/np.sqrt(len(rho)):.3f}",
                     f"{tau.mean():+.3f}"))
    print("-" * 74)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=1)

    print("\n各模型「排序方向正确」的靶点占比（ρ>0，随机应为 50%）:")
    for m, d in summary.items():
        cells = [f"{L} {d[L]['frac_positive']*100:.0f}%" for L in args.layers if L in d]
        if cells:
            print(f"  {m:20s} " + "   ".join(cells))
    print("\n注：这类模型的对比学习目标不约束亲和力顺序，")
    print("    所以「富集好但排序接近随机」是可能的，且本身就是结论。")
    print(f"\n已写入 {args.out}")


if __name__ == "__main__":
    main()
