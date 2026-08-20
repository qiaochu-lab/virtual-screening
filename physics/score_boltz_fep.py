"""Boltz-2 在 FEP 16 个体系上的逐配体排序能力（可在跑到一半时先看）。

口径
----
`affinity_pred_value` **越小表示结合越强**，实测 act (pAffinity) 越大越强，
所以取负号同向后再算相关——不统一方向的话符号是反的。

为什么中途也能看
----------------
排序指标是**逐体系**算的：某个体系的配体全跑完了，它那一行就是终值，
不会因为别的体系还没跑完而改变。只有「跨体系平均」那一行会随覆盖率变。
所以这里逐体系报覆盖率，覆盖不足的单独标出来，不混进平均。

与检索模型同口径对照
--------------------
右侧几列是 results/fep/ 下各检索模型在**同一体系、同一批配体**上的 Spearman，
由 score_fep.py 算出。这是 T6 想要的那张表——三类方法、同一批体系。
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

B = "/data/work/vs-benchmark"
FEP = f"{B}/code/LigUnity/test_datasets/FEP"
MIN_N, MIN_COV = 8, 0.8


def truth():
    """体系 -> [实测 pAffinity]（顺序与 prep_boltz_fep.py 生成输入时一致）。"""
    out = {}
    for e in json.load(open(f"{FEP}/fep_labels.json")):
        out[e["pockets"][0]] = [l["act"] for l in e["ligands"]]
    return out


def preds():
    """体系 -> {配体序号: 预测值}。文件名形如 affinity_tnks2__015.json。"""
    out = defaultdict(dict)
    for p in glob.glob(f"{B}/boltz_fep_out/shard_*/*/predictions/*/affinity_*.json"):
        name = os.path.basename(p)[len("affinity_"):-len(".json")]
        if "__" not in name:
            continue
        sysname, idx = name.rsplit("__", 1)
        try:
            out[sysname][int(idx)] = json.load(open(p))["affinity_pred_value"]
        except Exception:
            pass
    return out


def retrieval():
    """体系 -> {模型: Spearman}，来自检索模型已落盘的打分。"""
    out = defaultdict(dict)
    root = f"{B}/results/fep"
    if not os.path.isdir(root):
        return out
    for m in sorted(os.listdir(root)):
        d = f"{root}/{m}/FEP"
        if not os.path.isdir(d):
            continue
        for t in os.listdir(d):
            try:
                p = np.load(f"{d}/{t}/saved_preds.npy")
                y = np.load(f"{d}/{t}/saved_labels.npy")
            except Exception:
                continue
            if len(p) == len(y) and np.std(p) > 0:
                out[t][m] = stats.spearmanr(p, y).statistic
    return out


KEN = {}


def main():
    T, P, R = truth(), preds(), retrieval()
    global KEN
    models = sorted({m for v in R.values() for m in v})
    hdr = "".join(f"{m[:16]:>17s}" for m in models)
    print("Boltz-2 逐配体 vs 检索模型（同一批体系、同一批配体）")
    print("=" * (46 + 17 * len(models)))
    print("%-11s %6s %8s %10s%s" % ("体系", "n", "覆盖率", "Boltz-2 ρ", hdr))
    print("-" * (46 + 17 * len(models)))

    done, partial = [], []
    for s in sorted(T):
        y_all = T[s]
        got = P.get(s, {})
        idx = sorted(i for i in got if 0 <= i < len(y_all))
        cov = len(idx) / len(y_all) if y_all else 0
        if len(idx) < MIN_N:
            print("%-11s %6d %7.0f%% %10s" % (s, len(y_all), cov * 100, "跑得太少"))
            continue
        yv = np.array([y_all[i] for i in idx])
        pv = -np.array([got[i] for i in idx])          # 取负号同向
        rho = stats.spearmanr(pv, yv).statistic
        KEN[s] = stats.kendalltau(pv, yv).statistic
        cells = "".join((f"{R[s][m]:+.3f}".rjust(17) if m in R.get(s, {}) else "—".rjust(17))
                        for m in models)
        flag = "" if cov >= MIN_COV else "  ← 覆盖不足，未计入平均"
        print("%-11s %6d %7.0f%% %+10.3f%s%s" % (s, len(idx), cov * 100, rho, cells, flag))
        (done if cov >= MIN_COV else partial).append((s, rho))

    print("-" * (46 + 17 * len(models)))
    if done:
        r = np.array([x[1] for x in done])
        print(f"\nBoltz-2：{len(done)}/{len(T)} 个体系覆盖率 ≥{MIN_COV:.0%}")
        print(f"  平均 Spearman {r.mean():+.3f}   中位 {np.median(r):+.3f}   "
              f"方向正确 {int((r > 0).sum())}/{len(r)}")
        for m in models:
            v = [R[s][m] for s, _ in done if m in R.get(s, {})]
            if v:
                print(f"  同口径对照 {m:<26s} 平均 {np.mean(v):+.3f}（n={len(v)}）")
    if done:
        k = np.array([KEN[s] for s, _ in done if s in KEN])
        print(f"\n  Kendall τ 平均 {k.mean():+.3f}   中位 {np.median(k):+.3f}"
              "    ← 与 Uni-FEP 文献值 0.503 同口径")
        for m in models:
            wins = sum(1 for s, r in done
                       if m in R.get(s, {}) and R[s][m] > r)
            print(f"  {m:<26s} 在 {wins}/{len(done)} 个体系上反超 Boltz-2")
    if partial:
        print(f"\n还没跑完的 {len(partial)} 个体系（数字仅供参考，会变）: "
              + ", ".join(f"{s} {r:+.2f}" for s, r in partial))
    print("\n注：Boltz-2 的亲和力模块与检索模型口径不同——它输出的是预测亲和力，"
          "检索模型输出的是余弦相似度；这里比的是**排序能力**，不是数值本身。")


if __name__ == "__main__":
    main()
