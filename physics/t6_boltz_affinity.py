"""T6 最小动作：Boltz-2 的亲和力预测和实测值有没有相关。

背景
----
T2 已经测出：七个检索类模型的亲和力排序能力**全部接近零**
（Spearman −0.011 ~ +0.129），且已排除数据端解释。
于是 T6 的问题变得具体：**物理方法能不能补上这一块？**

现有材料
--------
建 T3 结构时，Boltz-2 的 affinity 模块顺带算了 929 个预测。
但**每个靶点只有一个代表配体**（建结构时取的亲和力最高那个），
所以做不了「靶点内排序」，只能做**跨靶点的绝对亲和力相关**。

两者测的不是一回事，但跨靶点相关能回答一个前置问题：
Boltz-2 的亲和力输出到底有没有信号？没有的话，后面投入逐配体重算就不划算。

⚠️ 已知的偏差来源
-----------------
1. **方向相反**：`affinity_pred_value` 越小表示结合越强，pAffinity 越大越强。
   算相关前必须统一方向，否则符号是反的。
2. **范围限制**：代表配体都是各靶点亲和力最高的那个，pAff 分布被截断，
   相关系数会被系统性压低。这是设计使然，不是模型的问题，报告时必须说明。
3. Boltz-2 的 affinity 模块不支持 >128 原子的配体，大分子/肽类系统性缺失。
"""
import glob
import json
import os
import re

import numpy as np
from scipy import stats

B = "/data/work/vs-benchmark"


def load_boltz():
    """uniprot -> Boltz-2 亲和力预测。"""
    out = {}
    for d in ["boltz_batch_out", "boltz_retry_out", "boltz_gap_out", "boltz_r2_out"]:
        for p in glob.glob(f"{B}/{d}/**/affinity_*.json", recursive=True):
            up = os.path.basename(p).replace("affinity_", "").replace(".json", "")
            try:
                j = json.load(open(p))
            except Exception:
                continue
            out[up] = j
    return out


def load_truth():
    """建结构时每个靶点用的代表配体 = 该靶点 pAff 最高的那个。这里复原它。"""
    best = {}
    for L in ["L3", "L4"]:
        p = f"{B}/data/t3/layers/{L}.jsonl"
        if not os.path.exists(p):
            continue
        for line in open(p):
            d = json.loads(line)
            try:
                v = float(d["paff"])
            except (TypeError, ValueError):
                continue
            u = d["uniprot"]
            if u not in best or v > best[u][0]:
                best[u] = (v, d["smiles"])
    return best


def main():
    bz = load_boltz()
    truth = load_truth()
    print(f"Boltz-2 亲和力预测: {len(bz):,}")
    print(f"T3 代表配体（各靶点 pAff 最高）: {len(truth):,}")

    ups = sorted(set(bz) & set(truth))
    print(f"可配对: {len(ups):,}\n")
    if len(ups) < 20:
        print("配对太少，无法分析")
        return

    pred = np.array([bz[u]["affinity_pred_value"] for u in ups])
    prob = np.array([bz[u].get("affinity_probability_binary", np.nan) for u in ups])
    true = np.array([truth[u][0] for u in ups])

    print(f"实测 pAff : 中位 {np.median(true):.2f}  范围 {true.min():.2f}–{true.max():.2f}"
          f"  标准差 {true.std():.2f}")
    print(f"Boltz 预测: 中位 {np.median(pred):.3f}  范围 {pred.min():.3f}–{pred.max():.3f}\n")

    print("=" * 62)
    print("跨靶点相关（注意：代表配体都是各靶点最强的，范围受限）")
    print("=" * 62)
    # affinity_pred_value 越小越强，取负号与 pAff 同向
    for name, x in [("affinity_pred_value（已取负号同向）", -pred),
                    ("affinity_probability_binary", prob)]:
        m = np.isfinite(x)
        if m.sum() < 20:
            continue
        r = stats.spearmanr(x[m], true[m])
        pe = stats.pearsonr(x[m], true[m])
        print(f"\n{name}  (n={m.sum()})")
        print(f"  Spearman ρ = {r.statistic:+.3f}   p = {r.pvalue:.2e}")
        print(f"  Pearson  r = {pe.statistic:+.3f}   p = {pe.pvalue:.2e}")

    # 分箱看是否单调 —— 相关系数低也可能是非线性
    print("\n" + "=" * 62)
    print("按 Boltz 预测值分五档，看实测 pAff 是否单调")
    print("=" * 62)
    # affinity_pred_value 越小越强，所以按 pred 降序 = 从最弱到最强
    order = np.argsort(-pred)
    k = len(order) // 5
    labels = ["最弱", "较弱", "中间", "较强", "最强"]
    for i in range(5):
        idx = order[i * k:(i + 1) * k] if i < 4 else order[4 * k:]
        print(f"  第{i+1}档（Boltz 预测{labels[i]}）"
              f" n={len(idx):3d}   实测 pAff 均值 {true[idx].mean():.2f}"
              f" ± {true[idx].std()/np.sqrt(len(idx)):.2f}")

    print("\n" + "=" * 62)
    print("怎么解读")
    print("=" * 62)
    print("· 相关明显 → 物理方法确实能补检索模型排不出强弱的短板，值得投入逐配体重算")
    print("· 相关接近零 → 要么范围限制掩盖了信号，要么亲和力预测本身就难；")
    print("               两种情况都需要在无范围限制的子集上再验一次才能定论")
    print("· 无论哪种结果都值得报——这一点在设计时就说清楚了")


if __name__ == "__main__":
    main()
