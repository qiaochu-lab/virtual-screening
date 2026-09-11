"""T6's minimal action: does Boltz-2's affinity prediction correlate with the measured value.

Background
------------
T2 already found that all seven retrieval-family models' affinity-ranking
ability is **near zero across the board** (Spearman -0.011 to +0.129), and
data-side explanations have been ruled out. So T6's question becomes
concrete: **can a physics method fill this gap?**

Material already on hand
----------------------------
When the T3 structures were built, Boltz-2's affinity module incidentally
computed 929 predictions. But **each target has only one representative
ligand** (the one with the highest affinity, picked while building the
structure), so "within-target ranking" cannot be done -- only **absolute
affinity correlation across targets**.

The two measure different things, but the cross-target correlation answers
a prerequisite question: does Boltz-2's affinity output carry any signal at
all? If not, investing further in a per-ligand rerun would not be worth it.

Warning: known sources of bias
----------------------------------
1. **Opposite direction**: `affinity_pred_value` is smaller for stronger
   binding, while pAffinity is larger for stronger binding. The direction
   must be aligned before computing correlation, or the sign comes out
   reversed.
2. **Restriction of range**: every representative ligand is each target's
   highest-affinity one, so the pAff distribution is truncated, which
   systematically depresses the correlation coefficient. This is by design,
   not a model problem, and must be stated when reporting.
3. Boltz-2's affinity module does not support ligands with >128 atoms, so
   large molecules/peptide-like systems are systematically missing.
"""
import glob
import json
import os
import re

import numpy as np
from scipy import stats

B = "/data/work/vs-benchmark"


def load_boltz():
    """uniprot -> Boltz-2 affinity prediction."""
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
    """The representative ligand used per target when building the structure = the one with the highest pAff for that target. Recovered here."""
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
    # affinity_pred_value is smaller for stronger binding; sign flipped to align with pAff's direction
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

    # bin and check for monotonicity -- a low correlation coefficient could also mean a nonlinear relationship
    print("\n" + "=" * 62)
    print("按 Boltz 预测值分五档，看实测 pAff 是否单调")
    print("=" * 62)
    # affinity_pred_value is smaller for stronger binding, so descending order by pred = weakest to strongest
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
