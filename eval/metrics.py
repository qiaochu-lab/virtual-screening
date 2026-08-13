"""统一评测指标。所有被评测模型共用，保证横评可比。

约定
----
- ``labels``：1 = active，0 = inactive/decoy
- ``scores``：越大表示越可能是 active
- 并列名次统一用「平均秩」处理，避免因各家排序实现不同产生偏差

这套实现的正确性由两层保证：
1. ``test_metrics.py`` 的单元测试（理论边界值）
2. 在 LigUnity 官方输出上复现 Patterns 论文数值（见 calibrate_against_ligunity.py）
"""
import math

import numpy as np
from scipy.stats import rankdata


def _ranks(scores):
    """返回 1-based 排名，分数越高排名越靠前；并列取平均秩。"""
    return rankdata(-np.asarray(scores, dtype=float), method="average")


def enrichment_factor(scores, labels, fraction):
    """EF@fraction：前 fraction 比例中 active 的富集倍数。

    EF = (前 N 名中的 active 数 / N) / (总 active 数 / 总数)

    理论上限为 1/fraction（所有 active 都排在最前面时取到）。

    取整规则用 **ceil**，与 RDKit ``CalcEnrichment`` 一致
    （其源码为 ``numPerFrac = [math.ceil(numMol * f) for f in fractions]``）。
    这一点很容易搞错：改用 round 会在 ``n * fraction`` 非整数时产生偏差，
    实测 DUD-E 上 102 个靶点有 37 个受影响，均值差 0.2%。
    """
    labels = np.asarray(labels)
    n_total = len(labels)
    n_active = int(labels.sum())
    if n_active == 0 or n_total == 0:
        return float("nan")

    n_top = max(1, int(math.ceil(n_total * fraction)))
    ranks = _ranks(scores)
    n_active_top = int(labels[ranks <= n_top].sum())

    return (n_active_top / n_top) / (n_active / n_total)


def roc_auc(scores, labels):
    """ROC AUC。用 Mann-Whitney U 的等价形式，天然正确处理并列。"""
    labels = np.asarray(labels)
    n_active = int(labels.sum())
    n_decoy = len(labels) - n_active
    if n_active == 0 or n_decoy == 0:
        return float("nan")

    ranks = rankdata(np.asarray(scores, dtype=float), method="average")
    return (ranks[labels == 1].sum() - n_active * (n_active + 1) / 2) / (n_active * n_decoy)


def bedroc(scores, labels, alpha=80.5):
    """BEDROC，Truchon & Bayly (2007) 定义。

    alpha=80.5 是虚筛领域惯例，对应「80% 的权重集中在前 2%」。
    返回值归一化到 [0, 1]，1 表示完美早期富集。
    """
    labels = np.asarray(labels)
    n_total = len(labels)
    n_active = int(labels.sum())
    if n_active == 0 or n_active == n_total:
        return float("nan")

    ranks = _ranks(scores)
    ratio = n_active / n_total

    # RIE = 观测到的指数加权富集 / 随机排序的期望值
    rie_sum = np.exp(-alpha * ranks[labels == 1] / n_total).sum()
    rie_random = ratio * (1 - np.exp(-alpha)) / (np.exp(alpha / n_total) - 1)
    rie = rie_sum / rie_random

    # 归一化到 [0,1]
    rie_max = (1 - np.exp(-alpha * ratio)) / (ratio * (1 - np.exp(-alpha)))
    rie_min = (1 - np.exp(alpha * ratio)) / (ratio * (1 - np.exp(alpha)))

    return (rie - rie_min) / (rie_max - rie_min)


def top_k_recall(scores, labels, k):
    """前 k 名中召回的 active 占全部 active 的比例。"""
    labels = np.asarray(labels)
    n_active = int(labels.sum())
    if n_active == 0:
        return float("nan")

    ranks = _ranks(scores)
    return int(labels[ranks <= k].sum()) / n_active


def bootstrap_ci(fn, scores, labels, n=1000, seed=0, ci=0.95):
    """对任意指标做 bootstrap 置信区间。

    参数
    ----
    fn : 形如 ``fn(scores, labels) -> float`` 的可调用对象。
         带额外参数的指标先用 functools.partial 固定，例如
         ``partial(enrichment_factor, fraction=0.01)``。

    返回 (下界, 上界)。
    """
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    size = len(labels)

    vals = []
    for _ in range(n):
        idx = rng.integers(0, size, size)
        v = fn(scores[idx], labels[idx])
        if not np.isnan(v):
            vals.append(v)

    if not vals:
        return (float("nan"), float("nan"))

    lo = (1 - ci) / 2 * 100
    return (float(np.percentile(vals, lo)), float(np.percentile(vals, 100 - lo)))


# ============================================================================
# T2 亲和力排序指标（PPT slide 11：Spearman ρ / R² / pairwise accuracy）
#
# 与 T1 的区别：T1 是「从大库里捞出活性分子」（二分类富集），
# T2 是「同一靶点内，活性强的能否排在活性弱的前面」（连续值排序）。
# 因此这里的 y_true 是实测亲和力（如 pIC50 / ΔG），不是 0/1 标签。
# ============================================================================


def spearman(pred, true):
    """Spearman 秩相关。对单调变换不敏感，是排序任务的主指标。"""
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    if len(pred) < 2 or np.all(pred == pred[0]) or np.all(true == true[0]):
        return float("nan")
    return float(np.corrcoef(rankdata(pred), rankdata(true))[0, 1])


def pearson(pred, true):
    """Pearson 线性相关。"""
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    if len(pred) < 2 or np.all(pred == pred[0]) or np.all(true == true[0]):
        return float("nan")
    return float(np.corrcoef(pred, true)[0, 1])


def r2_score(pred, true):
    """决定系数 R²。

    ⚠️ 注意：这里用的是 **Pearson r 的平方**，不是回归意义上的
    ``1 - SS_res/SS_tot``。虚筛模型输出的是相似度分数而非绝对亲和力，
    量纲不同，用后者会得到无意义的大负数。
    文献中报告 protein-ligand 排序的 R² 通常也指前者，但两者必须区分清楚。
    """
    r = pearson(pred, true)
    return float("nan") if np.isnan(r) else r * r


def pairwise_accuracy(pred, true, tol=0.0):
    """成对排序准确率：任取两个配体，预测的强弱关系与实测一致的比例。

    ``tol``：实测值差异小于该阈值的配体对视为「无法区分」而跳过，
    避免实验误差范围内的配体对稀释指标（FEP 数据常用 tol=0.5 kcal/mol）。
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    n = len(pred)
    if n < 2:
        return float("nan")

    ok = 0
    tot = 0
    for i in range(n - 1):
        dt = true[i + 1:] - true[i]
        dp = pred[i + 1:] - pred[i]
        valid = np.abs(dt) > tol
        tot += int(valid.sum())
        ok += int(((dp * dt) > 0)[valid].sum())

    return ok / tot if tot else float("nan")


def kendall_tau(pred, true):
    """Kendall τ-b。与 pairwise accuracy 同源，但对并列有标准处理。"""
    from scipy.stats import kendalltau
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    if len(pred) < 2:
        return float("nan")
    t = kendalltau(pred, true).correlation
    return float(t) if t is not None else float("nan")
