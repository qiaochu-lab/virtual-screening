"""Unified evaluation metrics. Shared by every model under evaluation, so the
head-to-head comparison stays comparable.

Conventions
----
- ``labels``: 1 = active, 0 = inactive/decoy
- ``scores``: higher means more likely to be active
- Ties are always broken with "average rank", to avoid bias from
  implementation differences across models' own ranking code

Correctness of this implementation rests on two layers:
1. ``test_metrics.py`` unit tests (theoretical boundary values)
2. Reproducing the Patterns paper's numbers on LigUnity's official output
   (see calibrate_against_ligunity.py)
"""
import math

import numpy as np
from scipy.stats import rankdata


def _ranks(scores):
    """Return 1-based ranks, higher score ranks first; ties get the average rank."""
    return rankdata(-np.asarray(scores, dtype=float), method="average")


def enrichment_factor(scores, labels, fraction):
    """EF@fraction: the enrichment factor of actives within the top fraction.

    EF = (actives in the top N / N) / (total actives / total)

    The theoretical ceiling is ``min(1/fraction, n_total/n_active)`` — **not
    1/fraction**. At a 1:50 active:decoy ratio, actives are only 1/51 of the
    pool, so EF@1% can reach at most 51, not 100. An earlier comment wrote
    the ceiling as 1/fraction, which made 39 look like it used only 39% of
    the available range, when it actually used 77%.

    Rounding uses **ceil**, matching RDKit's ``CalcEnrichment``
    (whose source has ``numPerFrac = [math.ceil(numMol * f) for f in
    fractions]``). This is easy to get wrong: switching to round introduces
    bias whenever ``n * fraction`` is not an integer — measured on DUD-E, 37
    of 102 targets are affected, with a 0.2% difference in the mean.

    Ties are handled by **expected value**: a tie group that straddles the
    cutoff only contributes the proportional share of actives it is due. The
    earlier approach used average rank (``labels[ranks <= n_top]``), which
    counted a large tie group straddling the cutoff **in full**, so
    ``n_active_top`` could exceed ``n_top`` and EF could exceed its
    theoretical ceiling — the ligand-only baseline (whose Tanimoto values are
    discrete, so ties are extremely common) measured 51.13 against a ceiling
    of 51.00. Real models' continuous scores are barely affected (largest
    observed difference 0.04, i.e. 0.1%).
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    n_total = len(labels)
    n_active = int(labels.sum())
    if n_active == 0 or n_total == 0:
        return float("nan")

    n_top = max(1, int(math.ceil(n_total * fraction)))
    order = np.argsort(-scores, kind="mergesort")
    s_sorted, y_sorted = scores[order], labels[order]

    got, left, i = 0.0, n_top, 0
    while i < n_total and left > 0:
        j = i
        while j < n_total and s_sorted[j] == s_sorted[i]:
            j += 1
        size = j - i
        act = int(y_sorted[i:j].sum())
        if size <= left:
            got += act
            left -= size
        else:                      # tie group straddles the cutoff, counted proportionally
            got += act * left / size
            left = 0
        i = j

    return (got / n_top) / (n_active / n_total)


def roc_auc(scores, labels):
    """ROC AUC. Uses the Mann-Whitney U equivalent form, which handles ties correctly by construction."""
    labels = np.asarray(labels)
    n_active = int(labels.sum())
    n_decoy = len(labels) - n_active
    if n_active == 0 or n_decoy == 0:
        return float("nan")

    ranks = rankdata(np.asarray(scores, dtype=float), method="average")
    return (ranks[labels == 1].sum() - n_active * (n_active + 1) / 2) / (n_active * n_decoy)


def bedroc(scores, labels, alpha=80.5):
    """BEDROC, as defined by Truchon & Bayly (2007).

    alpha=80.5 is the virtual-screening convention, corresponding to "80% of
    the weight concentrated in the top 2%".
    The return value is normalised to [0, 1], where 1 means perfect early
    enrichment.
    """
    labels = np.asarray(labels)
    n_total = len(labels)
    n_active = int(labels.sum())
    if n_active == 0 or n_active == n_total:
        return float("nan")

    ranks = _ranks(scores)
    ratio = n_active / n_total

    # RIE = observed exponentially-weighted enrichment / expected value under random ranking
    rie_sum = np.exp(-alpha * ranks[labels == 1] / n_total).sum()
    rie_random = ratio * (1 - np.exp(-alpha)) / (np.exp(alpha / n_total) - 1)
    rie = rie_sum / rie_random

    # normalise to [0,1]
    rie_max = (1 - np.exp(-alpha * ratio)) / (ratio * (1 - np.exp(-alpha)))
    rie_min = (1 - np.exp(alpha * ratio)) / (ratio * (1 - np.exp(alpha)))

    return (rie - rie_min) / (rie_max - rie_min)


def pr_auc(scores, labels):
    """PR-AUC (average precision).

    Why this benchmark needs it: EF@fraction is pinned to a single cutoff
    position, and the hit count can only be an integer, so its value is
    quantised into steps by the number of actives — with only 10 actives per
    target, EF@1%'s step size is about 8.5, while the per-layer means are
    only 8-39. Averaging across targets treats this coarse measurement and a
    fine one with equal weight. PR-AUC uses the whole ranking rather than a
    single cutoff, so it does not have this quantisation problem; it is also
    centred on the positive class, making it far more sensitive to an
    imbalance like 1:50 than ROC-AUC (which tends to look inflated under
    heavy imbalance).

    Uses the average-precision form: AP = Sum (R_n - R_{n-1}) * P_n, i.e.
    sklearn's ``average_precision_score``. This is a step-function sum, not
    trapezoidal interpolation — trapezoidal integration overestimates on a
    PR curve.

    Tied scores are handled as "entering together as one group": all samples
    in a group share the precision/recall at the point where that group
    ends, avoiding bias from implementation differences in ranking
    (consistent with the other metrics in this module).

    The expected value under random ranking equals the active fraction, so
    read this metric against that baseline, not against 0.5.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    n_active = int(labels.sum())
    if n_active == 0 or n_active == len(labels):
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    y_sorted = labels[order]

    # tied scores are grouped together, counted as one group
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    # index of the last position in each group
    group_end = np.r_[np.nonzero(np.diff(s_sorted))[0], len(s_sorted) - 1]

    tp_g = tp[group_end]
    fp_g = fp[group_end]
    precision = tp_g / (tp_g + fp_g)
    recall = tp_g / n_active

    # AP = Sum (R_n - R_{n-1}) * P_n, R_0 = 0
    d_recall = np.diff(np.r_[0.0, recall])
    return float((d_recall * precision).sum())


def top_k_recall(scores, labels, k):
    """Fraction of all actives recalled within the top k."""
    labels = np.asarray(labels)
    n_active = int(labels.sum())
    if n_active == 0:
        return float("nan")

    ranks = _ranks(scores)
    return int(labels[ranks <= k].sum()) / n_active


def bootstrap_ci(fn, scores, labels, n=1000, seed=0, ci=0.95):
    """Bootstrap confidence interval for any metric.

    Parameters
    ----
    fn : a callable of the form ``fn(scores, labels) -> float``.
         For metrics with extra arguments, fix them first with
         functools.partial, e.g. ``partial(enrichment_factor, fraction=0.01)``.

    Returns (lower bound, upper bound).
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
# T2 affinity ranking metrics (PPT slide 11: Spearman rho / R^2 / pairwise accuracy)
#
# Difference from T1: T1 is "pulling active molecules out of a large pool"
# (binary-classification enrichment); T2 is "within one target, do the more
# potent ligands rank ahead of the weaker ones" (continuous-value ranking).
# So y_true here is a measured affinity (e.g. pIC50 / dG), not a 0/1 label.
# ============================================================================


def spearman(pred, true):
    """Spearman rank correlation. Insensitive to monotonic transforms, the primary metric for ranking tasks."""
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    if len(pred) < 2 or np.all(pred == pred[0]) or np.all(true == true[0]):
        return float("nan")
    return float(np.corrcoef(rankdata(pred), rankdata(true))[0, 1])


def pearson(pred, true):
    """Pearson linear correlation."""
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    if len(pred) < 2 or np.all(pred == pred[0]) or np.all(true == true[0]):
        return float("nan")
    return float(np.corrcoef(pred, true)[0, 1])


def r2_score(pred, true):
    """Coefficient of determination R^2.

    Note: this uses the **square of Pearson r**, not the regression
    definition ``1 - SS_res/SS_tot``. Virtual-screening models output
    similarity scores rather than absolute affinities — the units differ,
    and using the regression form produces meaningless large negative
    numbers. The literature reporting R^2 for protein-ligand ranking usually
    also means the former, but the two must be kept distinct.
    """
    r = pearson(pred, true)
    return float("nan") if np.isnan(r) else r * r


def pairwise_accuracy(pred, true, tol=0.0):
    """Pairwise ranking accuracy: for any two ligands, the fraction where the
    predicted stronger/weaker relationship agrees with the measured one.

    ``tol``: ligand pairs whose measured difference is below this threshold
    are treated as "indistinguishable" and skipped, so pairs within
    experimental error don't dilute the metric (FEP data commonly uses
    tol=0.5 kcal/mol).
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
    """Kendall tau-b. Related to pairwise accuracy, but with a standard treatment of ties."""
    from scipy.stats import kendalltau
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    if len(pred) < 2:
        return float("nan")
    t = kendalltau(pred, true).correlation
    return float(t) if t is not None else float("nan")
