"""metrics.py 的单元测试。

用理论边界值验证实现正确性：完美排序、随机排序、全并列等。
真实数据上的校准由 calibrate_against_ligunity.py 负责。
"""
import numpy as np
import pytest

from metrics import bedroc, bootstrap_ci, enrichment_factor, roc_auc, top_k_recall


# ---------- enrichment factor ----------

def test_ef_perfect_ranking():
    """10 个 active 全排最前，EF@10% 应等于理论上限 1/0.1 = 10。"""
    scores = np.arange(100, 0, -1, dtype=float)
    labels = np.zeros(100)
    labels[:10] = 1
    assert enrichment_factor(scores, labels, 0.1) == pytest.approx(10.0)


def test_ef_uniform_ranking_is_about_one():
    """active 均匀散布时 EF 应接近随机基线 1。"""
    scores = np.arange(100, 0, -1, dtype=float)
    labels = np.zeros(100)
    labels[::10] = 1
    assert enrichment_factor(scores, labels, 0.1) == pytest.approx(1.0, abs=0.5)


def test_ef_worst_ranking_is_zero():
    """所有 active 排最后，前 10% 一个都没有 → EF = 0。"""
    scores = np.arange(100, 0, -1, dtype=float)
    labels = np.zeros(100)
    labels[-10:] = 1
    assert enrichment_factor(scores, labels, 0.1) == pytest.approx(0.0)


def test_ef_handles_ties():
    """全部同分时不应崩溃，结果落在合法区间内。"""
    scores = np.ones(100)
    labels = np.zeros(100)
    labels[:10] = 1
    result = enrichment_factor(scores, labels, 0.1)
    assert 0.0 <= result <= 10.0


def test_ef_no_actives_returns_nan():
    scores = np.arange(100, 0, -1, dtype=float)
    labels = np.zeros(100)
    assert np.isnan(enrichment_factor(scores, labels, 0.1))


def test_ef_small_fraction_rounds_to_at_least_one():
    """EF@0.1% 在只有 100 个分子时，前 N 名至少取 1 个，不能除零。"""
    scores = np.arange(100, 0, -1, dtype=float)
    labels = np.zeros(100)
    labels[0] = 1
    result = enrichment_factor(scores, labels, 0.001)
    assert result == pytest.approx(100.0)


# ---------- ROC AUC ----------

def test_roc_auc_perfect_is_one():
    scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    labels = np.array([1, 1, 0, 0, 0])
    assert roc_auc(scores, labels) == pytest.approx(1.0)


def test_roc_auc_inverted_is_zero():
    scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    labels = np.array([0, 0, 0, 1, 1])
    assert roc_auc(scores, labels) == pytest.approx(0.0)


def test_roc_auc_all_tied_is_half():
    """全并列时 AUC 应正好是 0.5。"""
    scores = np.ones(10)
    labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    assert roc_auc(scores, labels) == pytest.approx(0.5)


# ---------- BEDROC ----------

def test_bedroc_perfect_is_near_one():
    scores = np.arange(1000, 0, -1, dtype=float)
    labels = np.zeros(1000)
    labels[:10] = 1
    assert bedroc(scores, labels, alpha=80.5) > 0.99


def test_bedroc_worst_is_near_zero():
    scores = np.arange(1000, 0, -1, dtype=float)
    labels = np.zeros(1000)
    labels[-10:] = 1
    assert bedroc(scores, labels, alpha=80.5) < 0.01


def test_bedroc_in_unit_interval():
    """随机排序下 BEDROC 仍应落在 [0,1]。"""
    rng = np.random.default_rng(0)
    scores = rng.random(1000)
    labels = np.zeros(1000)
    labels[:50] = 1
    rng.shuffle(labels)
    result = bedroc(scores, labels, alpha=80.5)
    assert 0.0 <= result <= 1.0


# ---------- top-k recall ----------

def test_top_k_recall():
    """前 10 名里有 5 个 active，总共 10 个 active → recall = 0.5。"""
    scores = np.arange(100, 0, -1, dtype=float)
    labels = np.zeros(100)
    labels[:5] = 1
    labels[50:55] = 1
    assert top_k_recall(scores, labels, 10) == pytest.approx(0.5)


def test_top_k_recall_full():
    scores = np.arange(100, 0, -1, dtype=float)
    labels = np.zeros(100)
    labels[:5] = 1
    assert top_k_recall(scores, labels, 10) == pytest.approx(1.0)


# ---------- bootstrap ----------

def test_bootstrap_ci_brackets_point_estimate():
    """置信区间应包含点估计值。"""
    from functools import partial

    rng = np.random.default_rng(0)
    scores = rng.random(500)
    labels = np.zeros(500)
    labels[:50] = 1
    rng.shuffle(labels)

    ef = partial(enrichment_factor, fraction=0.1)
    point = ef(scores, labels)
    lo, hi = bootstrap_ci(ef, scores, labels, n=200, seed=0)

    assert lo <= point <= hi


def test_bootstrap_ci_is_deterministic_given_seed():
    from functools import partial

    scores = np.arange(200, 0, -1, dtype=float)
    labels = np.zeros(200)
    labels[:20] = 1

    ef = partial(enrichment_factor, fraction=0.1)
    a = bootstrap_ci(ef, scores, labels, n=100, seed=42)
    b = bootstrap_ci(ef, scores, labels, n=100, seed=42)
    assert a == b
