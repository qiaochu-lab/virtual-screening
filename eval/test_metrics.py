"""Unit tests for metrics.py.

Verifies correctness against theoretical boundary values: perfect ranking,
random ranking, all-tied, etc. Calibration against real data is handled by
calibrate_against_ligunity.py.
"""
import numpy as np
import pytest

import math

from metrics import (bedroc, bootstrap_ci, enrichment_factor, pr_auc,
                     roc_auc, top_k_recall)


# ---------- enrichment factor ----------

def test_ef_perfect_ranking():
    """All 10 actives ranked first; EF@10% should equal the theoretical ceiling 1/0.1 = 10."""
    scores = np.arange(100, 0, -1, dtype=float)
    labels = np.zeros(100)
    labels[:10] = 1
    assert enrichment_factor(scores, labels, 0.1) == pytest.approx(10.0)


def test_ef_uniform_ranking_is_about_one():
    """When actives are evenly spread out, EF should be close to the random baseline of 1."""
    scores = np.arange(100, 0, -1, dtype=float)
    labels = np.zeros(100)
    labels[::10] = 1
    assert enrichment_factor(scores, labels, 0.1) == pytest.approx(1.0, abs=0.5)


def test_ef_worst_ranking_is_zero():
    """All actives ranked last, none in the top 10% -> EF = 0."""
    scores = np.arange(100, 0, -1, dtype=float)
    labels = np.zeros(100)
    labels[-10:] = 1
    assert enrichment_factor(scores, labels, 0.1) == pytest.approx(0.0)


def test_ef_handles_ties():
    """Should not crash when all scores are tied; result should fall within the valid range."""
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
    """EF@0.1% with only 100 molecules: the top N must be at least 1, avoiding division by zero."""
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
    """AUC should be exactly 0.5 when everything is tied."""
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
    """BEDROC should still fall within [0,1] under random ranking."""
    rng = np.random.default_rng(0)
    scores = rng.random(1000)
    labels = np.zeros(1000)
    labels[:50] = 1
    rng.shuffle(labels)
    result = bedroc(scores, labels, alpha=80.5)
    assert 0.0 <= result <= 1.0


# ---------- top-k recall ----------

def test_top_k_recall():
    """5 actives among the top 10, 10 actives total -> recall = 0.5."""
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
    """The confidence interval should bracket the point estimate."""
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


# ---------------------------------------------------------------- PR-AUC

def test_pr_auc_perfect():
    """All actives ranked first -> AP = 1."""
    scores = [9, 8, 7, 3, 2, 1]
    labels = [1, 1, 1, 0, 0, 0]
    assert abs(pr_auc(scores, labels) - 1.0) < 1e-12


def test_pr_auc_worst():
    """All actives ranked last -> AP takes its minimum possible value."""
    scores = [9, 8, 7, 3, 2, 1]
    labels = [0, 0, 0, 1, 1, 1]
    # hits only start appearing at the very end: precision is 1/4, 2/5, 3/6 in turn
    expect = (1/3) * (1/4) + (1/3) * (2/5) + (1/3) * (3/6)
    assert abs(pr_auc(scores, labels) - expect) < 1e-12


def test_pr_auc_random_baseline():
    """Under random ranking, the expected AP ~= the active fraction (not 0.5).

    This is the baseline for reading PR-AUC, distinct from ROC-AUC's 0.5, and
    must be pinned down explicitly.
    """
    rng = np.random.default_rng(0)
    n, n_act = 2000, 40
    labels = np.zeros(n, dtype=int)
    labels[:n_act] = 1
    vals = []
    for _ in range(200):
        vals.append(pr_auc(rng.random(n), labels))
    assert abs(np.mean(vals) - n_act / n) < 0.005


def test_pr_auc_ties_handled_as_group():
    """All tied -> degenerates to the active fraction, matching the random baseline."""
    labels = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    scores = [5] * 10
    assert abs(pr_auc(scores, labels) - 0.2) < 1e-12


def test_pr_auc_matches_sklearn():
    """Matches sklearn's average_precision_score case by case.

    sklearn is the de facto reference implementation of this metric; a
    mismatch means we are wrong.
    """
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(7)
    for _ in range(30):
        n = int(rng.integers(20, 400))
        labels = (rng.random(n) < rng.uniform(0.02, 0.4)).astype(int)
        if labels.sum() in (0, n):
            continue
        # mix in repeated scores, specifically to test tie handling
        scores = np.round(rng.normal(size=n), 1)
        ours = pr_auc(scores, labels)
        theirs = sk.average_precision_score(labels, scores)
        assert abs(ours - theirs) < 1e-9, (n, ours, theirs)


def test_pr_auc_degenerate():
    assert math.isnan(pr_auc([1, 2, 3], [0, 0, 0]))
    assert math.isnan(pr_auc([1, 2, 3], [1, 1, 1]))
