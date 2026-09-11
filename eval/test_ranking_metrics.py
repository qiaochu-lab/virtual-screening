"""Tests for the T2 ranking metrics.

Lesson from last time: synthetic data that only covers the "happy path" is
equivalent to not testing at all. This deliberately includes edge cases —
perfect / inverted / tied / all-identical / single-point — plus cross-checks
against scipy.
"""
import numpy as np
import pytest
from scipy.stats import spearmanr, kendalltau as sp_kendall

from metrics import (kendall_tau, pairwise_accuracy, pearson, r2_score,
                     spearman)


# ---------- Spearman ----------

def test_spearman_perfect():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_spearman_inverted():
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_monotonic_invariance():
    """Spearman is insensitive to monotonic transforms — this is its key property relative to Pearson."""
    true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = np.array([0.1, 0.5, 0.9, 2.0, 9.0])
    assert spearman(pred, true) == pytest.approx(spearman(np.exp(pred), true))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_spearman_matches_scipy(seed):
    rng = np.random.default_rng(seed)
    true = rng.normal(size=40)
    pred = true + rng.normal(scale=0.7, size=40)
    assert spearman(pred, true) == pytest.approx(spearmanr(pred, true).correlation, abs=1e-9)


def test_spearman_degenerate_returns_nan():
    assert np.isnan(spearman([1.0, 1.0, 1.0], [1, 2, 3]))
    assert np.isnan(spearman([5.0], [3.0]))


# ---------- R² ----------

def test_r2_is_squared_pearson_not_regression_r2():
    """Confirm R^2 uses the square of Pearson r, not 1 - SS_res/SS_tot.

    Model output similarity scores are on a different scale from measured
    affinities; the regression form of R^2 would produce large negative
    numbers, so it must be the former here.
    """
    true = np.array([1.0, 2.0, 3.0, 4.0])
    pred = true * 100 + 50          # perfectly linear but on a 100x different scale
    assert r2_score(pred, true) == pytest.approx(1.0)
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    assert 1 - ss_res / ss_tot < -1000        # the regression form of R^2 is meaningless in this scenario


def test_r2_in_unit_interval():
    rng = np.random.default_rng(0)
    for _ in range(5):
        true = rng.normal(size=30)
        pred = rng.normal(size=30)
        v = r2_score(pred, true)
        assert 0.0 <= v <= 1.0


# ---------- pairwise accuracy ----------

def test_pairwise_perfect_and_inverted():
    true = [1.0, 2.0, 3.0, 4.0]
    assert pairwise_accuracy([1, 2, 3, 4], true) == pytest.approx(1.0)
    assert pairwise_accuracy([4, 3, 2, 1], true) == pytest.approx(0.0)


def test_pairwise_counts_all_pairs():
    """4 ligands should give C(4,2)=6 pairs; the prediction swaps the last two -> 5/6."""
    true = [1.0, 2.0, 3.0, 4.0]
    pred = [1.0, 2.0, 4.0, 3.0]
    assert pairwise_accuracy(pred, true) == pytest.approx(5 / 6)


def test_pairwise_tol_skips_indistinguishable_pairs():
    """Ligand pairs within tol should be skipped, not counted as errors."""
    true = [1.0, 1.2, 5.0]        # the first two differ by 0.2, within tol=0.5
    pred = [2.0, 1.0, 9.0]        # the first two are predicted in reversed order
    assert pairwise_accuracy(pred, true, tol=0.0) == pytest.approx(2 / 3)
    assert pairwise_accuracy(pred, true, tol=0.5) == pytest.approx(1.0)


def test_pairwise_all_tied_returns_nan():
    assert np.isnan(pairwise_accuracy([1, 2, 3], [7.0, 7.0, 7.0], tol=0.0))


# ---------- Kendall ----------

@pytest.mark.parametrize("seed", [0, 1])
def test_kendall_matches_scipy(seed):
    rng = np.random.default_rng(seed)
    true = rng.normal(size=25)
    pred = true + rng.normal(scale=0.5, size=25)
    assert kendall_tau(pred, true) == pytest.approx(sp_kendall(pred, true).correlation, abs=1e-9)


def test_kendall_equals_pairwise_when_no_ties():
    """With no ties, Kendall tau and pairwise accuracy satisfy tau = 2*acc - 1."""
    rng = np.random.default_rng(3)
    true = rng.normal(size=20)
    pred = rng.normal(size=20)
    assert kendall_tau(pred, true) == pytest.approx(2 * pairwise_accuracy(pred, true) - 1, abs=1e-9)
