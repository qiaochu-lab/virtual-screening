"""T2 排序指标的测试。

上次的教训：合成数据若只覆盖「好情况」，等于没测。
这里刻意包含完美/反向/并列/全同/单点等边界，以及与 scipy 的交叉验证。
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
    """Spearman 对单调变换不敏感——这是它相对 Pearson 的关键性质。"""
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
    """确认 R² 用的是 Pearson r 的平方，而非 1 - SS_res/SS_tot。

    模型输出的相似度分数与实测亲和力量纲不同，回归式 R² 会得到大负数，
    这里必须是前者。
    """
    true = np.array([1.0, 2.0, 3.0, 4.0])
    pred = true * 100 + 50          # 完全线性但量纲差 100 倍
    assert r2_score(pred, true) == pytest.approx(1.0)
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    assert 1 - ss_res / ss_tot < -1000        # 回归式 R² 在此场景毫无意义


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
    """4 个配体应有 C(4,2)=6 对；预测把最后两个排反 → 5/6。"""
    true = [1.0, 2.0, 3.0, 4.0]
    pred = [1.0, 2.0, 4.0, 3.0]
    assert pairwise_accuracy(pred, true) == pytest.approx(5 / 6)


def test_pairwise_tol_skips_indistinguishable_pairs():
    """tol 之内的配体对应被跳过，而不是算作错误。"""
    true = [1.0, 1.2, 5.0]        # 前两个差 0.2，在 tol=0.5 内
    pred = [2.0, 1.0, 9.0]        # 前两个预测反了
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
    """无并列时，Kendall τ 与 pairwise accuracy 满足 τ = 2*acc - 1。"""
    rng = np.random.default_rng(3)
    true = rng.normal(size=20)
    pred = rng.normal(size=20)
    assert kendall_tau(pred, true) == pytest.approx(2 * pairwise_accuracy(pred, true) - 1, abs=1e-9)
