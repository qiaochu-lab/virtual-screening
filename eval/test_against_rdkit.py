"""交叉验证：我们的 metrics.py 与 RDKit 的实现是否口径一致。

为什么必须做这一步
------------------
LigUnity 官方的 ``ensemble_result.py`` 用的是
``rdkit.ML.Scoring.Scoring`` 的 ``CalcBEDROC / CalcAUC / CalcEnrichment``。
横评要求所有模型走同一套指标代码，而我们自建的 eval/ 要能替代官方实现，
就必须先证明二者在同样输入下给出同样的数。

对不上不一定是我们错——也可能是定义差异（如并列名次、取整规则）。
但差异必须被查明并记录，不能糊过去。
"""
import numpy as np
import pytest
from rdkit.ML.Scoring.Scoring import CalcAUC, CalcBEDROC, CalcEnrichment

from metrics import bedroc, enrichment_factor, roc_auc


def _rdkit_input(scores, labels):
    """RDKit 要求：按分数降序排列的 [[score, label], ...]。"""
    arr = np.column_stack([np.asarray(scores, float), np.asarray(labels, float)])
    return arr[arr[:, 0].argsort()[::-1]]


def _cases():
    """构造多组规模/活性比例各异的随机数据。

    ⚠️ 必须包含 ``n * fraction`` 非整数的规模。早期版本只用了 300/500/1000/2000，
    这些数配 0.5%/1%/2%/5% 全是整数，掩盖了 round 与 ceil 的差异——
    直到在真实 DUD-E 数据（靶点分子数 2343、9448、52056…）上才暴露出来。
    """
    out = []
    for seed, n, n_act in [
        (0, 500, 25), (1, 1000, 10), (2, 2000, 100), (3, 300, 3),
        # 以下规模会让 n*fraction 落在非整数上，专门覆盖取整分歧
        (4, 2343, 40), (5, 9448, 158), (6, 1207, 37), (7, 4247, 13),
    ]:
        rng = np.random.default_rng(seed)
        labels = np.zeros(n)
        labels[:n_act] = 1
        rng.shuffle(labels)
        # 让分数与标签弱相关，模拟真实模型输出
        scores = rng.random(n) + labels * 0.6
        out.append((f"n={n},act={n_act}", scores, labels))
    return out


@pytest.mark.parametrize("name,scores,labels", _cases())
def test_auc_matches_rdkit(name, scores, labels):
    ours = roc_auc(scores, labels)
    theirs = CalcAUC(_rdkit_input(scores, labels), 1)
    assert ours == pytest.approx(theirs, abs=1e-6), f"{name}: ours={ours} rdkit={theirs}"


@pytest.mark.parametrize("name,scores,labels", _cases())
def test_bedroc_matches_rdkit(name, scores, labels):
    ours = bedroc(scores, labels, alpha=80.5)
    theirs = CalcBEDROC(_rdkit_input(scores, labels), 1, 80.5)
    assert ours == pytest.approx(theirs, abs=1e-6), f"{name}: ours={ours} rdkit={theirs}"


@pytest.mark.parametrize("name,scores,labels", _cases())
@pytest.mark.parametrize("frac", [0.005, 0.01, 0.02, 0.05])
def test_ef_matches_rdkit(name, scores, labels, frac):
    ours = enrichment_factor(scores, labels, frac)
    theirs = CalcEnrichment(_rdkit_input(scores, labels), 1, [frac])[0]
    assert ours == pytest.approx(theirs, rel=1e-6), (
        f"{name} EF@{frac}: ours={ours} rdkit={theirs}"
    )
