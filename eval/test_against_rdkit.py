"""Cross-check: does our metrics.py agree with RDKit's implementation?

Why this is necessary
------------------
LigUnity's official ``ensemble_result.py`` uses
``rdkit.ML.Scoring.Scoring``'s ``CalcBEDROC / CalcAUC / CalcEnrichment``.
The head-to-head comparison requires every model to go through the same
metric code, and for our own eval/ to stand in for the official
implementation, we first have to prove the two give the same numbers on the
same input.

A mismatch is not necessarily our error — it could also be a definitional
difference (e.g. tie-breaking, rounding rule). But any difference must be
identified and recorded, not glossed over.
"""
import numpy as np
import pytest
from rdkit.ML.Scoring.Scoring import CalcAUC, CalcBEDROC, CalcEnrichment

from metrics import bedroc, enrichment_factor, roc_auc


def _rdkit_input(scores, labels):
    """RDKit requires: [[score, label], ...] sorted by descending score."""
    arr = np.column_stack([np.asarray(scores, float), np.asarray(labels, float)])
    return arr[arr[:, 0].argsort()[::-1]]


def _cases():
    """Build several batches of random data with varying sizes and active
    ratios.

    Must include sizes where ``n * fraction`` is not an integer. An earlier
    version only used 300/500/1000/2000, all of which give integers at
    0.5%/1%/2%/5%, masking the difference between round and ceil — it only
    surfaced on real DUD-E data (target molecule counts of 2343, 9448,
    52056...).
    """
    out = []
    for seed, n, n_act in [
        (0, 500, 25), (1, 1000, 10), (2, 2000, 100), (3, 300, 3),
        # the following sizes make n*fraction land on a non-integer, specifically to cover the rounding disagreement
        (4, 2343, 40), (5, 9448, 158), (6, 1207, 37), (7, 4247, 13),
    ]:
        rng = np.random.default_rng(seed)
        labels = np.zeros(n)
        labels[:n_act] = 1
        rng.shuffle(labels)
        # make scores weakly correlated with labels, to simulate real model output
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
