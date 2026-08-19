# T4 — Target Fishing

**Question:** run the retrieval backwards — given a molecule, can the model find
which target it binds?

**Status:** not started. Deliberately deprioritised.

---

## The idea

Every model here scores a (pocket, molecule) pair. T1/T3 fix the target and rank
molecules; T4 fixes the molecule and ranks targets. Same scores, transposed
matrix.

This matters because the two directions are **not** symmetric. A model can be
good at "which of these 1,250 molecules binds EGFR" and bad at "which of these
1,000 proteins does imatinib bind", because the score distributions differ per
target — a target whose scores are uniformly high will win every molecule's
ranking regardless of truth. Per-target score calibration becomes the dominant
issue, and none of these models were trained with that in mind.

## Data

Reuses T3 directly. The eval set already contains, for every target, its actives
and its cross-target decoys — and a decoy for target A is by construction an
active for some dissimilar target B. So the (molecule → true target) mapping is
already there.

**No new inference needed.** All nine models' per-molecule scores are on disk.

## What would need building

1. Assemble the molecule × target score matrix from the per-target arrays
   (they are currently stored per target, with different molecule subsets each)
2. Decide the candidate target set per molecule — all 1,044, or only within a
   layer
3. Pick metrics: top-k accuracy, MRR, and enrichment over a random target ranking
4. Decide whether to calibrate scores per target before ranking, and report both
   ways — the uncalibrated version is what a user would naively get, the
   calibrated version measures the representation itself

## Why it is deprioritised

The scientific return is lower than the other tasks: it re-uses the same scores
and mostly measures calibration, which is a known weakness of contrastive models
rather than a new finding. It was scheduled last by agreement.

One reason it may be worth doing later: **ConGLUDe's paper includes a target
fishing task**, so there is a published baseline to compare against — that would
make our numbers interpretable rather than free-floating.

## Physics methods

❌ **Not applicable.** Target fishing requires scoring one molecule against
thousands of targets. Physics methods cost minutes per complex; the cost model
does not work in this direction at all.

## Code

Nothing specific to T4 yet. What it would build on:

| Input it would reuse | Where |
|---|---|
| Per-target score arrays for all 9 models | produced by [`t3/runners/`](../t3/runners/), scored by [`t3/analysis/score_t3.py`](../t3/analysis/score_t3.py) |
| Molecule → true-target mapping | already implicit in [`t3/build/build_t3_eval.py`](../t3/build/build_t3_eval.py) (a decoy for one target is an active for another) |
| Metrics (top-k recall exists; MRR would be added) | [`eval/metrics.py`](../eval/metrics.py) |
