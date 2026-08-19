# Unified Evaluation Layer

One validated implementation of every metric used in this benchmark. Models are
run with their own official code and official weights; **only the metric
computation is unified**, so any difference in a comparison table is
attributable to the models rather than to their metric code.

```
each model  →  dump raw per-molecule scores  →  this layer  →  comparison table
```

Why this is necessary: published numbers for the same model on the same
benchmark disagree across papers. DrugCLIP's DUD-E EF1% is 31.99 in its own
paper and 30.52 when re-run in the BindCLIP paper — a ~5% gap for an identical
model on an identical dataset. Each paper ships its own metric code, differing
in tie handling, cutoff rounding, and whether targets are pooled or averaged.

## Metrics

**Screening / enrichment** (binary active-vs-decoy labels):

| Function | Notes |
|---|---|
| `enrichment_factor(scores, labels, fraction)` | EF at any fraction, including **EF@0.1%** which the reference implementations do not expose |
| `roc_auc(scores, labels)` | Mann–Whitney U form; handles ties correctly by construction |
| `bedroc(scores, labels, alpha=80.5)` | Truchon & Bayly (2007) |
| `top_k_recall(scores, labels, k)` | |
| `bootstrap_ci(fn, scores, labels)` | Bootstrap confidence intervals for any of the above |

**Affinity ranking** (continuous measured affinity as ground truth):

| Function | Notes |
|---|---|
| `spearman(pred, true)` | |
| `pearson(pred, true)` / `r2_score(pred, true)` | R² is squared Pearson *r*, not `1 - SS_res/SS_tot` — model scores and measured affinities are on different scales |
| `pairwise_accuracy(pred, true, tol)` | `tol` skips pairs whose measured difference falls within experimental error |
| `kendall_tau(pred, true)` | |

## Validation

The implementation is checked at three levels.

**1. Definitions follow the literature.** BEDROC per Truchon & Bayly (2007) with
α = 80.5; AUROC via the Mann–Whitney U identity.

**2. Bit-level agreement with the reference implementation.** The official code of
the models in scope computes metrics with `rdkit.ML.Scoring.Scoring`.
`test_against_rdkit.py` compares against RDKit across several dataset sizes and
active ratios; agreement is within 1e-6.

**3. Published values are reproduced on real data.** Running these metrics on raw
scores produced by each model's official code reproduces the values reported in
the corresponding papers (deviation ≤ 2%).

```bash
python -m pytest eval/ -q      # 80 tests
```

## Two implementation details worth knowing

**Top-N cutoff uses `ceil`, not `round`.** RDKit's `CalcEnrichment` computes
`numPerFrac = [math.ceil(numMol * f) for f in fractions]`. An earlier version here
used `round`, and the synthetic tests passed anyway — they used dataset sizes of
300/500/1000/2000, for which `n * fraction` is an integer at every fraction tested,
so the two rounding modes coincide. The discrepancy only surfaced on real data
(a DUD-E target with 2343 molecules: `2343 × 1% = 23.43`, so `round` → 23 while
`ceil` → 24), where it affected 37 of 102 targets.

*Synthetic tests that only use round-number sizes do not exercise rounding logic.*
The test cases now include realistic sizes (2343, 9448, 1207, 4247).

**Ties are resolved by average rank** (`scipy.stats.rankdata(method="average")`).
Score ties are common in virtual screening, and relying on `argsort` stability
makes results depend on the sorting algorithm — on tied data this can change EF@1%
by several fold.

## Expected input format

Per target, two arrays:

```
saved_preds.npy    float, one score per molecule (higher = more likely active)
saved_labels.npy   int,   1 = active, 0 = inactive/decoy
```

Some model implementations print aggregate metrics without persisting per-molecule
scores; those need a small patch to save the arrays before unified evaluation is
possible.

```bash
python eval/score_ligunity.py <results_dir> [--ref ensemble|seq|drugclip|bindclip] [--bootstrap]
```

## Caveat on EF@0.1%

The number of molecules taken at 0.1% depends on library size:

| Benchmark | Molecules per target | Molecules at 0.1% |
|---|---|---|
| DEKOIS 2.0 | 1207–1240 | **1** |
| DUD-E | 2343–52056 | 2–52 |
| LIT-PCBA | 4247–361997 | 4–362 |

On DEKOIS, EF@0.1% degenerates to "was the top-ranked molecule active" — a
per-target indicator with high variance. The mean over targets remains a valid
statistic, but per-target values should not be interpreted individually, and
confidence intervals should always be reported alongside.

