# T2 — Affinity Ranking

**Question:** not just "can it separate actives from decoys", but **can it rank
binding strength**?

**Status:** run on all three datasets. They initially appeared to give opposite
answers; both reasons are now isolated — an ordering bug in our analysis (since
fixed) and range restriction from this benchmark's own `pAff ≥ 6` filter.

> 🔬 **Physics collaborators: this task and T6 are where physics methods matter
> most.** See "Where physics fits" at the bottom.

---

## Why this task is separate from T1

Contrastive learning pulls binding pairs together and pushes non-binding pairs
apart. **Nothing in that objective constrains the ordering of affinities.** So a
model can enrich well and rank badly, and the two capabilities must be measured
separately.

Independent evidence that the field knows this: **HypSeek ships two checkpoints
from the same training run** — `_vs.pt` selected on CASF BEDROC (screening) and
`_rk.pt` selected on FEP (ranking). One weight cannot do both well.

## Data

| Dataset | Scope | Ligand relationship | Status |
|---|---|---|---|
| **T3** (self-built) | 1,044 targets | **Cross-series** — pulled from databases, many scaffolds per target | ✅ 9 models |
| **FEP** (JACS 8 + Merck 8) | 16 systems, 461 ligands | **Within-series** — same scaffold, substituent changes | ✅ 3 models |
| **CASF-2016** | 285 complexes, 55 usable targets × ~5 | Within-target, **cross-scaffold** | ✅ 5 models |

FEP set = Wang et al. 2015 (BACE, CDK2, JNK1, MCL1, p38, PTP1B, thrombin, TYK2)
+ Schindler et al. 2020 (CDK8, c-Met, Eg5, HIF-2α, PFKFB3, SHP-2, SYK, TNKS2).
Shipped with the LigUnity data release; ligand counts verified against the
published sets one by one.

## How it was run

**No extra model inference.** T2 reuses the per-molecule scores already saved by
T1/T3, keeps only the actives (decoys have no measured affinity), and correlates
model score against measured pAffinity **within each target**, then averages
over targets.

```bash
python timesplit/analysis/score_t2.py --models <m1> <m2> ...   # on T3 data
python physics/score_fep.py                             # on FEP data
python physics/fep_compare_physics.py                   # vs the physics reference
```

## ⚠️ These numbers were corrected on 2026-08-21

An earlier version of this document reported that ranking ability on T3 was
**zero** for every model, and that a paired test proved ranking survives only
within a congeneric series. **Both claims were artifacts of a molecule-ordering
bug in our own analysis code**, not properties of the models. The bug, how it was
caught, and what it did are in [`PATCHES.md`](../PATCHES.md); the corrected
numbers are below, and the old (wrong) column is kept in
[`results/T2_on_T3.csv`](../results/T2_on_T3.csv) as `spearman_old_misaligned`
so the size of the correction is auditable.

## Results

### On T3 data — weak but real, and it decays like enrichment does

Per-target Spearman between model score and measured pAffinity, averaged over
targets ([`timesplit/analysis/score_t2_v2.py`](../timesplit/analysis/score_t2_v2.py)):

| Model | L1 | L2 | L3 | L4 | ρ>0 at L1 |
|---|---|---|---|---|---|
| **HypSeek `_rk`** | **+0.260** | **+0.114** | **+0.118** | **+0.096** | 79% |
| LigUnity-protein | +0.230 | +0.105 | +0.062 | +0.089 | 79% |
| LigUnity-pocket | +0.215 | +0.103 | +0.089 | +0.055 | 73% |
| LiTENCLIP | +0.171 | +0.051 | +0.044 | +0.053 | 70% |
| ConGLUDe | +0.129 | +0.026 | +0.047 | +0.056 | 70% |
| BindCLIP-randneg | +0.119 | +0.048 | −0.048 | +0.023 | 66% |
| BindCLIP-hardneg | +0.112 | +0.026 | +0.034 | +0.052 | 66% |
| DrugCLIP | +0.091 | +0.032 | +0.040 | +0.015 | 64% |
| ConPLex | +0.065 | −0.002 | +0.036 | +0.004 | 54% |

Standard errors are ±0.015–0.044; the L1 column is comfortably non-zero for
every structure model, and 64–79% of individual targets have the right sign
against a 50% baseline.

**Three things this says:**

1. **Ranking ability decays across layers, like enrichment.** HypSeek falls
   +0.260 → +0.096, LigUnity-protein +0.230 → +0.089. The novel-target penalty
   applies to both capabilities, which is a stronger version of T3's finding than
   we had before.
2. **The checkpoint selected for ranking is the best ranker.** HypSeek's `_rk`
   weight — chosen upstream on a FEP validation set — leads every layer. Earlier
   this document claimed the opposite; that claim came from the buggy path.
3. **Models trained on affinity-labelled data rank better.** The PocketAffDB
   group (HypSeek, LigUnity ×2, LiTENCLIP: +0.17 to +0.26 at L1) separates
   cleanly from the DrugCLIP-data group (+0.09 to +0.12), matching the pattern
   already seen in enrichment.

### On FEP data

| Model | Spearman | Pearson | Systems with correct direction |
|---|---|---|---|
| LigUnity-protein | **+0.396** | +0.428 | **16/16** |
| LigUnity-pocket | +0.392 | +0.434 | 13/16 |
| LiTENCLIP | +0.276 | +0.247 | 13/16 |

Per-system: [`results/T2_on_FEP.csv`](../results/T2_on_FEP.csv). These were never
affected by the bug — the FEP path stores one array per system in a defined
order.

### On CASF-2016

Same target, five **different scaffolds** per cluster
([`timesplit/analysis/score_casf.py`](../timesplit/analysis/score_casf.py)):

| Model | scoring power (ρ over 285) | Pearson r | ranking power (mean ρ within target) | targets |
|---|---|---|---|---|
| **HypSeek `_rk`** | **0.627** | **0.622** | **0.549** | 55 |
| LiTENCLIP | 0.364 | 0.356 | 0.371 | 55 |
| LigUnity-pocket | 0.360 | 0.316 | 0.424 | 55 |
| LigUnity-protein | 0.221 | 0.193 | 0.282 | 55 |
| ConPLex (sequence only) | 0.169 | 0.253 | 0.111 | 53 |

**HypSeek `_rk` leads by a wide margin here too** — 0.627 scoring power against
LigUnity-pocket's 0.360. Together with its first place on all three screening
benchmarks and on T3 ranking, one checkpoint now leads every axis we measure.

**Why only five models.** CASF needs (protein, ligand) → score, and
`casf_label_seq.json` ships sequences and SMILES for all 285 complexes, so
ConPLex ran with no preparation at all. The other four need work that is not
incidental: ConGLUDe wants a `.pdb` per target (285 PDBbind entries to fetch),
SPRINT needs foldseek 3Di tokens computed on those structures, and
DrugCLIP/BindCLIP have no CASF branch in their repositories at all — that code
path would have to be ported, and the two forks that do have it shipped it
broken (see [`PATCHES.md`](../PATCHES.md)). Recorded as a coverage gap rather
than done badly.

⚠️ CASF complexes come from PDBbind, which overlaps these models' training data,
and the field selects checkpoints on CASF — so this is close to in-distribution.
Five ligands per cluster also makes each per-target Spearman coarse.

### The paired test, redone

Same 14 targets present in **both** FEP and T3, T3 side now aligned by molecule
identity ([`timesplit/analysis/fep_vs_t3_v2.py`](../timesplit/analysis/fep_vs_t3_v2.py)):

| Model | FEP data | T3 data | paired Wilcoxon |
|---|---|---|---|
| LigUnity-pocket | +0.391 | **+0.289** | p = 0.33 |
| LigUnity-protein | +0.413 | **+0.290** | p = 0.27 |

**No significant difference.** The earlier version of this table read +0.391 vs
−0.055 (p = 0.0012) and was the basis for claiming that ranking collapses across
chemical series. That conclusion is withdrawn: on the same targets, T3 ligands
rank about as well as congeneric FEP ligands.

### Is it label noise?

T3 pools Ki / Kd / IC50 / EC50 across labs, so we tested whether cleaning the
labels raises the correlation
([`timesplit/analysis/t2_label_quality.py`](../timesplit/analysis/t2_label_quality.py)):
all actives → only the target's dominant assay type → only its single largest
`assay_id` (one experiment, one lab).

| Model | Layer | all | one assay type | one assay |
|---|---|---|---|---|
| LigUnity-protein | L1 | 0.171 | 0.216 | 0.095 |
| LigUnity-protein | L4 | 0.082 | 0.136 | 0.124 |
| LiTENCLIP | L1 | 0.138 | 0.137 | 0.048 |
| ConGLUDe | L1 | 0.129 | 0.110 | 0.050 |

**No monotone improvement.** Restricting to one assay type helps slightly at
times, and the single-assay tier is worse — but it also has a median of 9–16
ligands per target, where Spearman is very noisy. Label heterogeneity is not the
main driver.

### Why CASF looks so much better than T3: range restriction

The same models score ρ ≈ 0.42–0.55 within CASF targets and ρ ≈ 0.09–0.26 within
T3 targets. That 3–5× gap is mostly **an artifact of our own eval-set filter**,
not a property of the models
([`physics/t2_gap.py`](../physics/t2_gap.py),
[`results/T2_range_restriction.csv`](../results/T2_range_restriction.csv)).

T3's actives must pass `pAff ≥ 6`, which truncates the weak half of the
distribution. Correlations shrink mechanically when the spread of the true
values is compressed:

| | targets | ligands/target (median) | within-target pAff SD | range |
|---|---|---|---|---|
| CASF-2016 | 55 | 5 | **1.576** | 4.06 |
| T3 L1 | 349 | 24 | **0.783** | 2.99 |

The spread is almost exactly halved (ratio 2.01). Correcting the observed T3
values back to CASF's spread (Thorndike case II,
ρ_true ≈ ρk / √(1 + ρ²(k²−1))):

| Model | T3 L1 observed | range-corrected | CASF observed | gap explained |
|---|---|---|---|---|
| HypSeek `_rk` | 0.260 | **0.477** | 0.549 | 75% |
| LigUnity-pocket | 0.215 | **0.406** | 0.424 | 91% |

**So T3 and CASF do not disagree about these models.** They disagree about how
much affinity spread their ligand sets contain. The residual (HypSeek still 0.07
short) is consistent with the label heterogeneity measured above — CASF is
curated Kd/Ki from PDBbind, T3 is mixed ChEMBL assay types.

Note what this does **not** license: the corrected numbers are an estimate of
what these models would score on a CASF-like spread, not a measurement. Report
the observed T3 value, and cite the correction as the explanation for the gap.

### What the corrected picture looks like

> These models **do** rank affinity, weakly. On post-cutoff targets the
> correlation is ρ ≈ 0.1–0.26 and decays with target novelty; on curated
> congeneric benchmarks it is ρ ≈ 0.4. On the shared targets the two are
> statistically indistinguishable. Meanwhile a co-folding model with an affinity
> head reaches ρ = 0.615 on the same FEP ligands — **the gap between retrieval
> and physics is quantitative, not categorical.**

## Open decision

Which dataset is canonical for T2? Recommendation: **report all three
separately**, because they measure different things:
FEP = lead optimisation; CASF = standard scoring-function benchmark;
T3 = hit triage. Collapsing them into one average hides the main finding.

---

## Where physics fits 🔬

**Current T2 contains no thermodynamic or physical quantity at all.** Model
outputs are cosine similarities — dimensionless. That means these are impossible
today:

| Metric | Why not |
|---|---|
| ΔG (kcal/mol) | retrieval models emit no energy |
| RMSE vs experiment | different units; regression R² would be a large negative number |
| ΔΔG (relative binding free energy) | the core output of FEP methods |
| ΔH / −TΔS decomposition | needs TI or calorimetry |

Note `eval/metrics.py::r2_score` deliberately returns **Pearson r²**, not
`1 − SS_res/SS_tot`, for exactly this reason — the docstring says so.

**Once a physics method is plugged in, the table gains a second half:**

| Method | Kendall τ (16 FEP systems) | RMSE (kcal/mol) |
|---|---|---|
| LigUnity-pocket | 0.291 | not computable |
| LigUnity-protein | 0.284 | not computable |
| LiTENCLIP | 0.200 | not computable |
| **Uni-FEP** (reference, published) | **0.503** | 0.65–1.79 |
| **Boltz-2** | **0.474** (mean over 16 systems) | **computable** |

⚠️ The reference row is **Uni-FEP**, not Schrödinger FEP+ — same family of
free-energy methods, so the magnitude is indicative, but do not cite it as
"FEP+ results". Source: `dptech-corp/Uni-FEP-Benchmarks`; ligand counts match
ours system by system (verified).

**Physics does not win everywhere.** On SHP-2 the retrieval models reach
+0.545/+0.438 while Uni-FEP gets 0.120; same on HIF-2α and TNKS2. That is the
empirical basis for calling the two families *complementary* rather than one
dominating.

**Unused metric worth adding:** `eval/metrics.py::pairwise_accuracy(tol=...)`
skips ligand pairs whose measured difference falls inside experimental error
(~0.3–0.5 log units). Raw Spearman punishes a model for failing to order pairs
that are indistinguishable in the assay, which may be part of why T3 reads as
zero.

---

## Code

| What | File |
|---|---|
| Ranking metrics on T3 data (per-target Spearman/Kendall/Pearson, then averaged) | [`timesplit/analysis/score_t2.py`](../timesplit/analysis/score_t2.py) |
| Run the models on the 16 FEP systems | [`physics/run_fep.sh`](../physics/run_fep.sh) |
| Make the FEP task persist raw scores | [`physics/patch_fep_save.py`](../physics/patch_fep_save.py) |
| Recover scores from stored embeddings where a run predates the patch | [`physics/fep_recover_preds.py`](../physics/fep_recover_preds.py) |
| Score the FEP systems | [`physics/score_fep.py`](../physics/score_fep.py) |
| **The paired test on the 14 shared targets** — the experiment that corrected the conclusion | [`physics/fep_vs_t3_same_targets.py`](../physics/fep_vs_t3_same_targets.py) |
| Compare against the published physics reference | [`physics/fep_compare_physics.py`](../physics/fep_compare_physics.py) |
| Metric implementations (`spearman`, `kendall_tau`, `r2_score`, `pairwise_accuracy`) | [`eval/metrics.py`](../eval/metrics.py) |

Two of these exist because of a reporting problem rather than a modelling one:
`patch_fep_save.py` adds one line so raw scores survive, and
`fep_recover_preds.py` reconstructs them as `pocket_emb @ mol_emb.T` followed by
a max over pockets — byte-identical to the official computation, so nothing had
to be re-run.
