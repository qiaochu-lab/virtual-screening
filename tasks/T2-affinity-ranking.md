# T2 — Affinity Ranking

**Question:** not just "can it separate actives from decoys", but **can it rank
binding strength**?

**Status:** run on two datasets, which give **opposite answers**. The reason has
been isolated. A third dataset (CASF-2016) is available and not yet run.

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
| **CASF-2016** | 50 clusters × 5 ligands | Within-target, cross-scaffold | ⬜ data present, not run |

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
python tasks/scripts/score_t2.py --models <m1> <m2> ...     # on T3 data
python tasks/scripts/score_fep.py                           # on FEP data
python tasks/scripts/fep_compare_physics.py                 # vs physics reference
```

## Results

### On T3 data — essentially zero for everyone

| Model | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| DrugCLIP | −0.011 | +0.008 | +0.036 | +0.004 |
| LigUnity-protein | +0.021 | +0.023 | — | — |
| **HypSeek `_rk`** (ranking-optimised!) | **+0.028** | +0.023 | +0.003 | +0.030 |
| ConGLUDe | +0.129 | +0.026 | +0.047 | +0.056 |
| ConPLex | +0.065 | −0.002 | +0.036 | +0.004 |

Fraction of targets with ρ>0 is 48–53% — coin-flip. Full table:
[`results/T2_on_T3.csv`](../results/T2_on_T3.csv).

**Not a data artefact:** measured affinity spans a median of 2.3–3.0 log units
per target (200–1000× between weakest and strongest active); restricting to
targets with span ≥2 changes nothing.

### On FEP data — clearly non-zero

| Model | Spearman | Pearson | Systems with correct direction |
|---|---|---|---|
| LigUnity-protein | **+0.396** | +0.428 | **16/16** |
| LigUnity-pocket | +0.392 | +0.434 | 13/16 |
| LiTENCLIP | +0.276 | +0.247 | 13/16 |

Per-system: [`results/T2_on_FEP.csv`](../results/T2_on_FEP.csv).

### The two are reconciled by a paired test

Same 14 targets present in **both** datasets, same model:

| Model | FEP data | T3 data | paired p |
|---|---|---|---|
| LigUnity-pocket | **+0.391** | **−0.055** | 0.0012 |
| LigUnity-protein | **+0.413** | **−0.004** | 0.0001 |

Identical targets (thrombin, mcl1, cdk2, bace…), different ligand sets: ρ drops
from 0.4 to 0. **Target familiarity is ruled out** — the difference comes from
ligand composition.

### Conclusion (corrected)

> Within a **single chemical series**, these models have moderate ranking
> ability (ρ ≈ 0.4). Across series and across data sources, ranking ability
> **vanishes**.

They have learned **local structure–activity relationships** — whether adding
this fluorine helps — not absolute binding strength across scaffolds.

**Two natural rescues were tested and both failed:**
- Not a training-objective problem — HypSeek's `_rk` weight is selected
  specifically on ranking, and still gives +0.028 on T3
- Not a geometry problem — HypSeek uses hyperbolic space (the only non-Euclidean
  model here), tops every AUROC layer, and still ranks at zero

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
| **Boltz-2** | running | **computable** |

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
