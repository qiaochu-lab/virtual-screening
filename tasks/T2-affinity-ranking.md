# T2 — Affinity Ranking

**Question:** not just "can it separate actives from decoys", but **can it rank
binding strength**?

**Status:** run on all three datasets. They initially appeared to give opposite
answers; both reasons are now isolated — an ordering bug in our analysis (since
fixed) and range restriction from this benchmark's own `pAff ≥ 6` filter.

**Two findings added 2026-09-09, both of which narrow what this task can claim:**
splitting the actives by ligand novelty shows the ranking correlation is at
floor on unfamiliar chemistry at *every* layer, so the L1→L4 curve is not a
target-novelty effect; and **51.9% of CASF-2016 is literally in these models'
training files**, by a training-code decision that excludes DUD-E / DEKOIS /
LIT-PCBA targets but not CASF.

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

### On T3 data — weak but real, and it tracks chemical familiarity

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
   ⚠️ **This layer-wise reading is superseded below.** Splitting each layer's
   actives by ligand novelty shows the decay is not a target-novelty effect at
   all — on novel chemistry the correlation is already at floor at L1. See
   *Ranking ability is carried by familiar chemistry*.
2. **The checkpoint selected for ranking is the best ranker.** HypSeek's `_rk`
   weight — chosen upstream on a FEP validation set — leads every layer. Earlier
   this document claimed the opposite; that claim came from the buggy path.
3. **Models trained on affinity-labelled data rank better.** The PocketAffDB
   group (HypSeek, LigUnity ×2, LiTENCLIP: +0.17 to +0.26 at L1) separates
   cleanly from the DrugCLIP-data group (+0.09 to +0.12), matching the pattern
   already seen in enrichment.


## Dataset v2 — results on the ≥50-actives, VSDS-vd-matched subset

The advisor fixed two rules on 2026-09-04: **≥50 actives per target**, and a
class composition matched to **VSDS-vd** (Gu et al., *Nat Mach Intell* 7:509–520,
2025). That yields a 242-entry / 222-target subset — see
[T3 dataset v2](T3-dataset-v2.md). It is a *filter*, not a rebuild, so every
number below is a re-aggregation of scores that already existed; nothing was
re-inferred.

`score_t2_v2.py` originally stored only per-layer aggregates, so the subset
could not be re-aggregated at all — it now also emits `per_target`
(uniprot / spearman / kendall / n_actives). That is the only reason it had to
be re-run; no model was re-inferred.

⚠️ **This table was regenerated on 2026-09-09.** The version published before
that date was produced from a corrupted intermediate — `score_t2_v2.py` had the
per-target patch applied twice, so `ups.append(up)` ran twice per target while
the value lists ran once, and `zip(ups, new_r, new_t, ns)` silently truncated to
the shorter list. Every target was therefore labelled with **another target's**
correlation, and only the first half of targets appeared at all. The aggregate
statistics were never affected — they are computed from the value lists directly
— so the full-set table above and every conclusion resting on it stand. Only the
subset re-aggregation, which filters by uniprot, was wrong. See
[`PATCHES.md`](../PATCHES.md).

| Model | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| **HypSeek `_rk`** | **+0.225** ± .041 | +0.144 | +0.087 | **+0.139** |
| LigUnity-protein | +0.221 ± .036 | +0.116 | +0.008 | +0.102 |
| LigUnity-pocket | +0.174 ± .037 | +0.121 | +0.044 | +0.092 |
| ConGLUDe | +0.109 ± .030 | +0.046 | +0.027 | +0.056 |
| LiTENCLIP | +0.103 ± .031 | +0.081 | −0.002 | +0.092 |
| BindCLIP-hardneg | +0.087 ± .032 | +0.039 | −0.007 | +0.050 |
| BindCLIP-randneg | +0.073 ± .031 | +0.060 | −0.025 | +0.022 |
| DrugCLIP | +0.044 ± .035 | +0.037 | +0.004 | −0.005 |
| SPRINT | +0.013 ± .029 | +0.053 | +0.054 | +0.071 |
| ConPLex | −0.001 ± .028 | +0.004 | −0.018 | −0.024 |

Targets per layer: L1 47–55 · L2 138–170 · L3 16–19 · L4 59–68, each within the
subset's own 56 / 178 / 19 / 75. (The corrupted version reported up to 72 at L1,
above the subset's own L1 count — the arithmetic impossibility that exposed it.)

**The weakness holds; the decay does not.** HypSeek falls +0.260 → +0.096 across
layers on the full set but only **+0.225 → +0.139** on this subset, and L4
(+0.139) sits *above* L3 (+0.087). The subset's rule is ≥50 actives per target,
so a per-target Spearman here rests on many more ligands than one drawn from the
full set's long tail of 10-active targets. Read that as the layer-wise decay in
T2 being partly a small-sample effect at L4, not as two datasets disagreeing.

This does not rescue ranking ability: **+0.139 is still weak**, and the
decomposition below shows what carries even that much.

File: `results/T2_on_T3_subset.csv`.

### Ranking ability is carried by familiar chemistry

The layer-wise table above holds the target constant and lets the *chemistry*
vary with it: at L1, 53.9% of actives are near-duplicates of a training ligand
(Tanimoto ≥ 0.7); at L4, 6.4%. So "ρ falls from +0.26 to +0.10 across layers"
confounds two things at once — the same mistake the enrichment table made before
it was split by novelty tier (main table L4 EF1% 9.38 → 27.1 on seen chemistry,
4.5 on unseen).

Splitting each target's actives by their maximum Tanimoto to the training
ligands and computing the per-target Spearman **within each tier**
([`timesplit/analysis/t2_novelty_tiers.py`](../timesplit/analysis/t2_novelty_tiers.py)
→ [`results/T2_novelty_tiers.csv`](../results/T2_novelty_tiers.csv),
[`T2_novelty_paired.csv`](../results/T2_novelty_paired.csv)):

**Paired within target, familiar half (≥0.5) vs novel half (<0.5), Wilcoxon:**

| Model | L1 familiar | L1 novel | Δ | p | wins |
|---|---|---|---|---|---|
| LigUnity-protein | +0.242 | **+0.011** | +0.231 | **0.0001** | 55/79 |
| LiTENCLIP | +0.156 | **−0.038** | +0.194 | **0.0009** | 50/79 |
| HypSeek `_rk` | +0.281 | **+0.080** | +0.201 | **0.0013** | 50/78 |
| LigUnity-pocket | +0.195 | +0.070 | +0.125 | 0.024 | 47/79 |
| DrugCLIP | +0.070 | +0.023 | +0.048 | 0.21 | 47/81 |
| BindCLIP-randneg | +0.044 | +0.006 | +0.038 | 0.44 | 44/81 |
| BindCLIP-hardneg | +0.063 | +0.023 | +0.040 | 0.38 | 44/81 |

Across the 14 tests (7 models × L1/L4), BH-FDR keeps the first three;
LigUnity-pocket's 0.024 does not clear the threshold (0.0143).

**At L4 every model has p > 0.19**, with Δ between +0.03 and −0.05. The effect
is gone.

**The shape that matters:**

| | novel chemistry (<0.5) | familiar chemistry (≥0.5) |
|---|---|---|
| **L1** | +0.011 … +0.080 | +0.156 … +0.281 |
| **L4** | +0.049 … +0.087 | +0.035 … +0.104 |

**On novel chemistry there is no L1→L4 decay, because there is nothing left to
decay — the correlation is already at floor at L1.** The entire layer-wise decay
in T2 lives in the familiar-chemistry half (HypSeek +0.281 → +0.104,
LigUnity-protein +0.242 → +0.064).

> These models rank affinity **among molecules resembling ones they were trained
> on**. Presented with chemistry they have not seen, they are at ρ ≈ 0 on
> familiar and novel targets alike.

Note this is a *different* shape from the enrichment side, where L4 novel
chemistry still enriched 4.5-fold over random. Retrieval retains some ability to
find novel actives on novel targets; ordering them by strength it does not.

⚠️ **Confounder checked.** At L1 the familiar half also has a 35% wider
within-target affinity spread (median SD 0.790 vs 0.586; L2/L3/L4 are 0.98–1.16,
i.e. no difference). Correcting the L1 novel half up to the familiar half's
spread (Thorndike case II, k = 1.35) moves HypSeek +0.080 → +0.107 and
LigUnity-protein +0.011 → +0.015 — far below their familiar halves. The effect
survives.

**One limit on this table:** its four-tier version has only **6 targets** in
L1's `<0.35` tier — too thin to read. Every claim above rests on the two-half
split.

#### Measured against each model's own training ligands, all seven show it

The table above tiers every model against **PocketAffDB's** 428,767 training
ligands. DrugCLIP and BindCLIP ×2 do not train on those: their ligands are the
66,164 pocket–ligand pairs of `train_no_test_af`, **13,590 unique molecules —
31.6× smaller**. Recomputing the novelty cache against that set
([`timesplit/analysis/novelty_drugclip.py`](../timesplit/analysis/novelty_drugclip.py))
changes the picture of the candidate pool completely:

| reference set | T3 median max-Tanimoto | ≥0.7 | ≥0.5 | <0.35 |
|---|---|---|---|---|
| affinity half, 428,767 ligands | 0.419 | 8.8% | ~46% | 23.6% |
| **`train_no_test_af`, 13,590** | **0.303** | **0.9%** | **5.3%** | **72.6%** |

Against their own training set, **72.6% of T3 is novel chemistry for the
DrugCLIP family** — so the familiar/novel *halves* split barely exists for them
(5.3% above 0.5), and the three non-significant rows above were partly an
artifact of tiering them against somebody else's data.

Re-running with the corrected cache
([`results/T2_novelty_paired_Agroup.csv`](../results/T2_novelty_paired_Agroup.csv)),
and using the contrast that this reference set can actually support — the
extreme tiers rather than the halves:

| Model | L1 ≥0.7 | L1 <0.35 | Δ | p | wins |
|---|---|---|---|---|---|
| DrugCLIP | +0.241 | −0.004 | +0.244 | **0.00017** | 33/46 |
| BindCLIP-randneg | +0.195 | +0.010 | +0.185 | **0.013** | 29/46 |
| BindCLIP-hardneg | +0.193 | +0.037 | +0.156 | **0.012** | 27/46 |

**All three are significant.** The effect is not a property of the PocketAffDB
group — it is a property of all seven models tested, once each is measured
against the data it actually saw.

⚠️ **Which contrast is usable depends on the reference set, and the two groups
need different ones.** Against the 428,767-ligand set, the extreme tiers leave
only 5–9 targets with ≥5 actives in both, so the halves split is the usable
contrast; against the 13,590-ligand set the halves are diluted (n = 148, p =
0.08–0.60) while the extreme tiers have 46 targets. Both are reported above;
neither is cherry-picked, but they are not interchangeable and should not be
compared across groups as if they were the same statistic.

An earlier version of this section concluded that the effect held only for the
PocketAffDB-trained models. That was an artifact of the shared novelty cache and
is **withdrawn**.

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

#### Half of CASF is literally in the training set

The caveat this section used to carry — "CASF comes from PDBbind, which overlaps
these models' training data" — was qualitative. It is now measured, and the
number is large
([`timesplit/analysis/casf_train_overlap.py`](../timesplit/analysis/casf_train_overlap.py),
[`results/T2_casf_train_overlap.txt`](../results/T2_casf_train_overlap.txt)):

| | overlap with PocketAffDB training files |
|---|---|
| **CASF PDB IDs** | **148 / 285 = 51.9%** |
| CASF ligands (InChIKey) | 82 / 276 = 29.7% |
| CASF UniProts | 49 / 68 = 72.1% |

This is **identity, not similarity** — the same deposition, e.g. CASF `4eky`
(P00489) against training entry `4ekyA--4eky_D1J_A_1.lmdb` (P00489). No
threshold to argue about. (`1p1n` is in the overlap list, which is the very
example Graber et al. use to argue that *sequence identity* cannot detect
leakage; here it is not similar, it is the same entry.)

**And it is by construction, not accident.** `unimol/tasks/train_task.py` builds
the list of targets to drop from training as:

```python
if self.args.valid_set == "CASF":
    # remove all testset protein by default
    testset_uniprot_lst  = [x[0] for x in json.load(open("dude.json"))]
    testset_uniprot_lst += [x[0] for x in json.load(open("PCBA.json"))]
    testset_uniprot_lst += [x[0] for x in json.load(open("dekois.json"))]
```

DUD-E, LIT-PCBA and DEKOIS targets are removed. **CASF is not in the list**,
despite the branch being named for it. And the PDBbind half of the training
labels (`pair_label_1`) is filtered *only* under the `no_similar_protein`
variants — the default released weight does not filter it at all. CASF is drawn
from PDBbind. That is the mechanism behind the 51.9%.

**Consequence for the whole benchmark**: of the four benchmarks these weights
are evaluated on, DUD-E / DEKOIS / LIT-PCBA had their targets deliberately
excluded from training and **CASF did not**. T1's numbers are the relatively
clean ones; CASF is the exception.

#### What the leakage is actually worth: only one model, and only for ranking

Splitting the 68 target clusters into fully-contaminated (21), mixed (28) and
fully-clean (19)
([`timesplit/analysis/casf_clean_split.py`](../timesplit/analysis/casf_clean_split.py),
[`results/T2_casf_clean_split.csv`](../results/T2_casf_clean_split.csv)):

⚠️ **A confounder has to be removed first.** The fully-clean targets have a
*narrower* within-target affinity spread than the contaminated ones (median SD
1.375 vs 1.854, ratio **1.35**) — the same restriction-of-range mechanism this
document uses below to explain the CASF/T3 gap. It cannot be invoked only when
convenient. Correcting the clean column to the contaminated column's spread
(Thorndike case II) and bootstrapping over targets (2,000 draws):

| Model | fully-contaminated ρ [95%] | fully-clean ρ [95%] | clean, range-corrected | verdict |
|---|---|---|---|---|
| **HypSeek `_rk`** | +0.800 [+0.708, +0.877] | +0.307 [+0.114, +0.493] | **+0.399** | **leakage effect holds** |
| LigUnity-pocket | +0.554 [+0.346, +0.731] | +0.264 [−0.071, +0.579] | +0.347 | intervals overlap |
| LigUnity-protein | +0.254 [−0.092, +0.562] | +0.150 [−0.136, +0.429] | +0.200 | intervals overlap |
| LiTENCLIP | +0.485 [+0.277, +0.662] | +0.393 [+0.143, +0.621] | +0.499 | overlap (higher once corrected) |

**Scoring power shows no effect at all** — on clean complexes LigUnity-protein
*rises* 0.077 → 0.301 and LiTENCLIP 0.244 → 0.394.

So the honest statement is narrow: **the overlap is real and large, but its
measurable effect appears in one model, in one metric.** What makes that
uncomfortable rather than reassuring is *which* model — HypSeek `_rk` is the
checkpoint that leads the CASF table (scoring power 0.627 against
LigUnity-pocket's 0.360), and the claim that one checkpoint leads every axis we
measure rests partly on complexes it was trained on.

**Do not write "CASF's numbers are all leakage."** Three of four models cannot
be separated, n is 13 and 14 targets, and this is an observational split:
contaminated targets are the long-studied PDBbind classics, which differ from
the clean ones in label quality as well as spread, and only spread was corrected.

Because the "clean" group is clean only by *exact PDB ID*, some of those 19
targets are plausibly near-neighbours of training targets by pocket or
interaction similarity ([`LIMITATIONS.md`](../LIMITATIONS.md) §25). The bias
direction is therefore known: **0.800 vs 0.399 is a lower bound on the gap.**

#### The intervention that would have settled it does not exist

The obvious causal test is to hold the 285 complexes fixed and vary only the
model: LigUnity ships three checkpoints trained at different
`--protein-similarity-thres` (1.0 = no filtering, the released default; 0.8;
0.3). We ran all four extra variants on CASF
([`run_casf_thres.sh`](../physics/run_casf_thres.sh),
[`results/T2_casf_thres_intervention.csv`](../results/T2_casf_thres_intervention.csv)).

**The design does not test what it looks like it tests.** As the code above
shows, the filter removes training proteins similar to *DUD-E / LIT-PCBA /
DEKOIS* targets — never CASF. Bootstrapping the dirty-vs-clean interaction over
complexes (5,000 draws), one of four contrasts has an interval excluding zero
(pocket 1.0→0.8: −0.247 [−0.449, −0.047]), which is most plausibly mediated by
CASF and DUD-E sharing targets rather than by any CASF-specific filtering.

**Recorded as a negative result: no public checkpoint was trained with CASF
excluded, so the causal version of this experiment cannot be run without
retraining.** The two facts above — 51.9% identity overlap, and the training
code that produces it — do not need it.

⚠️ Five ligands per cluster also makes each per-target Spearman coarse.

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

⚠️ **And the target of the correction is itself contaminated.** This argument
treats CASF's ρ ≈ 0.42–0.55 as "what these models score when the spread is
normal". Half of CASF is in their training data (above), so that reference value
is inflated by an unknown amount. For HypSeek the fully-clean, spread-corrected
CASF ranking ρ is **+0.399**, while T3's L1 corrected to CASF's spread is
**+0.477** — the gap does not merely close, it reverses. Those two numbers are
corrected to *different* reference spreads (CASF overall SD 1.576 vs the
contaminated targets' 1.854) and so must not be subtracted from one another;
recomputing both against one common spread is the open item. **Until that is
done, "range restriction explains 75–91% of the gap" should be stated as one of
two live explanations, not the settled one.**

### What the corrected picture looks like

> These models **do** rank affinity, weakly, **and only among chemistry they
> have seen**. On post-cutoff targets the correlation is ρ ≈ 0.1–0.26 overall,
> but splitting by ligand novelty puts the novel half at ρ ≈ 0.01–0.08 at L1 and
> ρ ≈ 0.05–0.09 at L4 — at floor in both, so what decays across layers is the
> familiar-chemistry half, not ranking ability as such. On curated congeneric
> benchmarks it is ρ ≈ 0.4, and on the shared targets FEP and T3 are
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

### Head to head with a QM-based scorer, on the same targets and the same ligands

The Boltz-2 comparison above runs on 16 FEP systems, which are not T3 targets. A
collaborator's AIMNet2 pipeline (`xianyang123-bit/aimnet_score_pipelines`) scored
**93 T3 targets** directly — every one of them in our eval set — which allows the
comparison the FEP table cannot make: same targets, same layers.

⚠️ **What that pipeline is.** Its README states it plainly: *"a reconstruction of
the composite energy expression using public checkpoints, **not an official
affinity-trained AIMNet2(Score) release**."* Composite = minimized interaction +
desolvation + local ligand strain, on AIMNet2(2025) member 0; lower is better, and
it is correlated against pAffinity as **negative** energy. Fixed-pocket
minimization converged for 1,440 of 1,667 cases (86.4%); non-converged results are
retained and flagged rather than dropped, so there is no convergence-selection
bias. Cite it as a reconstruction, never as "AIMNet2(Score) results" — the same
care the Uni-FEP row above needs.

**The comparison has to be paired on ligands, not just on targets.** That pipeline
scores **10 actives per target**; T2 uses every active a target has (median
24–118). A Spearman over 10 points against one over 100 is not a comparison of
methods, it is a comparison of sample sizes. So we restrict our models to exactly
the molecules AIMNet2 scored, matched by InChIKey, and pair per target
([`timesplit/analysis/paired_aimnet.py`](../timesplit/analysis/paired_aimnet.py)
→ [`results/T2_paired_vs_aimnet.csv`](../results/T2_paired_vs_aimnet.csv)):

| Model | L1 retrieval ρ | L1 AIMNet2 ρ | Δ | paired p | wins |
|---|---|---|---|---|---|
| **HypSeek `_rk`** | **+0.292** | −0.030 | **+0.322** | **0.005** | 16/21 |
| **LigUnity-protein** | **+0.227** | −0.030 | **+0.257** | **0.012** | 16/21 |
| **LigUnity-pocket** | +0.173 | −0.030 | +0.203 | **0.044** | 14/21 |
| ConGLUDe | +0.205 | −0.022 | +0.227 | 0.076 | 14/21 |
| LiTENCLIP | +0.136 | −0.030 | +0.166 | 0.089 | 14/21 |
| DrugCLIP | +0.049 | −0.029 | +0.078 | 0.54 | 11/22 |

**On L2, L3 and L4 — 18 model × layer comparisons — not one is significant**
(every p > 0.10).

> **On familiar targets retrieval wins outright; on novel targets the two are
> statistically indistinguishable.**

**What this does not license.** At L3 the AIMNet2 point estimate (+0.197) exceeds
five of the six retrieval models, and it is the layer where the reconstruction
scores best while retrieval scores worst — the shape a crossover would have. **No
paired test reaches significance**, with 17–20 targets of 10 ligands each. Report
the crossover as a direction the data is consistent with, not as a result. The
experiment that would settle it is the same one throughout T2: more ligands per
target, not more targets.

Two checks worth recording. Restricting our models to those 10 ligands moves
HypSeek's L1 from +0.260 to **+0.292**, so the ligand subsetting introduces no
visible bias. And roughly 11 targets per model are dropped because neither
molecule order reproduces the labels — read straight off
[`results/frozen/T3_model_order.csv`](../results/frozen/T3_model_order.csv)
rather than rediscovered, which is what those tables are for.

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
| **Per-target Spearman split by ligand-novelty tier**, with molecule-order validation and the paired familiar/novel test | [`timesplit/analysis/t2_novelty_tiers.py`](../timesplit/analysis/t2_novelty_tiers.py) |
| **CASF split into contaminated / clean targets**, with the spread confounder and Thorndike correction | [`timesplit/analysis/casf_clean_split.py`](../timesplit/analysis/casf_clean_split.py) |
| Measure the CASF ↔ training-set overlap | [`timesplit/analysis/casf_train_overlap.py`](../timesplit/analysis/casf_train_overlap.py) |
| Run CASF under the three training-filter checkpoints (**negative result**) | [`physics/run_casf_thres.sh`](../physics/run_casf_thres.sh) |
| Metric implementations (`spearman`, `kendall_tau`, `r2_score`, `pairwise_accuracy`) | [`eval/metrics.py`](../eval/metrics.py) |

Two of these exist because of a reporting problem rather than a modelling one:
`patch_fep_save.py` adds one line so raw scores survive, and
`fep_recover_preds.py` reconstructs them as `pocket_emb @ mol_emb.T` followed by
a max over pockets — byte-identical to the official computation, so nothing had
to be re-run.
