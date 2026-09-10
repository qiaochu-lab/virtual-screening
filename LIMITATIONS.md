# Limitations

Every known reason a number in this repository could be wrong, or could be read
as saying more than it does. Collected in one place so a reviewer does not have
to reconstruct it from six task documents.

Ordered by how much they affect the headline claims.

---

## 1. The control layer is contaminated — measured, and it does not move the result

T3's layers come from a **time split** — records deposited after 2024-12. A 2025
database record does not mean the measurement is new: the pair may have been
measured years earlier and only re-entered, or entered a second database.

Pair-level check against the training sets
([`standard/check_pair_contamination.py`](standard/check_pair_contamination.py)):

| Layer | Records | (target, ligand) pair already in training | Molecule seen (with a different target) |
|---|---|---|---|
| L1 | 30,176 | **6,320 — 20.9%** | 34.4% |
| L2 | 131,100 | 18 — 0.01% | 0.0% |
| L3 | 7,698 | 0 | 5.7% |
| L4 | 49,070 | 0 | 6.2% |

**So the contaminated actives were deleted and every model re-scored**
([`timesplit/analysis/score_t3_clean.py`](timesplit/analysis/score_t3_clean.py) —
no GPU needed, the per-molecule scores are on disk; ~3,900 actives removed from
L1 per model, roughly a third of them):

| Model | L1 EF1% before → after | L1→L4 decay before → after |
|---|---|---|
| DrugCLIP | 18.80 → 19.25 | −63.9% → **−64.7%** |
| BindCLIP-randneg | 19.12 → 19.31 | −70.3% → −70.6% |
| BindCLIP-hardneg | 17.81 → 17.84 | −66.3% → −65.8% |
| LigUnity-pocket | 35.24 → 36.35 | −76.2% → −77.6% |
| LigUnity-protein | 39.18 → 40.31 | −77.4% → −78.7% |
| LiTENCLIP | 32.37 → 33.10 | −73.9% → −75.0% |
| HypSeek `_rk` | 36.63 → 37.90 | −80.0% → −80.9% |
| ConGLUDe | 13.63 → **12.99** | −71.6% → **−70.2%** |
| ConPLex | 7.66 → **6.67** | −73.3% → **−69.4%** |

**The headline decay survives.** For the seven structure-based models the
contaminated pairs were, if anything, ranked slightly *worse* than average —
removing them nudges L1 up and the decay grows. Only the two weakest,
sequence-based models were genuinely helped by them, and even there the decay
moves by 1.4–3.9 points, not by a category.

That split is itself informative: the pairs were identified against LigUnity's
training set, so they are genuinely seen data for the LigUnity-family models —
and those models show no memorisation benefit on them. The models that *do*
benefit are the ones that never saw them, which points at the molecules being
intrinsically easy (well-studied, prototypical actives) rather than at leakage.

⚠️ Residual caveat: contamination is measured against the **one** training set we
have in usable form. Records that predate the cutoff in some other database, or
in the three undisclosed training sets, are not covered.

## 2. Layer labels are defined by one model's training set

"Target seen / unseen" is judged against the **affinity half** of LigUnity's
training data (`train_label_blend_seq_full.json`, 2,196 UniProt). DrugCLIP and
BindCLIP trained on 16,744 PDB entries → 4,098 UniProt.

⚠️ **Those are not two rival corpora**, which an earlier version of this section
implied by quoting an overlap of 881. LigUnity's training also reads a *structure
half* (`train_label_pdbbind_seq.json`) whose 16,744 PDB entries are **exactly**
DrugCLIP's set. So the four affinity-trained models saw everything the other
three saw, plus the affinity half — see [§24](#24-a-published-finding-was-retracted-the-two-training-sets-are-nested).
The 881 figure compared only the affinity half against DrugCLIP's set and should
not be quoted.
([`standard/quantify_train_union.py`](standard/quantify_train_union.py)):

| Layer | Eval targets | Also in DrugCLIP's training set |
|---|---|---|
| L1 | 349 | 66.2% |
| L2 | 488 | 63.3% |
| L3 | 53 | **24.5%** |
| L4 | 254 | **19.3%** |

**Consequence:** for DrugCLIP and BindCLIP, one L3 target in four and one L4
target in five is not actually new, so **their measured decay understates the
true decay**. The cross-model comparison of *absolute* L4 values is affected;
the within-model L1→L4 gradient is not.

## 3. One model's training set is unavailable

**ConGLUDe** is the only remaining gap. Its list is on Zenodo
(`LB_train_val.zip`, record 20354834), which returned 504 on every attempt; its
layer labels are inherited from LigUnity's split and are approximate.

**SPRINT is no longer in this group.** `merged_data.zip` on the MERGED release
yields **336 training UniProt accessions** — an order of magnitude fewer proteins
than either pocket-family set. Only 33 of the 350-quota subset's targets fall
inside it, so its own seen/unseen split has almost no power (p = 0.84) and its
layer labels remain effectively inherited.

**ConPLex is no longer in this group.** It publishes training sequences without
accessions, which is enough: reverse-looking them up with mmseqs (identity, both
coverages ≥50%, E ≤1e-3) puts its set at **19.4% of T3 targets at ≥95%
identity** — 50% of L1, 51% of L2, 26% of L3, 12% of L4, the same shape as the
other two training sets
([`timesplit/analysis/conplex_train_coverage.py`](timesplit/analysis/conplex_train_coverage.py),
[`results/T3_per_model_audit.csv`](results/T3_per_model_audit.csv)). That check
also turned up a leak this benchmark had been reporting as a result — see
[§23](#23-one-model-was-trained-on-dud-e).

For ConGLUDe the overlap was measurable another way and is large: **37–43% of
L3/L4 targets appear in its training data**. It was checked whether that
mattered — seen and unseen targets are statistically indistinguishable
(p = 0.90), so contamination does not explain its results
([`timesplit/analysis/check_conglude_leak.py`](timesplit/analysis/check_conglude_leak.py),
[`conglude_leak_effect.py`](timesplit/analysis/conglude_leak_effect.py)). "Not
detectable" is not "not present".

## 4. T3 absolute numbers cannot be compared to published values

Decoys are **cross-target real molecules**, not DUD-E-style property-matched
ones, because property matching is the bias under examination. The ratio (1:50)
matches DUD-E but nothing else does.

**Only the decay across layers within this fixed setup is meaningful.** Any use
of a T3 EF value next to a number from a paper is a misreading.

## 5. Pockets in L1/L2 were induced by the test ligand

For targets with experimental structures, the pocket is cut from a complex whose
ligand may be one of the test actives — the pocket is pre-shaped to fit what the
model is being asked to find. This favours structure-based models specifically
in the layers where they score highest.

Measured, not assumed ([`timesplit/analysis/stratify_pocketfit.py`](timesplit/analysis/stratify_pocketfit.py)
→ [`results/T5_pocketfit_all10.txt`](results/T5_pocketfit_all10.txt)). The effect
is real, but **an earlier version of this section overstated how specific it is
to structure models, because it had only been run on two models** — ConGLUDe as
the "structure" arm and ConPLex as the control. Run on all ten:

| | L1 (where the split is defined) | L2 |
|---|---|---|
| seven pocket models | **6 of 7 significant** (p = 0.0025–0.038) | **7 of 7 significant** |
| LigUnity-protein (sequence) | 0.809 — clean | **0.0002 — significant** |
| ConPLex (sequence) | 0.753 — clean | 0.420 — clean |
| SPRINT (SaProt sequence) | 0.988 — clean | 0.260 — clean |

**At L1 the negative control holds cleanly**: all three non-pocket models show
nothing while six of seven pocket models do. **At L2 it breaks** — LigUnity-protein
reads only sequence and still shows the effect at p = 0.0002, and p = 0.0008 was
exactly the number this section used to quote. So the L2 effect is **not**
structure-specific. The likely reading is that the stratifier (co-crystal ligand
vs test ligand similarity) is partly a *ligand-familiarity* variable, which a
sequence model can exploit through the target's known chemistry.

Correcting the decay by using only the low-similarity half of L1 moves it by
**0–8 points**, and not systematically more for pocket models — SPRINT, which
reads no pocket, shows the largest shift (−46% → −38%) while LigUnity-protein
shows none (−77% → −77%). HypSeek moves −80% → −76%, DrugCLIP −64% → −57%.
**The conclusion stands and the magnitude shifts by less than the earlier
"−72% → −67%" implied; what does not survive is "only structure models".**

This control is computed on T3, where ConPLex is not contaminated. The same
argument must not be made from DUD-E — see [§23](#23-one-model-was-trained-on-dud-e).

## 6. L3 is small

53 targets in the dataset, 48–49 in the scored runs. Per-class breakdowns within
L3 are not interpretable, and its confidence intervals are wide. L4 (254/226) is
the layer to trust for "unseen target" claims.

## 7. SPRINT now has all four layers — the earlier limit was our own bug

This section previously recorded that SPRINT could not run L1/L2 because the
loader exhausted shared memory at ~146,000 molecules, and treated that as a
scale limit of the model. **It was not.** Four separate faults were stacked:
chunking disabled by a default of `20000**9`, `--num-workers 0` falling back to
`cpu_count()` (104 workers spawned per featurize call), a cached feature shape
mismatch, and PyTorch's default file-descriptor sharing strategy leaking one FD
per shared tensor. Details in [`PATCHES.md`](PATCHES.md).

With those fixed, SPRINT completes all four T3 layers (L1 282, L2 386, L3 39,
L4 202 targets) and all three T1 benchmarks. The results are weak — T3 AUROC
0.579 → 0.523, EF1% 2.4 → 1.6 — but they exist, and the weakness is now a
property of the model rather than of our tooling.

**The lesson worth keeping:** a documented limitation that originates in a
crash, not in a measurement, deserves one more look before it is written down.
This one stood for two weeks and cost a 20-hour run that wrote nothing.

## 8. Structure-source comparison is not randomised, and it is not null

Whether a target has an experimental structure is itself non-random — well-studied
targets have them — so the holo group may simply be easier. That caveat stands.
What changed is the result it qualifies.

Across all ten models, four differ at L4 with p < 0.05, **all favouring
experimental structures**, and 8 of 10 point that way (sign test p = 0.109).
LigUnity-pocket loses 46% of its EF1% (11.70 → 6.29). Across the 20 comparisons
only ConGLUDe survives BH-FDR, so the honest reading is a consistent direction
with one firmly established case
([`results/T5_structure_source.csv`](results/T5_structure_source.csv)).

The non-randomness is partly addressed: the two **sequence-only** models never
see a structure, so if target difficulty drove the gap they would show it too.
ConPLex (p = 0.66) and LigUnity-protein (p = 0.90) show nothing. That does not
make the comparison randomised, but it removes the most obvious confound.

⚠️ An earlier version of this section, and README finding 4, said predicted
structures substitute with no significant difference. That rested on two models
selected by accident — see [`PATCHES.md`](PATCHES.md).

## 9. Pockets over 511 atoms are randomly cropped

The models cap pocket size at 511 atoms and, above that, keep a center-weighted
**random** sample. At the 6 Å main setting this affects 1 pocket in 1,904 —
negligible. At 8 Å it is 10.8%, which is why the 8 Å result carries its own
truncation control (over-cap −58.8% vs under-cap −59.7%, i.e. not an artefact).

## 10. Affinity labels merge assay types

pAffinity is built from Ki / Kd / IC50 / EC50 (ChEMBL) and Ki-then-IC50
(BindingDB), pooled across labs and assay formats. IC50 depends on substrate
concentration and is not directly comparable to Ki even for the same complex.

**Consequence:** within-target affinity ordering has an irreducible noise floor.
This bounds what any method can achieve on T2 and is part of why T3's ranking
correlations are near zero — the FEP benchmarks, whose values come from single
consistent assays, are the cleaner test.

## 11. Physics reference values are Uni-FEP, not FEP+

The reference row in T2 (mean Kendall 0.503, RMSE 0.65–1.79 kcal/mol) comes from
`dptech-corp/Uni-FEP-Benchmarks` — the same family of free-energy methods, with
ligand counts verified system by system, but **not** Schrödinger FEP+. It is
indicative of magnitude and must not be cited as "FEP+ results".

## 12. Checkpoint selection is not under our control

| Model | What is public | Consequence |
|---|---|---|
| HypSeek | only the ranking weight `_rk` | its screening numbers may understate it; the screening weight would have to be trained |
| LiTENCLIP | one weight (`bedroc_0.50`) | a `bedroc_0.58` variant is referenced upstream and was not obtained |
| LigUnity | `_vs`, plus `_0.3` / `_0.8` variants filtered by training-set sequence distance | only the plain `_vs` was evaluated |

**Resolved since:** HypSeek's T3 runs used a 256-atom pocket cap while everything
else used 511, and 19.7% of 6 Å pockets exceed 256. Re-running at 511 changes
nothing material (L1 AUROC 0.923 → 0.924, L4 EF1% 7.34 → 7.34), so the
inconsistency is closed by measurement. See [`tasks/T3-time-split.md`](tasks/T3-time-split.md).

More generally, released checkpoints are chosen with DUD-E / LIT-PCBA scores in
view, so those benchmarks measure a selection decision as well as a model. This
is the motivation for T3, and it does not go away inside T3 — it only stops
applying to the *targets*, not to the weights.

## 13. Coverage gaps

- **T1: all 10 models run** on DUD-E, DEKOIS and LIT-PCBA. Target counts differ
  slightly by model (ProtBert's 2,000-residue limit, foldseek failures) and are
  recorded per row in [`results/T1_main.csv`](results/T1_main.csv). ⚠️ ConPLex's
  DUD-E column is not comparable — see [§23](#23-one-model-was-trained-on-dud-e).
  CASF-2016 exists for LigUnity ×2 only — the two forks' CASF code path calls
  their own model with the wrong signature.
- **T4: not started.**
- **T5: apo structures and MD conformers not tested** — only experimental vs
  predicted holo, and pocket cutoff.
- **T2: CASF-2016 present but not run.**
- **T6: the head-to-head is done** (461/461 ligands, Boltz-2 mean ρ +0.615), but
  only one physics-side method has been run. Docking and an actual free-energy
  method are still absent, and the cascade-rerank experiment has not started.

## 14. Boltz-2 structural limits

The affinity module rejects ligands over 128 atoms, and the predictor has a
~1170-residue limit. Macrocycles, peptides and large multi-domain proteins are
therefore systematically absent or truncated to a binding domain. Truncation was
validated against annotated binding sites (0% of truncations miss the site after
the fix), but a truncated protein is still not the full protein.

## 15. Boltz-2 sampling settings — tested, not assumed

Boltz-2 has two independent sampling controls and they are easy to confuse:

| Flag | Stage | Default | Ours |
|---|---|---|---|
| `--diffusion_samples` | structure | 1 | **1**, later **5** (see below) |
| `--diffusion_samples_affinity` | affinity | **5** | 5 (never overridden) |
| `--sampling_steps_affinity` | affinity | 200 | 200 |

Two things follow. The affinity score was **never** a single-sample prediction —
the affinity model always ran its own 5-sample diffusion. And extra structure
samples do **not** reach the affinity model as multiple poses: the structure
stage ranks its samples by confidence and passes only the rank-0 structure on,
so raising N buys best-of-N *selection*, not multi-pose rescoring. Boltz-2
cannot be handed externally generated poses at all — a point worth stating
because "multi-pose" invites the docking reading.

The rerank runs used `--diffusion_samples 1`, and the objection that a single
unfiltered draw could be a bad pose is a fair one. It was tested rather than
argued: the same 750 complexes were rerun at N=5, giving a paired comparison
over the 749 scored in both. AUROC moved by 0.002 with every p-value above 0.9
([`results/T6_rerank4.csv`](results/T6_rerank4.csv)). Structure sampling quality
is not what limits the rerank result.

**What is still untested:** the binding site was never supplied as a `pocket`
constraint, so Boltz-2 located it itself while every retrieval model was handed
a 6 Å pocket. That asymmetry is real and favours the retrieval side. It is
another route to a better input structure, and the N=5 result predicts it would
change little — but that is a prediction, not a measurement.

## 16. Checkpoint selection is not symmetric across models

Every retrieval model here ran a screening-selected checkpoint except HypSeek,
which ran `_rk`, selected on FEP ranking — the only weight its authors released.
Comparing it to models represented by their screening weights is not apples to
apples.

Measured against the **official** `_vs`, which the author released on
2026-09-07 in [issue #4](https://github.com/jianhuiwemi/HypSeek/issues/4):
`_rk` is the stronger screening weight on all seven measurements — DUD-E 56.39
vs 51.41, DEKOIS 28.83 vs 25.52, LIT-PCBA 8.34 vs 6.82, and every T3 layer. So
the asymmetry does not flatter HypSeek's screening numbers.

⚠️ An earlier version of this section made the same claim from a **self-trained**
`_vs` (T3 L1 EF1% 22.2 against 36.6). That weight has a diagnosed training defect
— a contrastive negative pool of 4 against the official 24 — so it could not
support the argument, and README finding 14 had already retracted the conclusion
that rested on it. The official weight now does support it. The per-task
checkpoint matrix is in [`MODELS.md`](MODELS.md).

## 17. Per-target actives counts vary by two orders of magnitude

T3 requires ≥10 actives per target; the medians are 24–66 by layer and the
maxima reach the thousands. On a 10-active target the top 1% is 6 slots, so one
additional hit moves EF@1% by **8.5** — against layer means of 8–39. Targets at
the floor are therefore very noisy, and the means weight them equally with
targets measured a hundred times more precisely
([`figures/fig4_actives_per_target.png`](figures/fig4_actives_per_target.png)).

**Tested by raising the floor to 20, 30 and 50**
([`results/T3_actives_gradient.csv`](results/T3_actives_gradient.csv)):

- The **L1→L4 decay is stable** — every model stays within a few points of its
  ≥10 value, DrugCLIP drifting most (68% → 59%). The headline finding survives.
- **L1, L2 and L4 absolute levels are flat**; **L3 is not.** It drops ~17% the
  moment the floor rises (EF1% 17.81 → 14.84) and stays down, on all four
  metrics. This is a real effect rather than noise from a shrinking sample:
  within L3, actives count correlates negatively with score (Spearman −0.26 to
  −0.44), and that correlation is absent in every other layer. It shows up only
  in the four PocketAffDB-trained models, not DrugCLIP. Since L3 has just 48
  usable targets, at ≥50 only 20 remain. **Quote L3 with the floor stated.**
- **Model orderings hold under BEDROC and PR-AUC and not under EF1% or AUROC.**
  At L4 the EF1% second and third place swap at every floor above 10.

The floor was left at 10 in the main tables — raising it would cost 27–56% of
targets and L3 outright — and the gradient is reported alongside instead. No
weighting by actives count is applied; bootstrap intervals
([`results/T3_main_ci.csv`](results/T3_main_ci.csv)) resample targets and carry
the variance.

- **Raising the floor deletes protein classes.** Well-studied targets have more
  actives, so a floor selects for them. Large classes hold their share (enzymes
  28% → 31%, kinases 26% → 28% at ≥50) but transporters fall 23 → 3, nuclear
  receptors 22 → 6 and P450 12 → 2. **The per-class finding cannot be made on a
  ≥50 subset**, which is part of why the floor was left at 10
  ([`results/T3_actives_floor_classes.csv`](results/T3_actives_floor_classes.csv)).

There is no industry floor to appeal to: DUD-E carries 100+ actives on 82% of
targets, but **DEKOIS gives every target exactly 40 and none reaches 50**, and it
has been standard for over a decade. What DEKOIS has is uniformity — every target
identical, so its per-target values are directly comparable. DUD-E spans 15×.
**Ours spans 326×**, and that spread, not the absolute count, is what makes our
per-target precision unequal.

The decoy ratio, by contrast, is **not** a source of incomparability: 1,143 of
1,144 targets sit at exactly 50.0×, the single exception at 43.6×.

## 18. One analysis bug reached the README before it was caught

Every analysis that joined the score arrays to *external* per-molecule data —
affinities, assay types, contamination flags — used the wrong molecule order for
seven of nine models, because LMDB cursor order is lexicographic and we read by
numeric index. It produced a headline claim ("ranking ability is zero on T3")
that was withdrawn on 2026-08-21. Details, and the check that now guards it, in
[`PATCHES.md`](PATCHES.md).

Metrics computed from `(scores, labels)` alone — T1, T3 and T5 in their entirety
— are unaffected, since both arrays come from the model in the same order.

**Standing implication for a reviewer:** treat any number that joins model output
to an external per-molecule attribute as needing the ordering check
([`timesplit/analysis/verify_order.py`](timesplit/analysis/verify_order.py))
before it is quoted.

## 19. Statistical practice

Multiple comparisons across models × layers × classes are corrected with
Benjamini–Hochberg (step-up). Bootstrap confidence intervals resample **targets**,
not molecules — pooling molecules across targets understates variance and
changes what EF means.

Intervals now exist for the whole main table
([`results/T3_main_ci.csv`](results/T3_main_ci.csv)) and they carry a warning:
**at L4 the top four models' intervals overlap almost entirely**, so their
ordering is not resolvable. Quote the decay within a model, and the gap between
the strong group and the weak one; do not quote "model A beats model B at L4".
L3 (48 targets) should not be used for model comparison at all.

---

## 18. The dataset-v2 subset is thin where it matters most

The ≥50-actives + VSDS-vd-matched subset has **54 targets at L4** (116 at the
≥50 filter alone, 254 in the full set). Relabelling just **6** of them — the
ones with ≥40% training homology — moves L4 EF1% by 20–25% and the headline
decay from −69% to −78%.

A layer where six targets swing the main conclusion is not a layer that
supports a precise effect size. Raising the quota from 250 to 350 restores L4
to 75 targets at the cost of class deviation rising from 4.8pp to 7.9pp; that
trade has not been decided.

L3 is worse: **20 targets, and that is the entire supply** at ≥50 actives. No
choice of quota fixes it.

## 19. Two protein classes cannot be filled at all

At ≥50 actives the pool holds **6 nuclear receptors and 2 cytochrome P450s**;
matching VSDS-vd's proportions at 147 targets would need 17 and 9. Taking every
one still falls short.

This is structural, not a filtering choice: both families were characterised
early and are small, so almost no new members appear after the 2024-12 cutoff —
**L3 and L4 contain zero of either**. A time-split benchmark cannot be class-balanced
against a benchmark that is free to choose its targets. Forcing VSDS-vd's exact
proportions would cap the whole dataset at 49 targets (P450 is the binding
constraint), discarding 129 of 141 available kinases to do it.

## 20. L1 is closer to a memorisation test than an enrichment test

32.1% of L1 actives are exact InChIKey matches to training ligands (decoy
background 3.7%), their median Tanimoto to the training set is 0.727, and 53.9%
sit above 0.7. A target-conditioned ligand-similarity oracle — no protein input,
but it reads the target's other known actives — reaches 98.7% of the theoretical
EF ceiling there.

L1 remains valid as the **control layer** — it demonstrates the pipeline is
wired correctly — but an L1 enrichment number should not be read as evidence of
pocket–ligand understanding. L2, L3 and L4 are clean by the same measurements:
their actives are *more* novel than the cross-target decoys.

## 21. The docking rerank was run on targets that are no longer in the dataset

The smina control docked 20 targets picked from the full L4. **5 of those 20
survive into the v2 subset**, and only 9 of the original 20 had enough actives
inside the top-200 to be scored at all. Docking scores cannot be re-aggregated
onto targets that were never docked, so that control currently describes the
full L4, not the published subset. Undecided whether to re-dock (~33 CPU-hours),
state the discrepancy, or demote the control.

## 22. The target swap only used unrelated substitutes

Every substitute was drawn at random from the same layer, subject to a pocket-size
match. So the result establishes that replacing a target with an **unrelated**
one destroys performance — it does not distinguish whether the model recognises
the *specific target* or its *family*. A within-family swap (same CD-HIT 40%
cluster) would separate those, and has not been run.

This matters for interpretation: "the model uses target information" is
supported, but "the model resolves individual targets" is not.

Only L1 and L4 were run. Each target drew 3 substitutes, and substitutes were not
deduplicated across targets.

## 23. One model was trained on DUD-E

ConPLex's contrastive objective draws its negatives from DUD-E decoys — that is
the method, not an ablation, and `contrastive: True` is the shipped default. The
training targets are the `train` half of `dataset/DUDe/dude_*_train_test_split.csv`;
the repository publishes two such splits whose union is **40 of the 102 targets
our T1 scores**.

We ran ConPLex on all 102 for eleven months and reported EF1% 18.70 as its
screening performance. On the 45 targets neither split file ever names, it is
**4.83**, with AUROC 0.577. The other nine models move between −10% and +7% on
the same restriction.

**Why it went unnoticed:** nothing about the run looks wrong. The model loads,
scores every target, and lands in a plausible position — last among the
dual-tower models, which is what a sequence-only model is expected to do. The
number is only visibly wrong once the evaluation set is split by the model's own
training list, and that list lives in a CSV in the repository rather than in the
paper or the model card.

**What it implies for the other two unavailable sets.** ConGLUDe and SPRINT
publish no usable target list. ConGLUDe was checked another way and shows no
detectable effect (p = 0.90), and SPRINT's DUD-E score is too low to hide one.
But the general point stands: **a benchmark cannot certify a model whose
training set it cannot see**, and three of ten models here were in that position
until this check. Detail and reproduction in
[`tasks/T1-enrichment.md`](tasks/T1-enrichment.md#conplex-trained-on-dud-e).

## 24. A published finding was retracted: the two training sets are nested

For one afternoon, finding 18 read: *"PocketAffDB membership is worth ~2.4 rank
places; `train_no_test_af` membership is worth nothing measurable."* It was
pushed, and it was wrong.

The error: LigUnity-family training reads **two** label files
(`unimol/tasks/train_task.py:523-524`) — an affinity half
(`train_label_blend_seq_full.json`, 2,196 UniProt) and a structure half
(`train_label_pdbbind_seq.json`, 3,468 UniProt / 16,744 PDB entries). Only the
first had ever been counted. The second is not separate data: its 16,744 PDB
entries are **exactly** DrugCLIP's `train_no_test_af` — 16,744 on each side,
intersection 16,744, neither exclusive.

**What is nested is the models, not the files.** The two label files overlap in
only 817 UniProt; 1,379 targets are affinity-only and 2,651 structure-only, so
neither file contains the other. What nests is what each *group of models* saw:
DrugCLIP and the BindCLIP pair saw the structure half; LigUnity ×2, LiTENCLIP
and HypSeek saw that **plus** the affinity half. This distinction matters for
reading the crossover below — the "affinity-only" cell is not empty, it holds
36 targets in the 350-quota subset and 135 across all common targets.

The crossover that produced "2.4 rank places" compared cells defined as "in A
only" and "in B only", which presumes the sets are disjoint. Recomputed against
the two label halves, the effect survives but weakens: **1.73 places, p = 0.0095**
(exact permutation, 2 of C(10,4) = 210 splits), against a perfect ten-model
separation in the first version. The structure-only cell holds **7 records** in
the 350-quota subset, and two models sit on the wrong side of zero — HypSeek
+0.12 among the four, ConGLUDe −0.19 among the six.

**Repeating it on all 840 common records removes that exception.** The
structure-only cell grows from 7 to 28, and HypSeek moves from +0.12 to −0.37,
putting all four affinity-trained models on the same side. The group difference
goes from 1.73 to **1.85 places, and p from 2/210 to 1/210 — a perfect ten-model
separation**. So the effect strengthens slightly across a threefold change in
sample size while the one anomaly does not survive it, which is what a
small-sample artefact looks like.
([`results/T3_train_set_crossover_full.csv`](results/T3_train_set_crossover_full.csv),
`train_set_crossover.py --subset all`)

Both the subset and the full recomputation were **independently reproduced by a
second agent** from the raw score arrays, along a separate code path: same four
cell counts, same ten per-model values to two decimals, same permutation p. That
check is now required before any published number is retracted or rewritten —
a rule this retraction is the reason for.

**The measurements were never wrong; the attribution was.** What is actually
shown is narrower: on targets reachable only through the affinity-labelled half,
the four models trained on it rank ~1.5 places better. On the structure half —
which all seven pocket models trained on — no group stands out, which is what
should happen when everyone has seen the data.

**Two things this changes elsewhere.** The layer labels are contaminated: 4 of
19 L3 targets and 8 of 75 L4 targets in the 350-quota subset were seen by the
four models through the structure half, which **understates** their decay.
And `target_mirroring.py` compares against 2,196 targets where it should compare
against the 4,847-target union, so its homology hit rate is an underestimate.

**What went wrong procedurally:** the training set was read from the file the
project had always used, not from the training code. One `grep` for
`train_label` in `unimol/tasks/` would have shown two files on adjacent lines.
The check is now: *before defining a model's training set, read the loader, not
the data directory.*

## 25. What the per-model layering does and does not settle

[`tasks/T3-leakage.md` §6](tasks/T3-leakage.md) re-cuts the layers using each
model's own training set. Two limits on how far that result reaches.

**The A-only cell holds 13 targets.** The crossover that separates PocketAffDB
membership from `train_no_test_af` membership compares targets only one set
contains. B-only has 32; A-only has 13. Two of the four PocketAffDB models reach
significance individually (p = 0.009, 0.005); the other two are directionally
consistent but not significant. The evidence that survives is at **group** level
— all four B models on one side of zero, all six others on the other, exact
p = 0.0048 — not per model. Widening the A-only cell needs a larger subset, not
a better test.

**Seen/unseen is decided by sequence, and sequence is the weakest layer of
protein similarity.** Membership here means exact UniProt match, or ≥95%
sequence identity for the two sets reconstructed by mmseqs. Pocket-level and
interaction-pattern-level similarity can be high where sequence identity is low,
so some targets counted as "unseen" are near-neighbours of training targets by a
criterion this analysis never applies. That biases every decay number in the
same direction — **understating** it — and it is orthogonal to the question §6
answers. §6 settles *whose training set the labels come from*; it does not
settle *at what level similarity should be measured*. The second question needs
its own experiment.

**That experiment has now been run, and L4 survives it.** Comajuncosa-Creus et
al. (*Nat Commun* 2024) released PocketVec descriptors for the human pocketome —
49,511 pockets over 10,539 UniProts, each a 128-dimensional ranking of how a
fixed lead-like library docks into that pocket. They report **over 3.5 million**
pocket pairs at distance < 0.17 whose proteins have TM-score < 0.35 and sequence
identity < 30%, which is precisely the failure mode this section worries about.

Matching our targets against those descriptors
([`timesplit/analysis/pocket_neighbors.py`](timesplit/analysis/pocket_neighbors.py)
→ [`results/T3_pocket_neighbors.csv`](results/T3_pocket_neighbors.csv)): for each
test target take its minimum cosine distance to **(A)** every pocket of every
training target, and to **(B)** an equal number of pockets sampled from human
proteins in neither the training set nor T3. Paired per target:

| Layer | n | to training | to control | Δ | paired p | closer to training |
|---|---|---|---|---|---|---|
| L1 | 265 | 0.0983 | 0.1132 | **−0.0109** | 1.7e-21 | 74% |
| L2 | 342 | 0.1017 | 0.1140 | −0.0084 | 3.6e-23 | 71% |
| L3 | 24 | 0.1034 | 0.1087 | −0.0059 | 0.0115 | 75% |
| **L4** | **123** | **0.1149** | **0.1144** | **+0.0013** | **0.153** | **44%** |

**L4 targets are no closer to training pockets than to arbitrary human pockets.**
The gradient across layers is monotone and matches the layer definitions exactly,
which is also a check on the layering itself. L3 retains a small but significant
proximity — expected, since L3 is defined as *family seen, target unseen*.

**The control is what makes this readable.** A first pass reported the fraction
of targets below the authors' 0.17 similarity threshold and got **100% at every
layer**, which says nothing: taking a minimum over 9,726 reference pockets is an
extreme-value statistic, and PocketVec vectors are rankings of 1–128, so two
*random* permutations sit at cosine distance ≈ 0.253 rather than 1. Absolute
thresholds do not survive a min-over-thousands; only the paired contrast does.

Three limits on the claim. PocketVec covers the **human** proteome only, so 123
of 254 L4 targets (48%) have descriptors and the cross-species orthologs we know
are in L4 are exactly the ones missing — this is a lower bound on any leakage.
The effect sizes are small in absolute terms (~0.01 against a random baseline of
0.253); the result rests on the pairing, not on the distances. And these are the
authors' Pfam-domain pockets, not our 6 Å ligand-induced ones, so the question
answered is "does this protein have a pocket resembling a training protein's"
rather than "is the pocket our models were shown similar".

**One methodological note, recorded because the first attempt was wrong.**
Difficulty was first divided out per target as
`EF(model, t) / median(EF(other models, t))`. That ratio is unusable: the
denominator approaches zero on hard targets and the quotient explodes, giving
DrugCLIP a median fold of 1.85 and a mean fold of 0.57 — opposite directions
from the same data. The reported analysis uses within-target **rank** across the
ten models instead, which cancels difficulty without dividing by anything. The
ratio version is kept in
[`results/T3_per_model_layers_ctrl.csv`](results/T3_per_model_layers_ctrl.csv)
only so the discrepancy is inspectable.

## 26. Half of CASF-2016 is in the PocketAffDB training set

**148 of CASF-2016's 285 complexes (51.9%) appear in PocketAffDB by exact PDB
ID.** Not "a similar structure" — the same deposition. PocketAffDB stores each
assay's pockets as `2q5sA--2q5s_NZA_A_1.lmdb`, whose first four characters are
the PDB entry, so the two sets join directly. The match is not an artefact of
that truncation: 123 of the 148 also agree on UniProt, e.g. CASF `4eky`
(P00489) against training `4ekyA--4eky_D1J_A_1.lmdb` (P00489).

Two weaker overlaps, for scale: **49 of 68 CASF UniProt accessions (72.1%)** are
in PocketAffDB, and **82 of 276 CASF ligands (29.7%)** match a training ligand by
exact InChIKey.

Four of the five models measured on CASF here — LigUnity ×2, LiTENCLIP and
HypSeek — train on PocketAffDB, and T2 uses the *ranking* checkpoint. LigUnity's
model card states that test proteins were removed when training the **screening**
weight; it says nothing about the ranking weight, and this measurement is
consistent with no removal having happened there.

This does not yet change any reported number. What it means is that **CASF
cannot be read as a held-out test for those four models** until the clean and
contaminated halves are scored separately. That rescoring is in progress in the
T2 workstream; the overlap itself is reproducible with
[`timesplit/analysis/casf_train_overlap.py`](timesplit/analysis/casf_train_overlap.py)
→ [`results/T2_casf_train_overlap.txt`](results/T2_casf_train_overlap.txt).

For context, Graber et al. (*Nat Mach Intell*, 2025, doi
10.1038/s42256-025-01124-5) report 49% of CASF having a near neighbour in
**PDBbind**, using a combined structure/ligand/affinity criterion. The 51.9%
here is a different and stricter thing — exact identity, against the training set
our models actually used.
