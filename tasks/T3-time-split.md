# T3 — Time-Split Generalization

**Question:** how much of the reported enrichment survives on targets that
appeared **after** the models' training data was assembled?

**Status:** complete. Nine models × four layers, 1,044 targets. This is the
project's centrepiece.

---

## Why we built our own dataset

Published numbers come from DUD-E / LIT-PCBA / DEKOIS. Three problems with
using those to judge generalization:

1. **DUD-E's decoys are synthetic** — property-matched but topologically
   dissimilar. A 2019 analysis showed a model can score well from the ligand
   alone, ignoring the protein.
2. **Those benchmarks participated in model selection.** DUD-E and LIT-PCBA are
   difficult to optimize together, so a released checkpoint is a trade-off
   picked with those scores in view. Scoring it on the same benchmarks partly
   measures that selection rather than the model.
3. **No temporal holdout.** Everything in existing benchmarks could have been
   seen during training.

Data postdating the training cutoff is immune to all three — it cannot have been
trained on, and cannot have influenced checkpoint selection.

## The dataset

**Cutoff 2024-12.** Not chosen arbitrarily: we read the `version` field out of
LigUnity's own training labels (max = 34, with 565 assays from ChEMBL v34) and
combined with BindingDB 2024m5. Cutting earlier would make 2024 data "not
future" for these models and invalidate the whole result.

**Sources:** ChEMBL 37 + BindingDB 202608, filtered to post-cutoff deposits,
set-differenced against the training sets, deduplicated by InChIKey.

**Four layers by novelty** — the *gradient* is the measurement, not any single
number:

| Layer | Target | Ligand | Eval targets |
|---|---|---|---|
| L1 | seen | new | 349 |
| L2 | seen | new Bemis–Murcko scaffold | 488 |
| L3 | **unseen** | same protein family seen | 53 |
| L4 | **unseen**, family unseen | | 254 |

**L1 is the control.** If a model is near chance on L1, our pipeline is broken,
not the model. Every conclusion below depends on L1 being clearly above chance.

**Actives:** measured pAffinity ≥ 6 (1 µM), InChIKey-deduplicated, ≥10 per target.

**Decoys — cross-target, not property-matched.** Real drug-like molecules from
the T3 pool that are active on *dissimilar* targets, 1:50. Excluded: the
target's own actives, molecules active on any target in the same mmseqs 40%
cluster, and molecules sharing a scaffold with the target's actives.

We deliberately avoid DUD-E-style property matching since that is the bias in
question. **Cost: absolute numbers are not comparable to published values.**
Only the L1→L4 decay within one fixed setup means anything.

The three exclusions can leave a target short of 50×, in which case the builder
takes what it can and records the true ratio — EF depends on library size, so
that number has to travel with the result. In practice the pool was large enough
that this almost never fired: **1,143 of 1,144 targets sit at exactly 50.0×**,
the one exception at 43.6×. Per-target EF values are therefore comparable to
each other, which they would not be under a varying ratio.

**What does vary is the number of actives, and it limits EF's resolution.** The
floor is 10 actives per target; the medians are 24 (L1), 66 (L2), 34 (L3) and 44
(L4), with maxima in the hundreds to thousands. On a 10-active target the top 1%
is 6 slots, so **one more hit moves EF@1% by 8.5** — against layer means of 8–39.
Small targets are therefore very noisy, and the reported means weight them
equally with targets measured a hundred times more precisely. The bootstrap
intervals in [`results/T3_main_ci.csv`](../results/T3_main_ci.csv) resample
targets and so carry this variance, but no weighting is applied. Raising the
floor or weighting by active count would both be defensible; neither was done,
and the L3 layer (48 targets) is where it matters most.

**Structures:** 1,466 new targets; experimental PDB where available, Boltz-2
predictions otherwise; usable pocket coverage **95.4%**. Pockets are
residue-level at 6 Å, validated against the authors' own published pockets with
**100% coordinate overlap** on five DUD-E targets.

Full construction pipeline and the four things that are easy to get wrong:
[`timesplit/README.md`](../timesplit/README.md).

## Results

Full table: [`results/T3_main.csv`](../results/T3_main.csv). EF1%:

| Model | L1 | L2 | L3 | L4 | decay |
|---|---|---|---|---|---|
| HypSeek (`_rk`) | 36.63 | 23.61 | 13.56 | 7.34 | **−80.0%** |
| LigUnity-protein | 39.18 | 30.20 | 17.81 | 8.83 | −77.4% |
| LigUnity-pocket | 35.24 | 26.39 | 13.90 | 8.39 | −76.2% |
| LiTENCLIP | 32.37 | 23.06 | 12.94 | 8.46 | −73.9% |
| BindCLIP-randneg | 19.12 | 12.86 | 8.21 | 5.68 | −70.3% |
| DrugCLIP | 18.80 | 12.56 | 6.42 | 6.78 | **−63.9%** |
| BindCLIP-hardneg | 17.81 | 12.57 | 7.90 | 6.00 | −66.3% |
| ConGLUDe | 13.63 | 7.75 | 5.36 | 3.87 | −71.6% |
| ConPLex | 7.66 | 3.80 | 3.24 | 2.04 | −73.3% |
| SPRINT | — | — | 1.32 | 1.37 | — |

### Four findings

**1. Absolute performance spans 5×; decay spans 64–80% for everyone.**
This is a property of the method family, not of any one model. It holds even for
HypSeek, the only model not using Euclidean embeddings.

**2. Training data explains the tiers better than architecture does.**
The three models trained on LigUnity's data (LigUnity ×2, LiTENCLIP) sit at
32–39 on L1; the three trained on DrugCLIP's data sit at 17–19. The
architectural differences between them (retrieval augmentation, molecule
encoder) matter less than which corpus they saw.

**2b. Sequence and pocket trade places by benchmark — neither wins
consistently.** LigUnity ships two parallel branches from one release —
`pocket_ranking` (protein side = 3D pocket) and `protein_ranking` (protein side
= amino-acid sequence) — with the same training set, the same ligand encoder and
the same checkpoint-averaging scheme. That makes them a controlled pair, and both
were run over the same targets and candidate sets, so the comparison pairs per
target ([`timesplit/analysis/seq_vs_pocket.py`](../timesplit/analysis/seq_vs_pocket.py),
[`standard/seq_vs_pocket_t1.py`](../standard/seq_vs_pocket_t1.py)).

**On the time-split data, sequence wins where the target was seen and ties where
it was not** ([`results/T3_seq_vs_pocket_per_target.csv`](../results/T3_seq_vs_pocket_per_target.csv)):

| Layer | Metric | seq wins | pocket wins | ties | win rate (ex-ties) | p |
|---|---|---|---|---|---|---|
| **L1** | EF1% | 123 | 56 | 131 | **68.7%** | **7.9e-07** |
| **L1** | AUROC | 177 | 120 | 13 | 59.6% | **2.3e-05** |
| **L2** | EF1% | 202 | 134 | 98 | **60.1%** | **5.5e-07** |
| **L2** | AUROC | 247 | 180 | 7 | 57.8% | **0.00017** |
| L3 | EF1% | 18 | 11 | 19 | 62.1% | 0.10 |
| L3 | AUROC | 27 | 20 | 1 | 57.4% | 0.40 |
| **L4** | EF1% | 62 | **67** | 95 | **48.1%** | **0.93** |
| **L4** | EF5% | 90 | 90 | 44 | **50.0%** | **0.91** |
| **L4** | AUROC | 110 | **114** | 0 | **49.1%** | 0.53 |

**On the standard benchmarks the ordering reverses**
([`results/T1_seq_vs_pocket_per_target.csv`](../results/T1_seq_vs_pocket_per_target.csv)):

| Benchmark | Metric | pocket | sequence | seq wins | pocket wins | ties | win rate | p |
|---|---|---|---|---|---|---|---|---|
| DUD-E | EF1% | 42.57 | 36.69 | 45 | 47 | 10 | 48.9% | 0.08 |
| DUD-E | BEDROC | 0.653 | 0.574 | 49 | 53 | 0 | 48.0% | 0.11 |
| DUD-E | AUROC | 0.892 | 0.887 | 51 | 51 | 0 | 50.0% | 0.53 |
| **DEKOIS** | EF1% | 24.62 | **27.04** | 30 | 22 | 29 | **57.7%** | **0.011** |
| **DEKOIS** | BEDROC | 0.728 | **0.785** | 55 | 26 | 0 | **67.9%** | **0.02** |
| DEKOIS | AUROC | 0.911 | 0.925 | 40 | 40 | 1 | 50.0% | 0.22 |
| **LIT-PCBA** | EF1% | **7.30** | 6.22 | **0** | **10** | 5 | **0.0%** | **0.0051** |
| **LIT-PCBA** | EF5% | **3.10** | 2.18 | 1 | 13 | 1 | **7.1%** | **0.0029** |
| **LIT-PCBA** | BEDROC | **0.088** | 0.075 | 2 | 13 | 0 | **13.3%** | **0.012** |
| LIT-PCBA | AUROC | 0.601 | 0.563 | 5 | 10 | 0 | 33.3% | 0.19 |

**Putting both tables together, there is no consistent winner.** The sequence
branch wins DEKOIS and T3's L1/L2; the pocket branch wins LIT-PCBA on every
early-enrichment metric — 10 of 10 decided targets on EF1% — and DUD-E is a tie
under pairing despite a 5.89 gap in the means. This is the same phenomenon as
finding 3 below, one level down: **not only does model ranking reverse by
benchmark, so does the ranking of two branches of a single model.**

**LIT-PCBA deserves the most weight of the three standard benchmarks** — its
decoys are experimentally confirmed inactives rather than generated — and it is
the one that favours the pocket most decisively. Any claim that a 3D pocket is
dispensable has to answer that.

Three reading notes.

**The per-target table matters more than the means here.** EF@1% is coarse at
this pool size (T3's median pool is ~1,200 candidates, so the top 1% is 12 slots)
and 131 of 310 L1 targets are exact ties, so a mean difference can be carried by
a handful of targets. DUD-E is the clearest case: the pocket branch leads by 5.89
EF1% in the mean and the paired test says 48.9% vs 51.1%, p = 0.08.

**AUROC settles nothing anywhere.** It is non-significant on all three standard
benchmarks (p = 0.19–0.53) and at L3/L4. Only the early-enrichment metrics
separate the two branches — the same point finding 4 makes about models.

**This is not evidence about representations in general.** Both weights were
trained by the authors; we know the branches share a training set, a ligand
encoder and a checkpoint-selection scheme, but not that every training detail
matched, and the paper's reported model is the **ensemble** of both plus an HGNN
post-step, which we did not run. What the pair supports is narrower: the two
protein representations are close enough that which one leads depends on the
evaluation, and on genuinely novel targets they are indistinguishable.

**3. Model ranking reverses by target class.** On kinases ConPLex (pure
sequence) beats ConGLUDe (geometric) 5.76 vs 2.07; on other enzymes it reverses,
4.73 vs 1.28 — both significant after BH-FDR correction. Reporting only the
overall mean would say "ConGLUDe is better", which is wrong per class.

**4. AUROC hides what EF shows.** On L4, ConGLUDe vs ConPLex is AUROC 0.570 vs
0.549 (p=0.37, n.s.) but EF1% 4.00 vs 2.16 (p=0.018, significant). Virtual
screening only cares about the top of the list. Report both.

### Actives per target: does the floor drive anything?

The floor is 10 actives per target and the maxima reach the thousands, so EF's
resolution varies a hundredfold across targets (the step is ~100/A; see
[`figures/fig4_actives_per_target.png`](../figures/fig4_actives_per_target.png)).
The means weight a target that can only score 0 or 8.5 equally with one measured
to 0.15. Raising the floor to 20, 30 and 50 tests whether that matters
([`timesplit/analysis/actives_gradient.py`](../timesplit/analysis/actives_gradient.py),
[`results/T3_actives_gradient.csv`](../results/T3_actives_gradient.csv)). The
same runs add **PR-AUC**, which uses the whole ranking rather than a cutoff and
so is not quantised by the actives count, and which unlike ROC-AUC is sensitive
to the 1:50 imbalance.

**Absolute levels barely move, except at L3** (LigUnity-protein, EF1%, target
count in brackets):

| Layer | ≥10 | ≥20 | ≥30 | ≥50 |
|---|---|---|---|---|
| L1 | 39.18 (310) | 39.50 (180) | 39.03 (131) | 39.23 (82) |
| L2 | 30.20 (434) | 30.49 (366) | 30.83 (327) | 31.31 (258) |
| **L3** | **17.81 (48)** | **14.84 (35)** | **15.56 (26)** | **14.95 (20)** |
| L4 | 8.83 (224) | 8.81 (168) | 8.60 (143) | 9.49 (107) |

**L3 drops ~17% as soon as the floor rises and stays down** — the same shift
appears in BEDROC (0.355 → 0.294), PR-AUC (0.320 → 0.263) and AUROC (0.761 →
0.720). L3's headline number is inflated by its small-actives targets. It is
also the layer with the fewest targets to begin with (48 usable), so at ≥50 only
20 remain and the layer stops supporting conclusions at all. **L3 numbers should
be quoted with the floor stated.** L1, L2 and L4 are flat to within noise.

**The L1→L4 decay — the headline result — is stable:**

| Model | ≥10 | ≥20 | ≥30 | ≥50 |
|---|---|---|---|---|
| HypSeek | 82% | 85% | 83% | 81% |
| LigUnity-protein | 79% | 80% | 80% | 78% |
| LigUnity-pocket | 78% | 79% | 79% | 78% |
| LiTENCLIP | 76% | 78% | 78% | 77% |
| BindCLIP-hardneg | 70% | 75% | 74% | 71% |
| BindCLIP-randneg | 74% | 72% | 70% | 66% |
| DrugCLIP | 68% | 67% | 67% | **59%** |

Every model stays inside a few points of its ≥10 value; DrugCLIP drifts most
(68% → 59%) and even that leaves the finding intact. PR-AUC reproduces the
BEDROC decay closely (top models 76/74/72/70% against 77/75/73/71%), which is
reassuring: two metrics with different failure modes agree on the shape.

**Model rankings are stable under BEDROC and PR-AUC, and not under EF1% or
AUROC.** At L4, EF1%'s top three at the current floor is LigUnity-protein >
LiTENCLIP > LigUnity-pocket; at every higher floor LiTENCLIP and LigUnity-pocket
swap. AUROC's top three moves at both L1 and L4. BEDROC and PR-AUC hold their
top three at all four floors. **Where an ordering matters, prefer BEDROC or
PR-AUC over a single EF cutoff** — the same lesson finding 4 draws from a
different angle.

What was *not* changed: the floor stays at 10 in the main tables, because moving
it would cost 27–56% of targets and L3 outright. The gradient is reported
alongside instead, and the caveat is recorded in
[`LIMITATIONS.md`](../LIMITATIONS.md).

### Robustness checks

| Check | Result |
|---|---|
| **Pocket-fit confound** — L1/L2 pockets are induced-fit to the test ligands (median Tanimoto 0.748 vs 0.12–0.28 for new targets) | Real (significant on L2, p=0.0008) but **only for structure models**; the sequence-model negative control shows nothing (p=0.73/0.45). Correcting moves decay −72% → −67%. Conclusion stands. |
| **Structure quality** — restrict to targets with experimental structures or high-confidence predictions (82.6% of the set) | Decay changes by ≤7 points for all models; ranking unchanged |
| **ConGLUDe contamination** — it was published 2026-01, so its training data may postdate our cutoff | 37–43% of our "new" L3/L4 targets **are** in its training set. But its performance on seen vs unseen targets is indistinguishable (L4: 3.72 vs 3.95, p=0.90), so no measurable inflation. Documented, not excluded. |

### A falsifiable prediction, checked

Before running T3 we predicted from code inspection that LigUnity's H-GNN —
which **queries the training set at inference time** — should decay more than
models that do not retrieve. Ranking came out as predicted (LigUnity ×2 largest
at 76–77%, DrugCLIP/BindCLIP smallest at 64–70%).

⚠️ **This is consistency, not proof.** ConPLex decays 73.3% while doing no
retrieval at all — only 4 points less than LigUnity. Retrieval augmentation
explains part of it, not all.

## Limits to state when reporting

- Absolute numbers are **not** comparable to published values (different decoys)
- L3 has only 53 targets (48–49 evaluated); all conclusions there carry CIs
- Nuclear receptors (1) and P450 (0) are essentially absent from the new-target
  layers — those two families are small and exhaustively studied, so few genuinely
  new targets appear post-cutoff. Structural limitation of a time split, not a
  selection error.
- T3 is **not** a clean holdout for ConGLUDe (see above)

## Physics methods

⚠️ **Not recommended.** 1,044 targets × ~1,250 molecules each. Co-folding is
infeasible; docking is possible but only worth doing on a subset.

---

## Confidence intervals — the L4 ordering is not resolvable

Target-level bootstrap, 2,000 resamples, EF1%
([`../timesplit/analysis/bootstrap_t3.py`](../timesplit/analysis/bootstrap_t3.py),
full table in [`../results/T3_main_ci.csv`](../results/T3_main_ci.csv)):

| Model | L1 | L4 |
|---|---|---|
| LigUnity-protein | 39.18 [37.58, 40.84] | 8.83 [6.86, 10.95] |
| HypSeek `_rk` | 36.63 [34.82, 38.33] | 7.34 [5.60, 9.19] |
| LigUnity-pocket | 35.24 [33.29, 37.16] | 8.39 [6.54, 10.39] |
| LiTENCLIP | 32.37 [30.35, 34.25] | 8.46 [6.65, 10.34] |
| BindCLIP-randneg | 19.12 [17.28, 21.00] | 5.68 [4.23, 7.18] |
| DrugCLIP | 18.80 [16.96, 20.67] | 6.78 [5.26, 8.37] |
| ConGLUDe | 13.63 [12.02, 15.37] | 3.87 [2.76, 5.11] |
| ConPLex | 7.66 [6.56, 8.82] | 2.04 [1.42, 2.73] |

**At L1 the two training-data tiers separate cleanly** — the PocketAffDB group
(32–39) and the DrugCLIP group (17–19) do not come close to overlapping.

**At L4 they do not.** The top four models' intervals overlap almost completely
(6.9–11.0 vs 5.6–9.2 vs 6.5–10.4 vs 6.7–10.3), and DrugCLIP's interval overlaps
all of them. Ranking models by their L4 point estimates is not supported by the
data; what *is* supported is the size of the drop within each model, and the
separation between the strong group and ConGLUDe/ConPLex.

L3 is worse still (48 targets, intervals ±4–5 EF units) and should not be used
for model comparison at all.

## Contamination: measured, removed, and now a first-class table

The time split does not guarantee novelty — 20.9% of L1's (target, ligand) pairs
already exist in the training set. Rather than caveat it, the contaminated
actives are deleted and every model re-scored. That table now ships alongside
the main one ([`../results/T3_main_clean.csv`](../results/T3_main_clean.csv),
produced by [`../timesplit/analysis/export_t3_clean.py`](../timesplit/analysis/export_t3_clean.py)),
so the headline decay is a range rather than a point:

| Model | decay, as measured | decay, contamination removed |
|---|---|---|
| HypSeek `_rk` | −80.0% | **−80.9%** |
| LigUnity-protein | −77.4% | −78.7% |
| LigUnity-pocket | −76.2% | −77.6% |
| LiTENCLIP | −73.9% | −75.0% |
| BindCLIP-randneg | −70.3% | −70.6% |
| DrugCLIP | −63.9% | −64.7% |
| ConGLUDe | −71.6% | −70.2% |
| ConPLex | −73.3% | −69.4% |
| SPRINT | −45.6% | −46.2% |

**Seven of nine models decay *more* after cleaning**, not less: the seen pairs
were ranked slightly worse than average, so removing them lifts L1. Only the two
weakest, sequence-based models move the way contamination would predict, and by
a few points. L3 and L4 are identical in both columns — those layers have zero
contaminated pairs, which is the built-in self-check that the filter works.

⚠️ **One口径 difference from a true rebuild.** We delete indices from the scored
arrays rather than regenerate the eval set and re-run inference. Removing actives
shifts the active:decoy ratio slightly away from 1:50, so EF's denominator moves
a little. A genuine rebuild would require re-running all nine models; the cost is
out of proportion to that difference, but the difference is real and is not
being papered over.

## Pocket-atom cap: the one inconsistency we introduced, and it was harmless

HypSeek was run on T3 with `--max-pocket-atoms 256` (to avoid an OOM at the
time) while every other model, and HypSeek's own T1 runs, used 511. At 6 Å,
**19.7% of pockets exceed 256 atoms** and get center-weighted random cropping —
so a fifth of HypSeek's T3 targets were seen through a partially cropped pocket
that no other model had to deal with.

Re-run at 511 ([`../timesplit/runners/run_t3_hypseek_511.sh`](../timesplit/runners/run_t3_hypseek_511.sh)):

| Layer | AUROC 256 → 511 | EF1% 256 → 511 |
|---|---|---|
| L1 | 0.923 → 0.924 | 36.63 → 36.84 |
| L2 | 0.878 → 0.879 | 23.61 → 23.75 |
| L3 | 0.770 → 0.767 | 13.56 → 13.35 |
| L4 | 0.710 → 0.710 | 7.34 → 7.34 |

**No material difference.** The inconsistency existed and is now closed by
measurement rather than by argument; HypSeek's position in the tables does not
depend on it.

## Code

Dataset construction is documented separately in [`timesplit/README.md`](../timesplit/README.md).
The short map:

| Stage | Files |
|---|---|
| Time split from the source databases | [`timesplit/build/chembl_timesplit.py`](../timesplit/build/chembl_timesplit.py), [`timesplit/build/bdb_timesplit.py`](../timesplit/build/bdb_timesplit.py) |
| Merge, dedup, assign L1–L4 | [`timesplit/build/build_t3.py`](../timesplit/build/build_t3.py), [`timesplit/build/cluster_t3.sh`](../timesplit/build/cluster_t3.sh) |
| Eval set: actives + cross-target decoys at 1:50 | [`timesplit/build/build_t3_eval.py`](../timesplit/build/build_t3_eval.py) |
| 3D conformers for the UniMol-family models | [`timesplit/build/gen_conformers.py`](../timesplit/build/gen_conformers.py), [`timesplit/build/resume_conformers.py`](../timesplit/build/resume_conformers.py) |
| Structures: PDB metadata, chain map, co-crystal choice, pocket extraction | [`timesplit/structure/`](../timesplit/structure/) |

Running the models:

| Model | Files |
|---|---|
| DrugCLIP / BindCLIP | [`timesplit/runners/run_t3_unimol.sh`](../timesplit/runners/run_t3_unimol.sh), input build [`timesplit/runners/build_t3_unimol.py`](../timesplit/runners/build_t3_unimol.py) |
| LigUnity / LiTENCLIP / HypSeek | [`timesplit/runners/run_t3_ligunity.sh`](../timesplit/runners/run_t3_ligunity.sh), [`run_t3_litenclip.sh`](../timesplit/runners/run_t3_litenclip.sh), [`run_t3_hypseek.sh`](../timesplit/runners/run_t3_hypseek.sh) |
| Task registration patches for those three repos | [`timesplit/runners/patch_ligunity_t3.py`](../timesplit/runners/patch_ligunity_t3.py), [`patch_t3_task.py`](../timesplit/runners/patch_t3_task.py) |
| ConGLUDe / ConPLex / SPRINT | [`timesplit/runners/run_t3_conglude.py`](../timesplit/runners/run_t3_conglude.py), [`run_t3_conplex.py`](../timesplit/runners/run_t3_conplex.py), [`run_t3_sprint.py`](../timesplit/runners/run_t3_sprint.py) |
| The hardcoded-`bsz=64` fix (see below) | [`timesplit/runners/fix_bsz.py`](../timesplit/runners/fix_bsz.py) |

Analysis — one script per claim in this document:

| Claim | File |
|---|---|
| Main table (9 models × 4 layers × 5 metrics) | [`timesplit/analysis/score_t3.py`](../timesplit/analysis/score_t3.py), [`collect_t3.py`](../timesplit/analysis/collect_t3.py) |
| Raw-output integrity check after a duplicate launch | [`timesplit/analysis/verify_t3_raw.py`](../timesplit/analysis/verify_t3_raw.py) |
| Target-class annotation and per-class table | [`timesplit/analysis/annotate_target_class3.py`](../timesplit/analysis/annotate_target_class3.py), [`report_by_class.py`](../timesplit/analysis/report_by_class.py) |
| Pocket-fit confound, stratified + negative control | [`timesplit/analysis/stratify_pocketfit.py`](../timesplit/analysis/stratify_pocketfit.py) |
| ConGLUDe training-set overlap, and whether it mattered | [`timesplit/analysis/check_conglude_leak.py`](../timesplit/analysis/check_conglude_leak.py), [`conglude_leak_effect.py`](../timesplit/analysis/conglude_leak_effect.py) |
| Structure-quality grading (A/B/C) and the high-quality subset | [`timesplit/analysis/t3_target_quality.py`](../timesplit/analysis/t3_target_quality.py), [`score_hq.py`](../timesplit/analysis/score_hq.py) |
| Export the CSVs in [`results/`](../results/) | [`timesplit/analysis/export_results.py`](../timesplit/analysis/export_results.py) |

**One bug worth reading the fix for.** `fix_bsz.py` repairs a `bsz = 64` that was
copied along with the surrounding code from the DEKOIS task, which made
`--batch-size` silently inert. It was caught because OOM allocation sizes stayed
byte-identical after lowering the batch size. Target failure rate went from 70%
to 2.4%.
