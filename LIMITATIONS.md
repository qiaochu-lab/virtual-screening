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

"Target seen / unseen" is judged against **LigUnity's** training set (2,196
UniProt). DrugCLIP and BindCLIP were trained on a different set (16,744 PDB
pockets → 4,098 UniProt; overlap with LigUnity's only 881)
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

## 3. Three models' training sets are unavailable

ConGLUDe, ConPLex and SPRINT do not publish target lists in a usable form, so
their layer labels are inherited from LigUnity's split and are approximate.

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

Measured, not assumed ([`timesplit/analysis/stratify_pocketfit.py`](timesplit/analysis/stratify_pocketfit.py)):
the effect is real (L2, p = 0.0008) and appears **only** in structure models —
the sequence-only negative control (ConPLex) shows nothing. Correcting for it
moves the decay from −72% to −67%. The conclusion stands; the magnitude shifts.

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

## 8. Structure-source comparison is not randomised

Whether a target has an experimental structure is itself non-random —
well-studied targets have them. The comparison (p = 0.28–0.74, no significant
difference) is therefore observational. The differences are small and
consistently signed, which supports the reading, but it is not proof that
predicted structures are equivalent.

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

More generally, released checkpoints are chosen with DUD-E / LIT-PCBA scores in
view, so those benchmarks measure a selection decision as well as a model. This
is the motivation for T3, and it does not go away inside T3 — it only stops
applying to the *targets*, not to the weights.

## 13. Coverage gaps

- **T1: 7 of 9 models run.** ConGLUDe, ConPLex and SPRINT need their inputs
  (sequences, structures, 3Di tokens) prepared for these benchmarks and are not
  queued. CASF-2016 exists for LigUnity ×2 only — the two forks' CASF code path
  calls their own model with the wrong signature.
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

## 15. One analysis bug reached the README before it was caught

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

## 16. Statistical practice

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
