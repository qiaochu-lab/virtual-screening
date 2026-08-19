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
2. **Those benchmarks participated in model selection.** A model author told us
   directly: DUD-E and LIT-PCBA are hard to optimize together, so the released
   checkpoint was "basically a compromise choice". Scoring that checkpoint on
   the same benchmarks partly measures the authors' trade-off.
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

**Structures:** 1,466 new targets; experimental PDB where available, Boltz-2
predictions otherwise; usable pocket coverage **95.4%**. Pockets are
residue-level at 6 Å, validated against the authors' own published pockets with
**100% coordinate overlap** on five DUD-E targets.

Full construction pipeline and the four things that are easy to get wrong:
[`t3/README.md`](../t3/README.md).

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

**3. Model ranking reverses by target class.** On kinases ConPLex (pure
sequence) beats ConGLUDe (geometric) 5.76 vs 2.07; on other enzymes it reverses,
4.73 vs 1.28 — both significant after BH-FDR correction. Reporting only the
overall mean would say "ConGLUDe is better", which is wrong per class.

**4. AUROC hides what EF shows.** On L4, ConGLUDe vs ConPLex is AUROC 0.570 vs
0.549 (p=0.37, n.s.) but EF1% 4.00 vs 2.16 (p=0.018, significant). Virtual
screening only cares about the top of the list. Report both.

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
