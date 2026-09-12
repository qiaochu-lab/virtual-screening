# Models and checkpoints

Nine retrieval models plus one co-folding model, all run with **official code and
official weights**. This page records exactly which weight was used, where it
came from, and what to know about each before reading its numbers.

| Model | Input on the protein side | Ligand side | Training data | Weight used |
|---|---|---|---|---|
| DrugCLIP | 3D pocket (UniMol) | 3D conformer | DrugCLIP set (`train_no_test_af`) | `checkpoint_best.pt` |
| BindCLIP-randneg | 3D pocket | 3D conformer | same as DrugCLIP | `BindCLIP_randneg.pt` |
| BindCLIP-hardneg | 3D pocket | 3D conformer | same as DrugCLIP | `BindCLIP_hardneg.pt` |
| LigUnity-pocket | 3D pocket | 3D conformer | PocketAffDB | `LigUnity_VS/pocket_ranking_vs/checkpoint_avg_41-50.pt` |
| LigUnity-protein | sequence | 3D conformer | PocketAffDB | `LigUnity_VS/protein_ranking_vs/checkpoint_avg_41-50.pt` |
| LiTENCLIP | 3D pocket | 3D conformer | PocketAffDB (same files) | `checkpoint.best_valid_bedroc_0.50.pt` |
| HypSeek | 3D pocket, hyperbolic embedding space | 3D conformer | PocketAffDB | `checkpoint_avg_41-50_rk.pt` |
| ConGLUDe | sequence + structure graph (`.pdb`) | graph | own (undisclosed list) | shipped with the repo |
| ConPLex | sequence (protein LM) | fingerprint | BindingDB | `BindingDB_ExperimentalValidModel.pt` |
| SPRINT | **SaProt structure-aware sequence** (AA + foldseek 3Di) | SMILES | own (`MERGED`) | `sprint.ckpt` |
| Boltz-2 | co-folded complex | co-folded complex | own | official release |

The UniMol-family models additionally load the shared pretrained encoders
`mol_pre_no_h_220816.pt` and `pocket_pre_220816.pt`.

## One training set across seven models, and half of them also get affinity labels

The seven pocket-family models are **not** trained on rival corpora. LigUnity's
training task reads two label files (`unimol/tasks/train_task.py:523-524`):

| Half | Size | Trained on by |
|---|---|---|
| `train_label_pdbbind_seq.json` — structures only | **16,744 PDB entries** → 3,468 UniProt | all seven |
| `train_label_blend_seq_full.json` — with pAff | **2,196 UniProt**, 26,748 assays | LigUnity ×2, LiTENCLIP, HypSeek |

**The structure half is the same data as DrugCLIP's `train_no_test_af`**:
16,744 PDB entries on each side, intersection 16,744, neither exclusive.
So DrugCLIP and the BindCLIP pair see the structure half; the other four see
that *plus* affinity labels for 2,196 targets. Union at UniProt level: 4,847.
([`standard/build_train_union.py`](standard/build_train_union.py))

⚠️ This page previously described these as two separate sets of 4,098 and 2,196
UniProt overlapping in 881. That counted only the affinity half for the second
group. The correction, and what it does to the findings that rested on it, is in
[`LIMITATIONS.md` §24](LIMITATIONS.md).

LiTENCLIP's `test_datasets/` are symlinks into LigUnity's, so the two share the
training files byte-for-byte — which is why T3's cutoff date is valid for both
without adjustment.

**This is the basis of a finding**, not just bookkeeping: the four models that
get affinity labels land at L1 EF1% 32–39, and the three with structures only
land at 17–19, across substantial architectural differences. Because the
structures are identical between the groups, the comparison isolates **what the
labels add**, not how much data each had.

## What the other three models trained on

The seven pocket-family models publish their training sets as target lists. The
three sequence-family models do not, which is why they sat outside the audit for
so long. What is actually obtainable, and how:

| Model | Published as | What we can reconstruct |
|---|---|---|
| **ConPLex** | training **sequences** in the repo, no accessions | `dataset/BindingDB/train.csv` → **1,026 unique sequences**; DAVIS 372; BIOSNAP carries a `Gene` column with **2,177** UniProt accessions outright. Reverse-lookup by mmseqs gives its coverage of any evaluation set. |
| **SPRINT** | `merged_data.zip` (371 MB) on the MERGED release | **336 train UniProt accessions** and 1.39M ligand ids, read from `merged_pos_uniq_train_rand.tsv`. MERGED = BindingDB 2022-04-14 + ChEMBL 30 + PubChem 2022-09-05. |
| **ConGLUDe** | Zenodo `LB_train_val.zip` | Not retrieved — Zenodo returned 504 during the check. Approximable from MERGED train ∪ test (5,416 accessions); the authors state they drop proteins >90% identical to any test set. |

**The checkpoint matters more than the paper here.** We run ConPLex's
`BindingDB_ExperimentalValidModel.pt`, so BindingDB is its training set — but
`contrastive: True` is the shipped default, which means it *also* trained on
DUD-E decoys. That second half is the one that invalidated a number we had been
reporting; see [`tasks/T1-enrichment.md`](tasks/T1-enrichment.md#conplex-trained-on-dud-e).

SPRINT's 336 accessions are worth noting for their size — against the pocket
family's 4,847-UniProt union, it is trained on an order of magnitude fewer
proteins. That is a plausible part of why it is last on
two of three standard benchmarks, and it is not something its paper foregrounds.
(42 further accessions are excluded as LIT-PCBA targets, leaving 294 for any
comparison against that benchmark.)

Lists are kept off this repository along with the rest of the data; the
reconstruction scripts are
[`timesplit/analysis/conplex_train_coverage.py`](timesplit/analysis/conplex_train_coverage.py)
and [`timesplit/analysis/per_model_audit.py`](timesplit/analysis/per_model_audit.py).
Training data separates the tiers better than architecture does.

## Checkpoint choices worth knowing

**HypSeek ships two weights from one training run** — `_vs` selected on CASF
BEDROC (screening) and `_rk` selected on FEP (ranking). This is itself evidence
for T2's premise: the authors found one weight could not do both jobs well.

Only `_rk` was public when this benchmark was built, so that is what every
HypSeek number here was measured with. The author released `_vs` on 2026-09-07
in [issue #4](https://github.com/jianhuiwemi/HypSeek/issues/4); both are now
measured side by side in
[`results/T1_hypseek_official.md`](results/T1_hypseek_official.md). The released
`_rk` is byte-identical to the copy used throughout
(md5 `02d7574254bc…`), and `_rk` outscores `_vs` on every screening
measurement — so reporting `_rk` did not inflate HypSeek's numbers relative to
its own screening weight, though it does mean **the screening tables report a
ranking-selected checkpoint**.

Using `_rk` also turns out to matter: it is **the best ranker in the benchmark**
(T3 Spearman +0.260 at L1, ahead of LigUnity-protein's +0.230 and DrugCLIP's
+0.091) *and* the best screener on all three standard benchmarks. An earlier
version of this file said the opposite — that the ranking weight gave only
+0.028 — which was a bug in our analysis code, not a property of the checkpoint
([`PATCHES.md`](PATCHES.md)).

**LigUnity publishes more variants than were used.** `_vs` (evaluated), plus
`_0.3` and `_0.8`, which filter the training set by sequence distance to the test
proteins. Only the plain `_vs` was run. The release notes state that DUD-E /
DEKOIS / LIT-PCBA test proteins were removed from training, which is why the
similarity ablation measures distance to the *nearest remaining* training protein
rather than to the target itself ([`standard/t1_sim3.py`](standard/t1_sim3.py)).

**BindCLIP's two weights differ in negative sampling** (random vs hard
negatives), not in architecture or data — a controlled pair, useful for reading
the effect of the training signal alone.

**ConPLex is a negative control — on T3, not on T1, and it is not the only one.**
Sequence-only, no structure, no pocket. Where an effect appears in structure
models and not in ConPLex, structure is implicated; where it appears in both, it
is not about structure. That analysis runs on T3, where ConPLex's training set
covers 19.4% of targets — the same band as the other two sets.

⚠️ **One control is not enough, and the pocket-fit analysis is the case that
showed why.** It was run for a year on ConPLex alone against a single "structure"
arm, and read as "the effect appears only in structure models". Run against all
three non-pocket models, ConPLex and SPRINT stay clean at both layers but
**LigUnity-protein — also sequence-only — shows the effect at L2 (p = 0.0002)**.
One clean control does not establish specificity; it takes every control you
have. → [`LIMITATIONS.md` §5](LIMITATIONS.md)

⚠️ **The same reasoning is invalid on DUD-E.** ConPLex's contrastive objective is
trained on DUD-E decoys from 40 of our 102 evaluation targets, and 74% of its
DUD-E enrichment disappears when those are excluded (EF1% 18.70 → 4.83). Any
control argument made from its DUD-E column measures its training set.
→ [`tasks/T1-enrichment.md`](tasks/T1-enrichment.md#conplex-trained-on-dud-e)

**SPRINT is not a sequence-only model**, despite reading like one. It consumes
SaProt structure-aware sequences, so it needs the same structures as the pocket
models ([`timesplit/structure/gen_saprot_seqs.py`](timesplit/structure/gen_saprot_seqs.py)) and
belongs on the structure side of any comparison.

## Interface quirks that cost time

| Model | Quirk |
|---|---|
| ConGLUDe | the score matrix is **ligands × proteins**, opposite to what its README says |
| ConGLUDe | reads `.pdb` only, not mmCIF — large entries without a legacy PDB file fall back to predicted structures |
| ConPLex | output column order is **reversed** relative to the input TSV |
| SPRINT | its TSV reader sniffs the delimiter and splits the header `Target Sequence` on the space |
| LigUnity/HypSeek/LiTENCLIP | the FEP branch stores embeddings only, no scores |

Full detail, and what breaks without each fix, in [`PATCHES.md`](PATCHES.md).

## Status by task, and which checkpoint each ran

Every model was evaluated with **one** checkpoint across all tasks — we did not
swap weights per task. The selection criterion behind that checkpoint differs
between models, and that asymmetry is worth seeing in one place.

| Model | Checkpoint selected on | T1 | T2 | T3 | T5 | T6 |
|---|---|---|---|---|---|---|
| DrugCLIP | screening | ✅ | ✅ | ✅ | ✅ (4/6/8 Å, apo) | — |
| BindCLIP ×2 | screening | ✅ | ✅ | ✅ | ✅ (4/6/8 Å, apo) | — |
| LigUnity-pocket | screening (`_vs`) | ✅ | ✅ (T3 + FEP + CASF) | ✅ | — | shortlist source |
| LigUnity-protein | screening (`_vs`) | ✅ | ✅ (T3 + FEP + CASF) | ✅ | — | shortlist source |
| LiTENCLIP | screening (`best_valid_bedroc`) | ✅ | ✅ (T3 + FEP + CASF) | ✅ | — | — |
| **HypSeek `_vs`** (official) | **screening** | ✅ | — | ✅ main table | — | — |
| **HypSeek `_rk`** (official) | **FEP ranking** | — | ✅ | ✅ derived analyses | ✅ (256 vs 511 cap) | — |
| HypSeek `_vs` (collaborator) | screening | ✅ | ✅ | ✅ | — | — |
| HypSeek `_vs` (ours, deficient) | screening | ✅ | — | ✅ | — | — |
| ConGLUDe | undisclosed | ✅ | ✅ | ✅ | — | — |
| ConPLex | undisclosed | ✅ | ✅ | ✅ | control | — |
| SPRINT | undisclosed | ✅ | — | ✅ | — | — |
| Boltz-2 | n/a (co-folding) | — | ✅ (FEP) | structures | — | ✅ |

**The asymmetry this used to carry is now closed.** For most of this
project `_rk` was the only HypSeek weight its authors had released, so HypSeek
was the one retrieval model represented by a ranking-selected checkpoint while
every other model ran a screening-selected one. That was raised as a criticism,
and the specific worry was selective leakage: `_rk` is chosen on benchmarks, so
scoring DUD-E and LIT-PCBA with it scores a benchmark using a checkpoint tuned
on benchmarks of that kind.

Since the author released `_vs`, **the screening tables (T1, T3 main) use `_vs`
and the ranking tables (T2) use `_rk`** — measured separately, which is what
was asked for. Note that `_rk` is the *stronger* screening weight (T3 L1 EF1%
36.63 vs 32.11), and that this is not a defence: a benchmark-selected weight
outscoring the same benchmarks is the pattern the objection predicts, so the
gap supports the concern rather than refuting it.

Two earlier readings of this are superseded. Our own `_vs` training scored far
below `_rk`, which we first read as evidence about the objective; it was a
deficit in our training (see [`MODELS_TRAINING.md`](MODELS_TRAINING.md)). A
collaborator's paper-faithful `_vs` also trails `_rk` at every T3 layer — EF1%
30.70 vs 36.63 at L1, 5.75 vs 7.34 at L4
([`results/T3_hypseek_three_way.csv`](results/T3_hypseek_three_way.csv)) — so
the ordering is real; what changed is that "`_rk` scores higher" is no longer
treated as a reason to report `_rk` on screening tasks.

⚠️ Analyses built before the switch — the target-swap rounds, the leakage
audit, the bootstrap CIs, the per-class breakdown and all of T5 — were computed
with `_rk` and have not been re-run. They are labelled at each appearance. Their
conclusions are cross-model patterns (ten of ten, twelve of twelve) that one
model's checkpoint choice does not turn on.

**A caveat that applies to anyone using this model.** The released `_rk` scores
*above* the paper's own published screening numbers — DUD-E EF1% 56.39 against
51.44 — measured with a pipeline that reproduces the paper's LigUnity baseline
to four decimals. The released checkpoint is not the model behind the paper's
Table 1, and the paper never mentions two checkpoints at all.

Where a model's own paper reports no metric for a task, that is a property of
the model's scope rather than of this benchmark, and the cell above says so
rather than implying the run failed.

