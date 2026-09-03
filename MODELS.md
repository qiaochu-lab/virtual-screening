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

## Only two training sets across seven models

| Set | Size | Used by |
|---|---|---|
| `train_no_test_af` | 16,744 PDB pockets → **4,098 UniProt** | DrugCLIP, BindCLIP ×2 |
| PocketAffDB | **2,196 UniProt** | LigUnity ×2, LiTENCLIP, HypSeek |

Overlap between them: **881 UniProt**. Union: 5,413.
([`standard/build_train_union.py`](standard/build_train_union.py))

LiTENCLIP's `test_datasets/` are symlinks into LigUnity's, so the two share the
training files byte-for-byte — which is why T3's cutoff date is valid for both
without adjustment.

**This is the basis of a finding**, not just bookkeeping: the three models
trained on PocketAffDB all land at L1 EF1% 32–39, and the three trained on the
DrugCLIP set all land at 17–19, across substantial architectural differences.
Training data separates the tiers better than architecture does.

## Checkpoint choices worth knowing

**HypSeek ships two weights from one training run** — `_vs` selected on CASF
BEDROC (screening) and `_rk` selected on FEP (ranking). Only `_rk` is public, so
that is what was evaluated. This is itself evidence for T2's premise: the authors
found one weight could not do both jobs well.

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

**ConPLex is the negative control.** Sequence-only, no structure, no pocket. Where
an effect appears in structure models and not in ConPLex, structure is implicated;
where it appears in both, it is not about structure. This is how the pocket-fit
confound was isolated.

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
| **HypSeek `_rk`** | **FEP ranking** | ✅ | ✅ | ✅ | ✅ (256 vs 511 cap) | — |
| HypSeek `_vs` (ours) | screening | ✅ | — | ✅ | — | — |
| ConGLUDe | undisclosed | ✅ | ✅ | ✅ | — | — |
| ConPLex | undisclosed | ✅ | ✅ | ✅ | control | — |
| SPRINT | undisclosed | ✅ | — | ✅ | — | — |
| Boltz-2 | n/a (co-folding) | — | ✅ (FEP) | structures | — | ✅ |

**The asymmetry to keep in mind: every retrieval model here ran a
screening-selected checkpoint except HypSeek, which ran a ranking-selected one**
— because `_rk` is the only weight its authors released. Reporting HypSeek's
screening numbers from a checkpoint chosen on FEP ranking is not obviously fair,
and it was raised as a criticism.

It turns out not to disadvantage HypSeek. We trained the screening-selected
weight ourselves from the published recipe, two seeds, and it is **worse** at
screening: T3 L1 EF1% 22.2 against `_rk`'s 36.6, and 20–24% lower on the
standard benchmarks ([`MODELS_TRAINING.md`](MODELS_TRAINING.md),
[`results/T1_T3_hypseek_seeds.csv`](results/T1_T3_hypseek_seeds.csv)). So `_rk`
is HypSeek's better screening weight as well as its ranking weight — which is
itself the finding, not a flaw in the setup.

Where a model's own paper reports no metric for a task, that is a property of
the model's scope rather than of this benchmark, and the cell above says so
rather than implying the run failed.

