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
([`t1/build_train_union.py`](t1/build_train_union.py))

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

Using `_rk` is also what makes one T2 result interpretable — a checkpoint chosen
*specifically for ranking* still gives Spearman +0.028 on T3, which rules out
"they just didn't optimise for ranking" as the explanation.

**LigUnity publishes more variants than were used.** `_vs` (evaluated), plus
`_0.3` and `_0.8`, which filter the training set by sequence distance to the test
proteins. Only the plain `_vs` was run. The release notes state that DUD-E /
DEKOIS / LIT-PCBA test proteins were removed from training, which is why the
similarity ablation measures distance to the *nearest remaining* training protein
rather than to the target itself ([`t1/t1_sim3.py`](t1/t1_sim3.py)).

**BindCLIP's two weights differ in negative sampling** (random vs hard
negatives), not in architecture or data — a controlled pair, useful for reading
the effect of the training signal alone.

**ConPLex is the negative control.** Sequence-only, no structure, no pocket. Where
an effect appears in structure models and not in ConPLex, structure is implicated;
where it appears in both, it is not about structure. This is how the pocket-fit
confound was isolated.

**SPRINT is not a sequence-only model**, despite reading like one. It consumes
SaProt structure-aware sequences, so it needs the same structures as the pocket
models ([`t3/structure/gen_saprot_seqs.py`](t3/structure/gen_saprot_seqs.py)) and
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

## Status by task

| Model | T1 | T2 | T3 | T5 |
|---|---|---|---|---|
| DrugCLIP | ✅ | ✅ | ✅ | ✅ (4/6/8 Å) |
| BindCLIP ×2 | ✅ | ✅ | ✅ | ✅ (4/6/8 Å) |
| LigUnity ×2 | — | ✅ (T3 + FEP) | ✅ | — |
| LiTENCLIP | — | ✅ (T3 + FEP) | ✅ | — |
| HypSeek `_rk` | — | ✅ | ✅ | — |
| ConGLUDe | — | ✅ | ✅ | — |
| ConPLex | — | ✅ | ✅ | control |
| SPRINT | — | — | ⚠️ L3/L4 only | — |
| Boltz-2 | — | 🔄 running | structures | — |
