# T3 — Time-Split Generalization Benchmark

A benchmark for testing whether pocket–ligand retrieval models generalize to
targets that appeared **after** their training data was assembled.

## Why this exists

Published enrichment numbers for these models are measured on DUD-E, LIT-PCBA
and DEKOIS. Three problems with using those to judge generalization:

1. **DUD-E's decoys are synthetic** — property-matched, topologically dissimilar
   molecules. A 2019 analysis showed this leaves a shortcut: a model can score
   well by looking at the ligand alone, ignoring the protein.
2. **The benchmarks participated in model selection.** One model author confirmed
   to us that DUD-E and LIT-PCBA are hard to optimize simultaneously, so the
   released checkpoint was "basically a compromise choice". Scoring that
   checkpoint on the same benchmarks measures the authors' trade-off as much as
   the model.
3. **No temporal holdout.** Targets and ligands in existing benchmarks may all
   have been visible during training.

T3 addresses all three: data that postdates the training cutoff cannot have been
seen, and cannot have participated in checkpoint selection.

## Design

**Cutoff: 2024-12.** Chosen because the training libraries of the models under
test are built from ChEMBL 34 + BindingDB 2024m5. Anything entering the databases
after that date is future data for every model evaluated.

**Sources:** ChEMBL 37 and BindingDB 202608, filtered to records deposited after
the cutoff, then set-differenced against the training sets and deduplicated by
InChIKey.

**Four difficulty layers** — the point is the *gradient*, not any single number:

| Layer | Target | Ligand | Eval targets |
|---|---|---|---|
| L1 | seen in training | new | 349 |
| L2 | seen in training | new Bemis–Murcko scaffold | 488 |
| L3 | **unseen** | — but same protein family seen | 53 |
| L4 | **unseen**, **family unseen** | — | 254 |

L1 is the control. If a model performs near chance on L1, the evaluation
pipeline is broken, not the model. Family membership uses CD-HIT clustering at
40% sequence identity.

**Actives:** measured pAffinity ≥ 6 (1 µM), deduplicated by InChIKey. Targets
with fewer than 10 actives are excluded — enrichment metrics are too noisy below
that.

**Decoys — cross-target, not property-matched.** For each target, decoys are
real drug-like molecules drawn from the T3 pool that are active on *dissimilar*
targets, at 1:50 (matching DUD-E's ratio). Three exclusions: the target's own
actives, molecules active on any target in the same mmseqs 40% cluster, and
molecules sharing a Bemis–Murcko scaffold with any of the target's actives.

We deliberately do **not** use DUD-E-style property matching, since that is the
bias being questioned. The cost is that absolute numbers are not comparable to
published values — **only the L1→L4 decay within a fixed setup is meaningful.**

**Structures:** 1,466 new targets. Experimental PDB structures where available,
Boltz-2 predictions otherwise. Usable pocket coverage 95.4%.

**Pockets:** residue-level at 6 Å — any residue with an atom within 6 Å of any
ligand atom enters the pocket, whole residue. This replicates DrugCLIP's
`get_different_raid()`. Validated: re-extracting five DUD-E targets reproduces
the authors' published pockets with **100% coordinate overlap**.

## Pipeline

```
build/
  chembl_timesplit.py     ChEMBL 37  → post-cutoff records
  bdb_timesplit.py        BindingDB  → post-cutoff records
  build_t3.py             merge, dedup, assign L1–L4
  cluster_t3.sh           mmseqs 40% identity clustering
  build_t3_eval.py        eval set: actives + cross-target decoys at 1:50
  gen_conformers.py       3D conformers (ETKDG + MMFF) for UniMol-family models
  resume_conformers.py    resume with per-molecule timeout

structure/
  fetch_pdb_meta.py       UniProt → PDB IDs → non-polymer ligands (RCSB GraphQL)
  fetch_chain_map.py      PDB entry → {UniProt: chains}
  fetch_sequences.py      UniProt sequences
  rank_crystal_ligands.py rank candidate co-crystal ligands per target
  extract_pocket.py       pockets from Boltz-2 predicted complexes
  extract_pocket_pdb.py   pockets from experimental structures (chain-aware)
  truncate_domains2.py    domain truncation for sequences over Boltz-2's limit
  validate_truncation.py  independent check that truncation keeps binding sites
  prep_boltz*.py          Boltz-2 input preparation
  gen_saprot_seqs.py      SaProt structure-aware sequences (foldseek 3Di)

runners/                  per-model adapters (official code + official weights)
analysis/                 metrics, stratified controls, robustness checks
```

## Four implementation details that are easy to get wrong

**1. Chain assignment in multi-protein complexes.** 47.1% of PDB entries contain
more than one UniProt (ribosomes, proteasomes, respiratory chain). Using every
chain in the file means a ligand bound to subunit A becomes "the pocket" of
subunit B. Protein atoms must be restricted to the target's own chains; if the
ligand does not contact them, that (PDB, ligand) pair is invalid for that target
and the next candidate must be tried. Fixing this moved 153 targets out of the
usable set — every one of which previously had a bogus pocket.

**2. Choosing the co-crystal ligand.** Ranking candidates by raw Tanimoto to the
target's test ligands lets a 138 Da fragment win on a meaningless 0.02
similarity margin, producing a pocket that covers a fraction of the real site.
Ranking by size instead selects cardiolipin (1464 Da) and other membrane lipids,
which mark the transmembrane face rather than a drug site. The working rule
buckets similarity into 0.1 bins, prefers a drug-like MW window [250, 700]
within a bucket, and blocks ions, buffers, lanthanide phasing atoms, lipids,
detergents, glycans. Nucleotide cofactors (ATP/GTP/SAH/SAM/NAD) are kept — for
kinases and methyltransferases those *are* the drug sites.

**3. Domain truncation must follow binding sites, not construct length.** For
multi-domain proteins exceeding the structure predictor's length limit,
selecting the widest PDB construct is wrong: different entries cover different
domains, and the widest has no relation to where ligands bind. Validation
against UniProt binding/active-site annotations showed 33% of truncations
contained **no** annotated site. Ranking candidate regions by site coverage
first drops that to 0%.

**4. Enrichment cutoffs use `ceil`, not `round`.** RDKit's `CalcEnrichment` uses
`math.ceil(numMol * fraction)`. Synthetic tests with dataset sizes of
300/500/1000/2000 pass under either convention because `n × fraction` is an
integer at every fraction tested; the discrepancy only appears on real data. See
`eval/README.md`.

## Reproducing the dataset

Raw data is not included — ChEMBL and BindingDB have their own licenses and the
processed set is tens of GB. To rebuild:

1. Download ChEMBL 37 and BindingDB 202608
2. Obtain the training-set target lists of the models being evaluated
3. Run `build/` in order, then `structure/`, then `runners/`

Every filtering rule and parameter is in the scripts; the docstrings state what
each decision is for and what breaks without it.

## A note on paths

The scripts contain a hardcoded working directory (`B = "/data/yicheng/xqc/..."`)
from the machine they were run on. They are published as a record of exactly what
was executed rather than as a turnkey package — change `B` at the top of each
script to run them elsewhere. Docstrings state the intent of each step, so the
logic transfers even where the paths do not.
