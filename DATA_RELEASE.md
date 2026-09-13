# Data release

Everything needed to check this benchmark, in three tiers. Each tier answers a
different question, and you only need the one that matches what you want to do.

| Tier | Where | Size | What it lets you do |
|---|---|---|---|
| **1. Index** | this repository | 66 MB | see every molecule, label and affinity; re-derive the layering |
| **2. Raw scores** | Zenodo (DOI pending) | 279 MB | **recompute every metric we report**, under any cutoff or layering, without a GPU |
| **3. Structures** | Zenodo (DOI pending) | 3.1 GB | re-run the models, or run your own |

Tier 1 alone tells you what the benchmark *is*. Tier 1 + 2 lets you check every
number in this repository on a laptop. Tier 3 is needed only to run inference.

## Tier 1 — index (in this repository)

`results/frozen/`, produced by [`timesplit/analysis/freeze_t3.py`](timesplit/analysis/freeze_t3.py):

| File | Contents |
|---|---|
| `T3_molecules.csv.gz` | unique molecules: `mol_id`, `inchikey`, `smiles`, two novelty scores |
| `T3_index_L1..L4.csv.gz` | per target, per molecule: `layer`, `uniprot`, `jsonl_pos`, `lmdb_pos`, `mol_id`, `label`, `paff` |
| `T3_model_order.csv` | which molecule ordering each model used per target, and whether it passed strict validation |

The two position columns exist because a model reading an lmdb sees
lexicographic cursor order (0, 1, 10, 100, …) while the eval-set jsonl is in a
different order. Joining scores to labels by the wrong one silently mislabels
every molecule — that bug cost this project two retracted conclusions, so the
mapping is shipped rather than left to be re-derived.

## Tier 2 — raw per-molecule scores (Zenodo)

One `.npz` per model per task, keys `"<task>/<layer>/<uniprot>/{preds,labels}"`,
scores `float32`, labels `int8`. Rebuild with
[`eval/pack_raw.py`](eval/pack_raw.py); verify against `manifest_T3.json` /
`manifest_T1.json`, which carry a sha256 per package.

**T3 — 11 packages, 238.5 MB.** All ten published models, plus both HypSeek
weights: `drugclip`, `bindclip_randneg`, `bindclip_hardneg`,
`ligunity_pocket_ranking`, `ligunity_protein_ranking`, `litenclip`,
`hypseek_official_vs`, `hypseek_rk`, `conglude`, `conplex`, `sprint`.

⚠️ **T1 — 3 packages, 40.5 MB, not ten.** Only `conglude`, `conplex` and
`sprint` have per-molecule T1 scores. The other seven models' forks do not
persist scores on the DUD-E / DEKOIS / LIT-PCBA path — the patches in
[`PATCHES.md`](PATCHES.md) that add score-saving were written for the T3 branch
only. **So T1 numbers can be independently recomputed for 3 of 10 models, and
T3 numbers for all of them.** Fixing this needs the patches extended to the T1
branch and those seven models re-run; it is not a gap in what was kept.

## Tier 3 — pockets and ligands (Zenodo)

The 350-quota subset only, split by layer. 6 Å pockets and the ligand lmdbs.

| Archive | Targets | Size |
|---|---|---|
| `T3_6A_L1.tar.gz` | 56 | 273 MB |
| `T3_6A_L2.tar.gz` | 171 | 2,067 MB |
| `T3_6A_L3.tar.gz` | 19 | 125 MB |
| `T3_6A_L4.tar.gz` | 69 | 626 MB |

**315 of the subset's 328 records.** The 13 without a directory are listed in
`manifest_data.json` under `missing`, so "absent from the deposit" is
distinguishable from "absent from the subset". They are the known
pocket-extraction gap (95.4% coverage), not a packaging loss:

```
L2/P09619 L2/P35916 L2/P47869 L2/P62813 L2/Q13489 L2/Q16584 L2/Q6ZQW0
L4/P09246 L4/Q9BQA1 L4/Q9UBM7 L4/Q9UHH9 L4/Q9UKP4 L4/Q9ULC5
```

Rebuild with [`eval/pack_dataset_subset.sh`](eval/pack_dataset_subset.sh).

## ⚠️ Which subset file is which

The quota is a parameter, not a target count. **350 quota yields 328 records
over 293 unique targets** (L1 56 · L2 178 · L3 19 · L4 75) — 35 targets appear
in more than one layer, which is why any subset filter must key on
`(layer, uniprot)` and never on `uniprot` alone.

| File | Records / targets | Status |
|---|---|---|
| `results/T3_vsds_matched.csv` | 328 / 293 | **current** — the 350 quota |
| `results/T3_vsds_matched_q250.csv` | 242 / 222 | superseded — the earlier 250 quota |

The unsuffixed file is the newer one. This reads backwards and has caused
mistakes; check the row count before using either.

## Recomputing a number

```python
import numpy as np
from eval.metrics import enrichment_factor

z = np.load("T3_hypseek_official_vs.npz")
targets = {k.rsplit("/", 1)[0] for k in z.files}
ef = [enrichment_factor(z[f"{t}/preds"], z[f"{t}/labels"], 0.01) for t in targets]
print(np.mean(ef))
```

To re-layer instead — using your own definition of which targets a model has
seen — walk `results/frozen/T3_index_*.csv.gz`, keep the position column that
`T3_model_order.csv` says that model used, and join to the scores above. A
worked example is in [`results/FROZEN_TABLES.md`](results/FROZEN_TABLES.md).

## Provenance

Molecules and affinities come from public sources (ChEMBL, BindingDB, PDBbind);
structures from the PDB. The layering, the decoy assignment and the pocket
extraction are this project's. See [`tasks/T3-time-split.md`](tasks/T3-time-split.md)
for how the set was built and [`LIMITATIONS.md`](LIMITATIONS.md) for what it
does not support.
