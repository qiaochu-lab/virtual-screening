# Data release

Everything needed to check this benchmark, in three tiers. Each tier answers a
different question, and you only need the one that matches what you want to do.

| Tier | Where | Size | What it lets you do |
|---|---|---|---|
| **1. Index** | this repository | 66 MB | see every molecule, label and affinity; re-derive the layering |
| **2. Raw scores** | companion data package | 380 MB | **recompute every metric we report**, under any cutoff or layering, without a GPU |
| **3. Structures** | companion data package | 3.1 GB | re-run the models, or run your own |

Tier 1 alone tells you what the benchmark *is*. Tier 1 + 2 lets you check the
canonical T1 and T3 numbers on a laptop. Tier 3 is needed only to run inference.
The companion delivery includes restore instructions and official checkpoint
download locations.

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

## Tier 2 — raw per-molecule scores (companion package)

One `.npz` per model per task, keys `"<task>/<layer>/<uniprot>/{preds,labels}"`,
scores `float32`, labels `int8`. Rebuild with
[`eval/pack_raw.py`](eval/pack_raw.py); verify against `manifest_T3.json` /
`manifest_T1.json`, which carry a sha256 per package.

**T3 — 13 packages.** Eleven canonical screening models, the HypSeek `_rk`
ranking/labelled-control package, and a second LigUnity-pocket package.
LigUnity-pocket was re-run on a newer
screening checkpoint on 2026-09-14, so its scores exist in two versions:
`T3_ligunity_pocket_ranking.npz` (current checkpoint — what every table on the
paper's 350-target convention is computed from) and
`T3_ligunity_pocket_ranking_ckpt1.npz` (first checkpoint — what the full
1,144-target auxiliary tables are still computed from, since those were not
re-run). ⚠️ **The 2026-09-13 archive shipped this wrong and was corrected on
2026-09-15**: the current-checkpoint filename carried the first checkpoint's
scores, with the archive's own manifest agreeing with the file, so a checksum
check there passed on the wrong data. The archive now holds both packages under
their right names. **A copy fetched before 2026-09-15 needs those two files
re-fetched.** See [`results/RAW_SCORES.md`](results/RAW_SCORES.md). The packages
include `drugclip`, `drugjepa`, `bindclip_randneg`, `bindclip_hardneg`,
`ligunity_pocket_ranking`, `ligunity_protein_ranking`, `litenclip`,
`hypseek_official_vs`, `hypseek_rk`, `conglude`, `conplex`, `sprint`.

✅ **T1 — 11 packages.** All eleven canonical models have per-molecule T1
scores across all three benchmarks (102 DUD-E, 80–81 DEKOIS, 13–15 LIT-PCBA).

⚠️ The per-target score *files* were completed on 2026-09-13, but the three
`.npz` **packages** for LigUnity-pocket, LigUnity-protein and LiTENCLIP were only
built on 2026-09-14 — the release directory held 7 T1 packages until then, and
this page said 10. LigUnity-pocket's package is from the new checkpoint, matching
`T1_main.csv`; the other two are unchanged scores, packaged for the first time.

`ligunity_pocket_ranking`, `ligunity_protein_ranking` and `litenclip` were the
three that used to be missing: their upstream repositories write only the
embedding and never the score, so nothing was lost in a run — there was never a
score on disk to keep. They are separate checkouts (`code/LigUnity`,
`code/LiTENCLIP`), and the patch that added score-saving for T3 covered the
first only, on its T3 branch. `timesplit/runners/patch_t1_save_preds.py` adds
the same save to `test_dude_target`, `test_dekois_target` and
`test_pcba_target` in both, and the three models were re-run across the three
benchmarks (~90 minutes on three GPUs).

**Nothing in any published table changed.** The re-run was validated against
`T1_main.csv` before the files were copied into place
(`timesplit/analysis/verify_t1_preds.py`): DUD-E and DEKOIS reproduce to within
5e-5 on all four metrics, LIT-PCBA to ~1e-3. That last gap is worth knowing if
you recompute from the files — see the caveat in
[`results/RAW_SCORES.md`](results/RAW_SCORES.md); it comes from fp16 inference
placing a few molecules differently at the top-5% boundary on the two largest
targets, and 13 of 15 LIT-PCBA targets still reproduce to the last digit.

So **T1 and T3 can both be independently recomputed for all canonical models.**

⚠️ Two corrections worth keeping visible. An earlier version of this page said
3 of 10 could be recomputed and named the wrong cause: that came from checking
only `results/t1_raw`, which holds three models, while four more sit directly
under `results/<model>/<benchmark>/` — the same failure `eval/pack_raw.py`
documents for T3, one root checked, several in use. The true count before the
re-run was 7 of 10.

## Tier 3 — pockets and ligands (companion package)

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

The quota is a parameter, not a target count. The pre-strict 350-quota subset
contains 328 records over 293 unique targets (L1 56 · L2 178 · L3 19 · L4 75).
The canonical release relabels and filters it to **L1 56 · L2 178 · L3 20 · L4
41**. Use `results/release-2026-10-06/T3_targets_strict.csv`; it carries the
final labels explicitly. Targets can occur in more than one original layer, so
compatibility filters must key on `(layer, uniprot)` and never on `uniprot`
alone.

| File | Records / targets | Status |
|---|---|---|
| `results/release-2026-10-06/T3_targets_strict.csv` | 295 / 263 | **canonical release** — final strict labels |
| `results/T3_vsds_matched.csv` | 328 / 293 | supporting — pre-strict 350 quota |
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
