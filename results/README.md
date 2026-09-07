# Result tables

## Dataset v2 (≥50 actives, VSDS-vd-matched) — added 2026-09-06

| File | What |
|---|---|
| `T3_vsds_matched.csv` | the selected subset: 242 target×layer entries, 222 unique targets |
| `T3_main_vsds_subset.csv` | T3 main table on the subset, both original and corrected layering |
| `T2_on_T3_subset.csv` | affinity ranking on the subset |
| `T5_structure_source_subset.csv` | structure-source control on the subset |
| `T5_pocket_threshold_subset.txt` | 4/6/8 Å curve on the subset |
| `T5_apo_subset.txt` | apo control on the subset (15 targets — under-powered) |
| `T6_recall_subset.txt` / `T6_recall_full.txt` | shortlist recall ceiling, subset vs full |

## Leakage audit — added 2026-09-06

| File | What |
|---|---|
| `T3_ligand_only.csv` | per-target ligand-only baseline (no protein input) |
| `T3_normalized_by_ceiling.csv` | model EF1% divided by that ceiling |
| `T3_ligand_novelty.csv` | novelty tiers: pool composition and what models retrieve |
| `T3_target_mirroring.csv` | each T3 target's closest training-set homologue (mmseqs, cov ≥50%) |
| `T3_target_redundancy.csv` | all-vs-all identity within the subset (pairs ≥20%) |
