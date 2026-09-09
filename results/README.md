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
| `T3_ligand_only.csv` | per-target **chemical-series oracle ceiling** — no protein, but conditioned on that target's known actives |
| `T3_normalized_by_ceiling.csv` | model EF1% divided by that ceiling |
| `T3_ligand_novelty.csv` | novelty tiers: pool composition and what models retrieve |
| `T3_novelty_tiered_ef.csv` | enrichment computed separately per novelty tier |
| `T3_target_mirroring.csv` | each T3 target's closest training-set homologue (mmseqs, cov ≥50%) |
| `T3_target_redundancy.csv` | all-vs-all identity within the subset (pairs ≥20%) |

## HypSeek official weights — added 2026-09-08

| File | What |
|---|---|
| `T1_hypseek_official.md` | write-up: `_vs` vs `_rk`, both `alpha_prot` settings, the retraction |
| `T1_hypseek_official.csv` | T1 across five HypSeek weight variants |
| `T3_hypseek_official.csv` | T3 four layers, same variants |

## Target swap — added 2026-09-09

| File | What |
|---|---|
| `T3_target_swap.csv` | 10 models × L1/L4 × 3 metrics, correct vs swapped target, paired p |

