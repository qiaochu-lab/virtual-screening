# Result tables

The release boundary is [`CANONICAL_TABLES.md`](CANONICAL_TABLES.md). Cite the
files under `release-2026-10-06/`; the remaining files in this directory are
supporting analyses and research intermediates with broader model panels.

The canonical release includes DrugJEPA, excludes LiTENCLIP v2 / HypLiTEN, and
uses the strict L1–L4 target convention (56 / 178 / 20 / 41 before
model-specific missing targets).

## Dataset v2 (≥50 actives, VSDS-vd-matched) — added 2026-09-06

| File | What |
|---|---|
| `T3_vsds_matched.csv` | the selected subset: **328** target×layer entries, **293** unique targets (quota 350) |
| `T3_main_vsds_subset.csv` | T3 main table on the subset, both original and corrected layering |
| `T2_on_T3_subset.csv` | affinity ranking on the subset |
| `T2_ligand_only_subset.csv` | E2 — the ranking oracle and the target-blind regressor, on the subset |
| `T2_ligand_only_subset_per_target.csv` | the same, per target |
| `T2_paired_vs_ligand_only_subset.csv` | each model vs the target-blind regressor, paired per target |
| `T2_vs_mw_baseline_subset.txt` | each model vs molecular weight, paired per target, on the subset |
| `T3_main_ci_subset.csv` | bootstrap confidence intervals on the subset |
| `T2_ligand_only.csv` | E2 on the full 1,144-target set — auxiliary check, not the paper's convention |
| `T2_ligand_only_per_target.csv` | the same, per target |
| `T5_structure_source_subset.csv` | structure-source control on the subset |
| `T5_pocket_threshold_subset.txt` | 4/6/8 Å curve on the subset |
| `T5_apo_subset.txt` | apo control on the subset (15 targets — under-powered) |
| `T6_recall_subset.txt` / `T6_recall_full.txt` | shortlist recall ceiling, subset vs full |

## Leakage audit — added 2026-09-06

| File | What |
|---|---|
| `T3_ligand_only.csv` | per-target **chemical-series oracle ceiling** — no protein, but conditioned on that target's known actives |
| `T2_novelty_tiers_subset.csv` | T2 by ligand-novelty tier, on the subset |
| `T2_novelty_paired_subset.csv` | familiar vs novel half, paired per target, on the subset |
| `T2_novelty_tiers_Agroup_subset.csv` | the same against the DrugCLIP family's own training ligands |
| `T2_novelty_paired_Agroup_subset.csv` | and its paired test |
| `T3_novelty_tiered_ef_subset.csv` | enrichment by novelty tier, on the subset |
| `T3_ligand_only_noscaf.csv` | the same oracle with the decoy scaffold rule switched off — #3b |
| `T3_main_clean_subset.csv` | contamination-removed main table, on the subset |
| `T3_seq_vs_pocket_per_target_subset.csv` | LigUnity's two branches, paired per target, on the subset |
| `T3_ligand_only_subset.csv` | the chemical-series oracle ceiling on the 350-quota subset, per layer |
| `T3_normalized_by_ceiling.csv` | model EF1% divided by that ceiling |
| `T3_ligand_novelty_subset.csv` | novelty tiers on the 350-quota subset: pool composition and what models retrieve |
| `T3_ligand_novelty.csv` | the same on the full 1,144-target set — auxiliary check |
| `T3_exact_overlap_subset.csv` | exact InChIKey overlap with both training halves, on the subset |
| `T3_exact_overlap.csv` | the same on the full set — reproduces the published §3a table |
| `T3_ligand_novelty_Agroup_subset.csv` | novelty tiers against the structure half's own 13,590 ligands, on the subset |
| `T3_ligand_novelty_Agroup.csv` | the same on the full set — the table §3c's correction rests on |
| `T3_novelty_tiered_ef.csv` | enrichment computed separately per novelty tier |
| `T3_target_mirroring.csv` | each T3 target's closest training-set homologue (mmseqs, cov ≥50%) |
| `T3_target_redundancy.csv` | all-vs-all identity within the subset (pairs ≥20%) |
| `T3_capability_map_subset.csv` | EF@1% per model over (homology-to-training band) × (ligand-similarity tier), with per-cell SE and a thin-cell flag |
| `T3_capability_step_tests_subset.csv` | the two tests on that map, BH-corrected: in-training vs not, and across the three sequence-novel bands |

## HypSeek official weights — added 2026-09-08

| File | What |
|---|---|
| `T1_hypseek_official.md` | write-up: `_vs` vs `_rk`, both `alpha_prot` settings, the retraction |
| `T1_hypseek_official.csv` | T1 across five HypSeek weight variants |
| `T3_hypseek_official.csv` | T3 four layers, same variants |

## Target swap — added 2026-09-09

| File | What |
|---|---|
| `T3_target_swap.csv` | task-specific model panel × L1/L4 × 3 metrics, correct vs swapped target, paired p |

## DrugJEPA — added 2026-09-22

A fourth-party model (`github.com/Saoge123/DrugJEPA`, MIT, under submission),
built on Uni-Mol like most of this table: contrastive learning plus a Joint
Embedding Predictive Architecture and a Mixture-of-Experts encoder. It is the
only model here with prospective wet-lab validation — TYK2, TAAR1 and PUS1,
hit rates 40% / 20% / 30%.

**It shares its training corpus with four models already here.** Verified by
sha256, not by reading the paper: see model-side convention 5 in
[`REPRODUCING.md`](../REPRODUCING.md). That makes it a controlled architecture
comparison, and it means the exposure figures below are a property of the
corpus, not of this model.

Its rows were **appended** to the existing tables; no other model row was
recomputed. Seventeen supporting tables carry a `drugjepa` row, including the
per-model training-set stratifications for which its training corpus is
established byte for byte.

**Where it lands.** On the public benchmarks its DUD-E EF@1% is 47.09 and its
LIT-PCBA EF@1% is 6.34. On the time split it is second overall and **first on L4
AUROC (0.707)**, the layer with no training homologue. It is the only model in
this table whose public-benchmark rank is *worse* than its time-split rank —
which is the shape the wet-lab result would predict, and the opposite of what
the rest of the table does.

Target swap behaves as for every other model: replacing the pocket with a
random target's destroys performance (L1 35.79 → 0.14, p = 2.4e-10), replacing
it with a same-family target's changes nothing (L1 35.01 → 36.73, p = 0.59).

## Excluded historical model

LiTENCLIP v2 / HypLiTEN was evaluated during development but is not part of the
2026-10-06 release. Its training-data provenance and stable public checkpoint
did not meet the release requirement. Its code, checkpoint, alpha sweeps and
rows were removed from the current tree; Git history retains the audit trail.
`LiTENCLIP` in current tables means v1.
