# Result tables

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
| `T3_target_swap.csv` | 10 models × L1/L4 × 3 metrics, correct vs swapped target, paired p |

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

Its rows were **appended** to the existing tables; no published row was
recomputed. Seventeen tables now carry a `drugjepa` row, including four that
LiTENCLIP 2.0 cannot appear in (`T3_per_model_layers.csv`, its `_ctrl` and
`_seen_effect` variants, and `T3_train_set_crossover.csv`) because those
stratify by *the model's own* training set, which for LiTENCLIP 2.0 is unknown
and for DrugJEPA is established byte for byte.

**Where it lands.** On the public benchmarks it is third on DUD-E (EF@1% 47.09,
behind HypSeek `_vs` 51.41 and LiTENCLIP 2.0 50.57) and unremarkable on
LIT-PCBA (6.34). On the time split it is second overall and **first on L4
AUROC (0.707)**, the layer with no training homologue. It is the only model in
this table whose public-benchmark rank is *worse* than its time-split rank —
which is the shape the wet-lab result would predict, and the opposite of what
the rest of the table does.

Target swap behaves as for every other model: replacing the pocket with a
random target's destroys performance (L1 35.79 → 0.14, p = 2.4e-10), replacing
it with a same-family target's changes nothing (L1 35.01 → 36.73, p = 0.59).

## LiTENCLIP 2.0 — added 2026-09-17

A third-party model handed over mid-project. It is a **different architecture**,
not LiTENCLIP with new weights: three towers (molecule / pocket / protein
sequence), Lorentz hyperbolic embeddings, built inside HypSeek's codebase. See
convention 4 in [`REPRODUCING.md`](../REPRODUCING.md) for the porting hazard and
for what is and is not known about its training data.

Its rows were **appended** to the existing tables — no published row was
recomputed. Tables that now carry a `litenclip_v2` row: `T1_main.csv`,
`T3_main.csv`, `T3_main_ci.csv`, `T3_main_vsds_subset.csv`,
`T3_novelty_tiered_ef_subset.csv`, `T3_recall_at_k.csv`,
`T3_chemical_memory_pref.csv`,
`T3_target_swap.csv`, `T3_target_swap_family.csv`, `T5_structure_source.csv`,
`T2_on_T3_subset.csv`.

Two of those tables were missing `hypseek_official_vs` entirely; that gap was
filled in the same pass (`T3_chemical_memory_pref.csv`, `T3_recall_at_k.csv`,
`T2_on_T3_subset.csv`).

| File | What |
|---|---|
| `T1_litenclip_v2_alpha_sweep.csv` | the two tower weights swept on DUD-E / DEKOIS / LIT-PCBA, 9 ratios x 4 metrics |
| `T3_litenclip_v2_alpha_sweep.csv` | the same sweep on T3 L1-L4 (350-target subset, corrected layering) |
| `T3_L4_per_target_ef1.csv` | per-target EF@1% on L4 across 12 models, with target metadata |

**On the alpha sweep.** Scoring is
`alpha_poc * max(pocket @ mol) + alpha_prot * max(protein @ mol)`. Every metric
here is rank-based, so scaling both alphas by the same positive constant changes
nothing — only the **ratio** matters, which makes a 2x2 grid over {0,1}^2
over-parameterised: `(0,0)` is degenerate (all scores tied, which under this
project's proportional tie convention lands exactly on the random baseline,
EF = 1.0 and AUROC = 0.5), and `(1,1)` is identical to `(0.5,0.5)`. The files
therefore sweep the ratio instead. Cost was zero GPU: T3 stores both tower
scores separately at inference time, and T1 stores float32 embeddings from which
both towers reconstruct exactly (worst relative error 3.4e-07 over 36 sampled
targets; ratio 1 reproduces the published v2 row digit for digit).

Read these as a **sensitivity analysis, not as tuning**: picking the best cell
while looking at the test layers would make the published numbers no longer
held-out. Selecting the ratio on L1 and applying it unchanged to L4 moves L4's
EF@1% by +0.26 and its AUROC by -0.008.

Note the name collision recorded as trap 17 in `REPRODUCING.md`: in this repo
`T3_novelty_tiered_ef.csv` is the **full 1,144-target** table and the subset
version carries the `_subset` suffix, while on the working machine the
unsuffixed name holds subset numbers. v2's rows went into the subset table,
which is the paper's convention; the full-set table does not carry v2, and
regenerating it would fold the newer tie convention into numbers README cites.
