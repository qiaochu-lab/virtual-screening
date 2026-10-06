# Canonical result set — 2026-10-06

This page is the release boundary for the final benchmark tables.  Files under
`results/release-2026-10-06/` are the tables to cite and reproduce.  Other files
under `results/` are supporting analyses or retained research intermediates.

## Locked policy

- T3 uses the strict target convention: L1/L2 are retained and L3/L4 targets
  seen by any of the four known training-set groups are excluded.  The target
  counts are **56 / 178 / 20 / 41** before model-specific missing targets.
- **DrugJEPA is included.**
- **LiTENCLIP v2 / HypLiTEN is excluded.**  Its code, checkpoint and rows are not
  part of this release.  LiTENCLIP in the release means the public v1 model.
- Screening tables use the official HypSeek `_vs` checkpoint.  T2 affinity
  ranking uses the official `_rk` checkpoint.  Target-swap tables predate the
  `_vs` release and therefore retain `_rk`; this is stated in their provenance.
- ConGLUDe's complete training-target list is unavailable.  “Strict” is exact
  for the four known training groups, but is not a claim of absolute zero
  leakage for ConGLUDe.

## Canonical files

| File | Role |
|---|---|
| `T1_main.csv` | Standard-benchmark screening results, HypSeek `_vs` |
| `T3_main_strict.csv` | Main strict L1–L4 screening table, HypSeek `_vs` |
| `T2_on_T3_strict.csv` | Strict affinity ranking, HypSeek `_rk` |
| `T3_recall_at_k_strict.csv` | Strict top-k screening recall |
| `T3_novelty_tiered_ef_strict.csv` | Strict ligand-novelty enrichment |
| `T3_chemical_memory_pref_strict.csv` | Strict chemical-memory preference |
| `T3_target_swap_strict.csv` | Strict random target swap, historical `_rk` run |
| `T3_target_swap_family_strict.csv` | Strict same-family target swap, historical `_rk` run |
| `T5_structure_source_strict.csv` | Strict structure-source control |
| `T3_targets_strict.csv` | Exact strict target set with final layer labels rewritten to 56 / 178 / 20 / 41 |
| `T3_target_exposure.csv` | Target-by-training-group exposure audit |
| `SHA256SUMS` | Integrity hashes for this release directory |

Regenerate the directory deterministically with:

```bash
python tools/build_release_20261006.py
sha256sum -c results/release-2026-10-06/SHA256SUMS
```

The per-molecule NPZ score packages needed to recompute metrics are distributed
in the companion data package rather than GitHub because of their size.

The older root-level `T3_vsds_matched_strict.csv` is a compatibility membership
filter for scripts that relabel internally; it deliberately retains the
pre-correction layer labels. Do not count layers from that compatibility file.
