# physics/ — physics-based methods and the FEP benchmark

Everything that produces a *physical* quantity, or compares one against the
retrieval models. It is a directory of its own rather than living under a task,
because it serves two: the FEP benchmark supplies T2's within-series ranking
numbers, and the same code plus Boltz-2 is the whole of T6. Read [`../tasks/T6-physics.md`](../tasks/T6-physics.md) first
for what the numbers mean; this file is the code map.

## The FEP benchmark

16 systems, 461 ligands: JACS (BACE, CDK2, JNK1, MCL1, p38, PTP1B, thrombin,
TYK2) + Merck (CDK8, c-Met, Eg5, HIF-2α, PFKFB3, SHP-2, SYK, TNKS2). Shipped
with the LigUnity data release; ligand counts were verified system by system
against the published sets.

It matters here because it is **congeneric** — one scaffold per system,
substituent changes only. That is exactly the regime where free-energy methods
are validated, and (as it turns out) the only regime where retrieval models can
rank at all.

## Files

| File | What it does |
|---|---|
| `run_fep.sh` | run the retrieval models on all 16 systems |
| `patch_fep_save.py` | one-line patch so the FEP task persists `saved_preds.npy` |
| `fep_recover_preds.py` | reconstruct scores from stored embeddings (`pocket_emb @ mol_emb.T`, then max over pockets) — identical to the official computation, so old runs need not be repeated |
| `score_fep.py` | per-system Spearman / Pearson / Kendall, signed |
| `fep_vs_t3_same_targets.py` | the paired test on the 14 targets present in both datasets |
| `fep_compare_physics.py` | retrieval vs the published physics reference, per system |
| `prep_boltz_fep.py` | Boltz-2 inputs for all 461 complexes |
| `fep_truncate.py` | truncate systems over Boltz-2's 1170-residue limit to the binding domain |
| `run_boltz_fep.sh` | launch Boltz-2, 3 shards |
| `t6_boltz_affinity.py` | cross-target correlation of Boltz-2's affinity head against measured pAffinity |

## Two conventions that cause sign errors

1. **Boltz-2's `affinity_pred_value` is lower-is-stronger**; pAffinity is
   higher-is-stronger. Every correlation here flips the sign explicitly.
2. **The official FEP scoring reports R² and clamps it to 0 when `corr < 0`**,
   collapsing "systematically backwards" and "no relationship" into one number.
   A negative correlation is real information, so signed Spearman is reported
   alongside.

## Adding a physics method

The interface is one float array per target plus the label array. Drop them
where the runners write theirs and [`../eval/`](../eval/) computes every metric
unchanged — nothing in the metric layer is model-specific. Pockets are already
extracted at 4 / 5 / 6 / 8 Å; use 6 Å to stay comparable with the retrieval side.
