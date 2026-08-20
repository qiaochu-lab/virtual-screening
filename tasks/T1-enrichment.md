# T1 — Enrichment Retrieval

**Question:** can the model rank known actives above decoys, on the benchmarks
the field already uses?

**Status:** 7 of 9 models complete on all three benchmarks. CASF-2016 done for LigUnity ×2, blocked by a fork bug for the other two.

---

## Data

| Dataset | Targets | Notes |
|---|---|---|
| DUD-E | 102 | Property-matched decoys, 1:50. The field's default. |
| LIT-PCBA | 15 | Experimentally confirmed inactives from PubChem HTS. Built to remove DUD-E's bias. |
| DEKOIS 2.0 | 81 | Property-matched decoys, smaller libraries (~1,200 mol/target). |

All three are public and were downloaded from the model authors' own links —
the three DrugCLIP-family repos point at the same Google Drive folder, verified
by inode (312680540), so every model reads byte-identical files.

## How it was run

Each model runs its **own official code with its own official weights**. Only
the metric computation is unified (see `eval/`). Two of the repos print
aggregate metrics without persisting per-molecule scores; those were patched to
save `saved_preds.npy` + `saved_labels.npy` before any comparison was possible.

```bash
# DrugCLIP / BindCLIP
python ./unimol/test.py ./data --user-dir ./unimol --valid-subset test \
  --results-path <out> --task drugclip --loss in_batch_softmax --arch drugclip \
  --path <ckpt> --max-pocket-atoms 511 --test-task DUDE   # or PCBA / DEKOIS

# LigUnity / LiTENCLIP / HypSeek (different task + loss + arch)
python ./unimol/test.py ./test_datasets --user-dir ./unimol --valid-subset test \
  --results-path <out> --task test_task --loss rank_softmax --arch <arch> \
  --path <ckpt> --max-pocket-atoms 511 --test-task DUDE
```

Then [`timesplit/analysis/score_t3.py`](../timesplit/analysis/score_t3.py) (same scorer,
different `--root`) recomputes all metrics from the raw scores.

## Results so far

Five models complete on all three benchmarks; LiTENCLIP and HypSeek are running.

**DUD-E (102 targets)** — all numbers recomputed by [`eval/`](../eval/) from raw
scores, not copied from papers:

| Model | EF0.1% | EF1% | EF5% | BEDROC | AUROC |
|---|---|---|---|---|---|
| **HypSeek `_rk`** | **59.87** | **56.39** | **17.87** | **0.884** | **0.967** |
| LigUnity (full ensemble) | 59.84 | 52.52 | 15.87 | 0.795 | 0.932 |
| LigUnity pocket + HGNN | 56.76 | 49.35 | 15.10 | 0.749 | 0.912 |
| LigUnity-pocket | 52.15 | 42.57 | 13.57 | 0.653 | 0.892 |
| LiTENCLIP | 52.22 | 43.95 | 14.29 | 0.677 | 0.903 |
| LigUnity-protein | 50.82 | 36.69 | 12.10 | 0.574 | 0.887 |
| BindCLIP-randneg | 46.21 | 32.81 | 10.97 | 0.512 | 0.818 |
| DrugCLIP | 46.58 | 31.94 | 10.66 | 0.500 | 0.807 |
| BindCLIP-hardneg | 41.81 | 27.64 | 9.33 | 0.434 | 0.785 |

**LIT-PCBA (15 targets)** — everything collapses:

| Model | EF0.1% | EF1% | EF5% | BEDROC | AUROC |
|---|---|---|---|---|---|
| **HypSeek `_rk`** | 25.48 | **8.34** | **3.47** | **0.098** | **0.613** |
| LigUnity (full ensemble) | 33.34 | 7.67 | 2.64 | 0.089 | 0.591 |
| LiTENCLIP | 27.77 | 6.56 | 2.86 | 0.076 | **0.612** |
| LigUnity pocket + HGNN | 28.27 | 7.52 | 3.09 | 0.089 | **0.602** |
| LigUnity-pocket | 23.61 | 7.30 | **3.10** | 0.088 | 0.601 |
| LigUnity-protein | **37.28** | 6.22 | 2.18 | 0.075 | 0.563 |
| BindCLIP-randneg | 23.60 | 6.36 | 3.03 | 0.079 | 0.588 |
| BindCLIP-hardneg | 27.45 | 6.23 | 2.70 | 0.077 | 0.578 |
| DrugCLIP | 20.83 | 5.55 | 2.27 | 0.062 | 0.572 |

**DEKOIS 2.0 (81 targets)** — LigUnity variants, plus DrugCLIP/BindCLIP after a
task branch was added to their repos (their `--test-task` originally offered only
`DUDE` and `PCBA`):

| Model | EF0.1% | EF1% | EF5% | BEDROC | AUROC |
|---|---|---|---|---|---|
| **HypSeek `_rk`** | **29.49** | **28.83** | **17.24** | **0.889** | **0.964** |
| LigUnity (full ensemble) | 29.68 | 28.07 | 15.96 | 0.848 | 0.938 |
| LigUnity pocket + HGNN | 28.91 | 26.80 | 15.11 | 0.801 | 0.920 |
| LigUnity-protein | 28.15 | 27.04 | 14.30 | 0.785 | 0.925 |
| LigUnity-pocket | 26.04 | 24.62 | 13.53 | 0.728 | 0.911 |
| LiTENCLIP | 25.47 | 23.97 | 13.84 | 0.712 | 0.909 |

| Model | DUD-E | LIT-PCBA | DEKOIS | CASF-2016 |
|---|---|---|---|---|
| DrugCLIP | ✅ | ✅ | ✅ | — |
| BindCLIP-randneg / -hardneg | ✅ | ✅ | ✅ | — |
| LigUnity-pocket / -protein | ✅ | ✅ | ✅ | — |
| LiTENCLIP | ✅ | ✅ | ✅ | ❌ fork bug |
| HypSeek `_rk` | ✅ | ✅ | ✅ | ❌ fork bug |
| LigUnity-pocket / -protein (CASF) | | | | ✅ |
| ConGLUDe, ConPLex, SPRINT | — | — | — | — |

Reproduction check: every model matched its published values within 2%, and the
LigUnity variants matched to **0.0%** on all three benchmarks — which validates
both the metric layer and the score-saving patches.

⚠️ EF@0.1% on DEKOIS takes **one molecule** per target (libraries are ~1,200), so
per-target values there are an indicator variable, not a rate. Read the mean, not
the cells.

**Two observations already visible:**

1. All models do well on DUD-E and DEKOIS but drop to near-random on LIT-PCBA
   (AUROC 0.56–0.60). LIT-PCBA is precisely the dataset built to remove DUD-E's
   decoy bias. ⚠️ This is *consistent with* the 2019 hidden-bias report but is
   **not independent evidence** — LIT-PCBA is also harder for unrelated reasons
   (larger libraries, lower active ratios, noisier HTS actives).
2. **The released checkpoints were selected against these same benchmarks.**
   DUD-E and LIT-PCBA are difficult to optimize simultaneously, so a public
   checkpoint represents a trade-off chosen with the benchmark scores visible.
   Evaluating it on those benchmarks therefore measures the trade-off as much as
   the model — which is the motivation for T3.

## To finish

**Partly done, resuming.** LiTENCLIP and HypSeek were queued on DUD-E, DEKOIS,
LIT-PCBA and CASF-2016 — both are LigUnity-family, so they read the same
`test_datasets` and needed only a `--test-task` value, no data preparation.
CASF-2016 also fills T2's missing third dataset. Three gaps remain, for three
different reasons:

- **LiTENCLIP × DEKOIS** stopped at 12 of 81 targets when the run was killed
  externally. Nothing wrong with it; needs re-running.
- **HypSeek × LIT-PCBA** hit a real bug: the code replicated the single protein
  embedding once per ligand and then formed `[N_lig, N_lig]`, which for
  LIT-PCBA's largest target (361,997 molecules) asks for **488 GiB**. All rows
  were identical and a `max` was taken over them, so removing the replication is
  value-identical — the DUD-E and DEKOIS numbers above are unaffected.
- **CASF for both** is still blocked. The hardcoded `open("/casf_label_seq.json")`
  was only the first layer; underneath, both forks' `inference_pdbbind` calls
  `model.forward()` with a signature their own model class does not have —
  LiTENCLIP raises `missing 1 required positional argument: 'mol_src_coord'`,
  HypSeek raises `not enough values to unpack (expected 4, got 3)`. This code
  path was evidently never run by the authors. Fixing it means reading each
  model's `forward` and re-deriving which outputs are the pocket and ligand
  embeddings — doable, not yet done. LigUnity's own CASF branch is correct and
  ran fine.

The first two fixes are applied
([`standard/patch_forks_t1.py`](../standard/patch_forks_t1.py)); the LIT-PCBA fix
is what let HypSeek finish, and its DUD-E/DEKOIS numbers are unaffected by it.

One more output quirk, worth knowing if you re-score: HypSeek's LIT-PCBA branch
saves embeddings but no `saved_preds.npy`, and names the pocket array
`saved_pocket_embed.npy` where the rest of the family uses
`saved_target_embed.npy`. Scores are recomputed with the same rule the official
code uses — `(pocket @ mol.T).max(axis=0)`.

**Still to do:** ConGLUDe, ConPLex and SPRINT need per-target sequences or
structures prepared for these three benchmarks (their inputs are not the pocket
LMDBs the UniMol family reads), so they are a separate piece of work rather than
a queue entry.

## Physics methods

⚠️ **Not recommended here.** DUD-E has ~10,000 molecules per target × 102
targets. Co-folding (Boltz-2) takes minutes per complex — completely infeasible.
Docking is possible but the scoring functions are weak. If attempted, do it on a
**subset** (20–30 targets) to answer "is physics also good at enrichment, or
only at ranking?"

## Code

| What | File |
|---|---|
| Launch DrugCLIP on DUD-E / LIT-PCBA | [`standard/run_drugclip.sh`](../standard/run_drugclip.sh) |
| Launch BindCLIP (both weights) | [`standard/run_bindclip.sh`](../standard/run_bindclip.sh), [`standard/run_bindclip_pcba.sh`](../standard/run_bindclip_pcba.sh) |
| Launch on DEKOIS | [`standard/run_dekois.sh`](../standard/run_dekois.sh) |
| Metrics from raw scores, auto-compared to published values | [`eval/score_ligunity.py`](../eval/score_ligunity.py) |
| Metric definitions and their validation | [`eval/`](../eval/) |

**Related ablation — performance vs. distance to the training set.** Not part of
T1 proper, but it uses T1's scores and explains part of the spread:

| What | File |
|---|---|
| EF vs. sequence identity to the nearest training protein | [`standard/t1_sim.py`](../standard/t1_sim.py), [`standard/t1_sim3.py`](../standard/t1_sim3.py) |
| Union of all evaluated models' training sets | [`standard/build_train_union.py`](../standard/build_train_union.py) |
| How much the layer labels shift if a different model's training set is used | [`standard/quantify_train_union.py`](../standard/quantify_train_union.py) |
| Pair-level contamination check (target+ligand already in training) | [`standard/check_pair_contamination.py`](../standard/check_pair_contamination.py) |

Removing training-similar proteins costs 28–45% of EF1%. Both `t1_sim` scripts
bootstrap **at the target level**, not the molecule level — pooling molecules
across targets would understate the variance and silently change what EF means.
