# T1 — Enrichment Retrieval

**Question:** can the model rank known actives above decoys, on the benchmarks
the field already uses?

**Status:** 3 of 9 models done. This is the largest remaining gap.

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

Then [`t3/analysis/score_t3.py`](../t3/analysis/score_t3.py) (same scorer,
different `--root`) recomputes all metrics from the raw scores.

## Results so far

| Model | DUD-E | LIT-PCBA | DEKOIS |
|---|---|---|---|
| DrugCLIP | ✅ 102 | ✅ 15 | ✅ 81 |
| BindCLIP-randneg | ✅ 102 | ✅ 15 | ✅ 81 |
| BindCLIP-hardneg | ✅ 102 | ✅ 15 | ✅ 81 |
| LigUnity ×2, LiTENCLIP, HypSeek, ConGLUDe, ConPLex, SPRINT | — | — | — |

Reproduction check: each model's own numbers matched its published values
within 2%. DrugCLIP on DEKOIS matched the baseline reported in the LigUnity
paper to **0.0%**, which independently validates the score-saving patch.

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

Run the six remaining models. Data and environments are already in place; only
GPU time is needed.

## Physics methods

⚠️ **Not recommended here.** DUD-E has ~10,000 molecules per target × 102
targets. Co-folding (Boltz-2) takes minutes per complex — completely infeasible.
Docking is possible but the scoring functions are weak. If attempted, do it on a
**subset** (20–30 targets) to answer "is physics also good at enrichment, or
only at ranking?"

## Code

| What | File |
|---|---|
| Launch DrugCLIP on DUD-E / LIT-PCBA | [`t1/run_drugclip.sh`](../t1/run_drugclip.sh) |
| Launch BindCLIP (both weights) | [`t1/run_bindclip.sh`](../t1/run_bindclip.sh), [`t1/run_bindclip_pcba.sh`](../t1/run_bindclip_pcba.sh) |
| Launch on DEKOIS | [`t1/run_dekois.sh`](../t1/run_dekois.sh) |
| Metrics from raw scores, auto-compared to published values | [`eval/score_ligunity.py`](../eval/score_ligunity.py) |
| Metric definitions and their validation | [`eval/`](../eval/) |

**Related ablation — performance vs. distance to the training set.** Not part of
T1 proper, but it uses T1's scores and explains part of the spread:

| What | File |
|---|---|
| EF vs. sequence identity to the nearest training protein | [`t1/t1_sim.py`](../t1/t1_sim.py), [`t1/t1_sim3.py`](../t1/t1_sim3.py) |
| Union of all evaluated models' training sets | [`t1/build_train_union.py`](../t1/build_train_union.py) |
| How much the layer labels shift if a different model's training set is used | [`t1/quantify_train_union.py`](../t1/quantify_train_union.py) |
| Pair-level contamination check (target+ligand already in training) | [`t1/check_pair_contamination.py`](../t1/check_pair_contamination.py) |

Removing training-similar proteins costs 28–45% of EF1%. Both `t1_sim` scripts
bootstrap **at the target level**, not the molecule level — pooling molecules
across targets would understate the variance and silently change what EF means.
