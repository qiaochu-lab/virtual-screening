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

Then `tasks/scripts/score_t3.py` (same scorer, different `--root`) recomputes
all metrics from the raw scores.

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
2. A model author confirmed that DUD-E and LIT-PCBA are hard to optimize
   simultaneously, and the released checkpoint was "basically a compromise
   choice". So these benchmarks participated in model selection — which is the
   motivation for T3.

## To finish

Run the six remaining models. Data and environments are already in place; only
GPU time is needed.

## Physics methods

⚠️ **Not recommended here.** DUD-E has ~10,000 molecules per target × 102
targets. Co-folding (Boltz-2) takes minutes per complex — completely infeasible.
Docking is possible but the scoring functions are weak. If attempted, do it on a
**subset** (20–30 targets) to answer "is physics also good at enrichment, or
only at ranking?"
