# T1 — Enrichment Retrieval

**Question:** can the model rank known actives above decoys, on the benchmarks
the field already uses?

**Status:** 9 of 9 models have DUD-E and DEKOIS; SPRINT's DUD-E and LIT-PCBA are queued. CASF-2016 done for four models.

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

## Results — nine models, three benchmarks

All numbers recomputed by [`eval/`](../eval/) from raw per-molecule scores.
Machine-readable: [`results/T1_main.csv`](../results/T1_main.csv).

**EF1%**

| Model | Protein input | DUD-E (102) | DEKOIS (81) | LIT-PCBA (15) |
|---|---|---|---|---|
| **HypSeek `_rk`** | 3D pocket, hyperbolic | **56.39** | **28.83** | 8.34 |
| LigUnity ensemble | pocket + sequence + HGNN | 52.52 | 28.07 | 7.67 |
| LiTENCLIP | 3D pocket | 43.95 | 23.97 | 6.56 |
| LigUnity-pocket | 3D pocket | 42.57 | 24.62 | 7.30 |
| LigUnity-protein | sequence | 36.69 | 27.04 | 6.22 |
| BindCLIP-randneg | 3D pocket | 32.81 | 19.36 | 6.36 |
| DrugCLIP | 3D pocket | 31.94 | 17.86 | 5.55 |
| BindCLIP-hardneg | 3D pocket | 27.64 | 17.56 | 6.23 |
| ConGLUDe | sequence + structure graph | 26.48 | 16.55 | **13.24** ⚠️ |
| ConPLex | sequence only | 18.70 | 10.46 | 2.15 |
| SPRINT | SaProt 3Di sequence | running | 3.09 | running |

**AUROC** tells the same story more smoothly: HypSeek 0.967 / 0.964 / 0.613,
DrugCLIP 0.807 / 0.791 / 0.572, ConPLex 0.683 / 0.666 / 0.554.

**Three things worth reading off this table.**

1. **Representation richness orders the models, and it holds across three
   independent benchmarks.** Sequence only (ConPLex) < sequence + graph
   (ConGLUDe) < explicit 3D pocket (everything else). The same ordering appears
   in T3, on completely different data.
2. **One checkpoint leads everywhere.** HypSeek's `_rk` — the weight its authors
   selected for *affinity ranking*, and the only one they released — is first on
   all three screening benchmarks, first on T3 ranking, and first on CASF. That
   makes "train the screening weight `_vs` and see" a real question rather than
   a completionist one.
3. ⚠️ **ConGLUDe's 13.24 on LIT-PCBA beats every pocket model** — but on 13 of
   15 targets, so the comparison is not like-for-like. If it survives the two
   missing targets it is the most interesting cell in the table, because
   LIT-PCBA is the only benchmark here with experimentally confirmed inactives.

**Target counts differ by model** and the differences are recorded rather than
hidden: ConPLex loses one DEKOIS and one LIT-PCBA target to ProtBert's
2,000-residue limit; ConGLUDe loses two LIT-PCBA targets; SPRINT loses targets
wherever foldseek cannot produce 3Di tokens. Comparing EF across models with
different denominators needs this in view.

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
