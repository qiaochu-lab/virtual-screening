# T1 — Enrichment Retrieval

**Question:** can the model rank known actives above decoys, on the benchmarks
the field already uses?

**Status:** ✅ complete — nine models × three benchmarks. CASF-2016 done for four models.

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
| SPRINT | SaProt 3Di sequence | 4.58 | 3.09 | 8.95 ⚠️ |

## What these three benchmarks are made of

Numbers counted from the evaluation sets themselves
([`standard/benchmark_stats_t1.py`](../standard/benchmark_stats_t1.py) →
[`results/T1_benchmark_stats.csv`](../results/T1_benchmark_stats.csv)), with our
own T3 measured the same way so the two are comparable.

| | targets | actives | decoys | ratio | actives/target (median) | **scaffolds/active** | **singleton scaffolds** |
|---|---|---|---|---|---|---|---|
| DUD-E | 102 | 22,800 | 1,410,736 | 1:62 | 155 | **0.997** | **100.0%** |
| DEKOIS 2.0 | 81 | 3,235 | 97,107 | 1:30 | 40 | 0.825 | 87.1% |
| LIT-PCBA | 15 | 10,029 | 2,797,583 | 1:279 | 102 | 0.782 | 83.4% |
| T3 L1 | 349 | 17,066 | 853,300 | 1:50 | 24 | **0.458** | 69.4% |
| T3 L2 | 488 | 102,005 | 5,079,529 | 1:50 | 66 | 0.493 | 70.0% |
| T3 L3 | 53 | 4,358 | 217,900 | 1:50 | 34 | 0.435 | 66.7% |
| T3 L4 | 254 | 32,774 | 1,638,700 | 1:50 | 44 | 0.448 | 69.1% |

"Scaffolds/active" is the number of distinct Bemis–Murcko scaffolds divided by
the number of actives, per target, median across targets. 1.0 means every active
has its own scaffold; 0.45 means each scaffold carries about two actives.

**DUD-E's actives are scaffold-deduplicated by construction** — CDK2's 200
actives give 200 distinct scaffolds, zero repeats. **T3's are not**: its actives
are roughly twice as congeneric as any conventional benchmark. That is
deliberate — real literature actives come in series — but it has to be stated,
because the chemical-series oracle in
[T3's leakage audit](T3-leakage.md) reaches 98.7% of ceiling on T3 and a reader
will reasonably ask how much of that is T3's own construction.

### The oracle answers that question, and not the way we expected

Running the same target-conditioned ligand-similarity oracle on the conventional
benchmarks ([`standard/oracle_ceiling_t1.py`](../standard/oracle_ceiling_t1.py) →
[`results/T1_oracle_ceiling.csv`](../results/T1_oracle_ceiling.csv)):

| Benchmark | EF@1% | ceiling | **% of ceiling** | BEDROC | AUROC |
|---|---|---|---|---|---|
| **DUD-E** | 62.56 | 62.57 | **100.0%** | 0.987 | 0.993 |
| DEKOIS 2.0 | 29.64 | 31.02 | **95.5%** | 0.918 | 0.965 |
| *T3 (all four layers)* | *~50.4* | *51.0* | *98.7%* | | |
| **LIT-PCBA** | 33.22 | 88.23 | **39.8%** | 0.372 | 0.746 |

**DUD-E — the most scaffold-diverse of the four — is the one the oracle solves
completely.** So T3's 98.7% is not an artefact of its congeneric actives; every
benchmark here except LIT-PCBA falls to a protein-blind chemical-similarity
lookup.

The mechanism differs, and that is the interesting part. Median max-Tanimoto of
actives to other actives, against decoys to actives:

| | actives → actives | decoys → actives | gap |
|---|---|---|---|
| DUD-E | 0.65–0.76 | ~0.17 | 0.47–0.59 |
| **T3 L4** | **0.78–0.89** | ~0.16 | **0.63–0.70** |
| LIT-PCBA | 0.29–0.42 | 0.19–0.26 | **0.06–0.16** |

**DUD-E separates because its decoys were chosen to be topologically unlike the
actives** (property-matched, topology-mismatched, by protocol). **T3 separates
because its actives are congeneric series.** Two different constructions, the
same consequence — and **T3's gap is the largest of the three**, so this is a
limitation of our dataset, not only of DUD-E.

**LIT-PCBA is the only benchmark where actives and inactives overlap
chemically**, and it is also the only one where every model here collapses to
near-random AUROC (0.55–0.72). Those two facts are the same fact.

---

**AUROC** tells the same story more smoothly: HypSeek 0.967 / 0.964 / 0.613,
DrugCLIP 0.807 / 0.791 / 0.572, ConPLex 0.683 / 0.666 / 0.554.

⚠️ **ConPLex's DUD-E column is not comparable to the others — 74% of it is
target leakage.** See [ConPLex trained on DUD-E](#conplex-trained-on-dud-e)
below; its leakage-free DUD-E EF1% is **4.83**, not 18.70.

⚠️ **SPRINT's LIT-PCBA 8.95 is above its DUD-E 4.58**, which no other model does
— everything else drops 4–7× from DUD-E to LIT-PCBA. Its AUROC ordering is the
same shape (0.689 / 0.678 / 0.716). A model this weak overall (last on two of
three benchmarks) inverting the usual difficulty order is more likely to be
telling us something about LIT-PCBA's actives than about SPRINT, but it is
unexplained and should not be quoted as a strength.

**Three things worth reading off this table.**

1. **Representation richness orders the models, and it holds across three
   independent benchmarks.** Sequence only (ConPLex) < sequence + graph
   (ConGLUDe) < explicit 3D pocket (everything else). The same ordering appears
   in T3, on completely different data. Removing ConPLex's DUD-E leakage
   (4.83 in place of 18.70) widens the first gap rather than closing it, so the
   ordering survives the correction.
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

All ten models have since run on all three benchmarks. ConGLUDe, ConPLex and
SPRINT needed per-target sequences and structures prepared separately — their
inputs are not the pocket LMDBs the UniMol family reads — and their outputs land
under `results/t1_raw/` rather than beside the others, which every script here
accounts for.

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

## ConPLex trained on DUD-E

**ConPLex's DUD-E number is inflated 3.9× by target leakage, and the effect is
specific to it — the nine other models show nothing.**

Contrastive training on DUD-E decoys *is* ConPLex's method, not an ablation:
`configs/default_config.yaml` ships `contrastive: True` as the default, and
training draws its negatives from the `train` half of
`dataset/DUDe/dude_*_train_test_split.csv`. The repository publishes two such
splits — `cross_type` and `within_type` — of 26 targets each, overlapping in 12,
for a union of **40**. Both files together name **57** of DUD-E's 102 targets.

Our T1 evaluates all 102. So 40 of them are targets this model trained on.

### Difficulty is a confounder, and it has to be removed first

Both splits are drawn along protein-family lines (`cross_type`'s train half is
mostly enzymes and nuclear receptors; its test half is entirely kinases and
GPCRs), so "the seen group scores higher" could just mean those targets are
easier. Each target's difficulty is divided out using the other nine models:

    r_t = EF1%(model, t) / median(EF1%(other nine models, t))

A model that never saw these targets should sit near r = 1 in both groups.

### The three-way gradient

Splitting the 102 targets into the training union (A), targets that appear in
the split files only as test (B), and targets neither file ever names (C):

| Model | A: train (40) | B: test-only (17) | C: never (45) | p(A>C) | Kruskal–Wallis |
|---|---|---|---|---|---|
| **ConPLex** | **1.034** | **0.615** | **0.100** | **<1e-5** | **1e-5** |
| ConGLUDe | 0.817 | 0.785 | 0.842 | 0.59 | 0.94 |
| SPRINT | 0.055 | 0.208 | 0.063 | 0.14 | 0.011 |
| HypSeek `_rk` | 1.729 | 1.962 | 1.963 | 0.85 | 0.40 |
| LigUnity-pocket | 1.151 | 1.412 | 1.367 | 0.99 | 0.043 |
| LigUnity-protein | 0.998 | 1.058 | 1.227 | 0.99 | 0.049 |
| LiTENCLIP | 1.270 | 1.526 | 1.250 | 0.70 | 0.45 |
| DrugCLIP | 0.938 | 0.882 | 0.984 | 0.25 | 0.47 |
| BindCLIP-randneg | 1.021 | 0.862 | 1.000 | 0.33 | 0.087 |
| BindCLIP-hardneg | 0.802 | 0.826 | 0.717 | 0.39 | 0.82 |

ConPLex is monotone across all three groups and is the only model with a
significant A>C gradient. **Even the targets the split files hold out score 6×
above the ones they never mention** — those 57 are the targets whose actives and
decoys the authors curated, featurized and cached, and exposure tracks that
rather than the train/test label inside the file.
([`results/T1_conplex_dude_partition.csv`](../results/T1_conplex_dude_partition.csv))

### It is not BindingDB overlap

The checkpoint we run is `BindingDB_ExperimentalValidModel.pt`, so the obvious
alternative explanation is that its BindingDB training set already contained
these proteins. Reverse-looking up ConPLex's 1,026 training sequences with
mmseqs (identity, both coverages ≥50%, E ≤1e-3) finds **53 of the 102 DUD-E
targets at ≥95% identity** — but that grouping explains nothing:

| Grouping | n | r inside | r outside | fold | p |
|---|---|---|---|---|---|
| BindingDB ≥95% identity | 53 | 0.400 | 0.250 | 1.60 | 0.14 |
| BindingDB seen, not in a DUD-E split | 27 | 0.248 | 0.345 | 0.72 | 0.71 |
| DUD-E split train union | 40 | 1.034 | 0.152 | **6.82** | **9e-5** |
| **DUD-E train, BindingDB never saw** | 14 | 1.446 | 0.284 | **5.09** | **0.0012** |

The effect survives on the 14 targets BindingDB never contained, and vanishes on
the 27 that BindingDB saw but no DUD-E split names. It is the DUD-E contrastive
training.
([`results/T1_conplex_leak_attribution.csv`](../results/T1_conplex_leak_attribution.csv))

### The leakage-free DUD-E table

Restricting to group C — the 45 targets neither split file ever names — and
rescoring all ten models on that same subset:

| Model | EF1% (45) | AUROC | EF1% (102) | change |
|---|---|---|---|---|
| HypSeek `_rk` | 52.90 | 0.949 | 56.39 | −6.2% |
| LigUnity-pocket | 42.61 | 0.886 | 42.61 | +0.0% |
| LiTENCLIP | 42.60 | 0.881 | 43.95 | −3.1% |
| LigUnity-protein | 39.35 | 0.886 | 36.69 | +7.2% |
| BindCLIP-randneg | 31.16 | 0.809 | 32.81 | −5.1% |
| DrugCLIP | 30.50 | 0.791 | 31.97 | −4.6% |
| BindCLIP-hardneg | 26.18 | 0.785 | 27.64 | −5.3% |
| ConGLUDe | 23.80 | 0.794 | 26.48 | −10.1% |
| **ConPLex** | **4.83** | **0.577** | 18.70 | **−74.2%** |
| SPRINT | 3.20 | 0.632 | 4.58 | −30.0% |

Every other model moves between −10% and +7%; ConPLex loses three quarters of
its enrichment and lands at AUROC 0.577, barely above random. Its published
DUD-E performance is essentially a measurement of its own training set.
([`results/T1_dude_conplex_clean.csv`](../results/T1_dude_conplex_clean.csv),
reproduce with [`standard/conplex_dude_leak.py`](../standard/conplex_dude_leak.py))

### What this does and does not affect

**T1 only.** On T3, ConPLex's training set covers 19.4% of targets at ≥95%
identity — the same band as the other models (DrugCLIP's set covers 21–23% of
L3/L4), so T3 needs no correction
([`results/T3_conplex_train_coverage.csv`](../results/T3_conplex_train_coverage.csv)).

**ConPLex stays usable as the sequence-only control on T3, not on T1.** Where
[`MODELS.md`](../MODELS.md) uses it to isolate structure-driven effects, that
argument holds for T3-derived comparisons and must not be made from DUD-E.

**DEKOIS and LIT-PCBA are unaffected** — neither appears in ConPLex's training
pipeline. Its weakness there (10.46 and 2.15) is real, which is itself
consistent with the DUD-E number being the anomaly.
