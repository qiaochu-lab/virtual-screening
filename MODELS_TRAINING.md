# Training HypSeek's screening weight — a failed reproduction, documented

HypSeek releases only `_rk`, the checkpoint selected for affinity **ranking**.
Since that one weight leads every axis this benchmark measures, the obvious
question is what the **screening** weight (`_vs`) would do. We tried to train it
and the attempt failed in a way worth recording: **no parameter ever updated.**

## What was run

Official `train.sh` with four necessary changes, none of them hyperparameters:
data path pointed at LigUnity's `test_datasets` (HypSeek ships no data and its
README says the data comes from LigUnity), `save_root` moved off the hardcoded
`/save_root`, seed made a parameter, and `mode=CASF` — one of the script's own
two options, which selects the best checkpoint by `valid_bedroc`. That選択 is
what makes the result a screening weight; `mode=FEP` reproduces the released
`_rk`.

Four files had to be created because the two projects package the same data
differently: `casf.lmdb` split into `valid_lig.lmdb` + `valid_prot.lmdb`,
`fep_assays.json` symlinked to our `fep_assay_ids.json`, `test_datasets/`
symlinked, and a `cache/` directory that `PairDataset` writes into without
creating.

Leakage removal worked and is visible in the log: 26,748 assays → 26,729 after
dropping FEP assays → **18,316** after dropping every protein that appears in
DUD-E, DEKOIS, LIT-PCBA or CASF.

Two runs completed 50 epochs each: 3 GPUs (effective batch 72) and the official
4 GPUs (96).

## What came out

| Benchmark | trained `_vs` | released `_rk` |
|---|---|---|
| DUD-E EF1% | **1.21** | 56.39 |
| DUD-E AUROC | **0.573** | 0.967 |
| DEKOIS EF1% | **1.29** | 28.83 |
| DEKOIS AUROC | **0.595** | 0.964 |

Near-random. Before reporting that as "the screening objective trains worse",
we checked whether the model had trained at all. It had not:

- The molecule tower and pocket tower are **byte-identical to the pretrained
  initialisation** — 0 of 193 tensors changed in each.
- `curv` is still `log(1.0)`; `mol_alpha`, `pocket_alpha` and `protein_alpha`
  are all still `log(128^-0.5)`.
- `checkpoint_best.pt`, `checkpoint_last.pt` and `checkpoint41.pt` are the same
  weights, and score identically to four decimals.
- `valid_bedroc` printed **0.034 at every epoch of both runs** — constant, not
  merely flat.
- The fp16 loss scale climbed to 524,288, which happens when no gradient ever
  overflows, i.e. when gradients are zero.

The dataset is not the problem: the training split builds 45,144 samples, and
all 36,662 pocket names in the label file resolve in the pocket LMDB.

## Leading hypothesis, untested

The loss is `loss_hcc + γ·loss_cone + loss_reg`, and `loss_cone` is built from
`relu(dist − r_k)` and `relu(φ − η·ω)`. If the pretrained embeddings already
satisfy both margins for every pair, those terms are exactly zero **with zero
gradient**. That would explain a constant validation metric, a growing loss
scale, and untouched weights, all at once. Confirming it needs an instrumented
run that prints the three loss components separately — the stock logging emits
`loss=-1`, `bsz=0`, `gnorm=0` as placeholders and never records the real values,
which is why this went unnoticed for two full runs.

## What this does and does not license

- ❌ It does **not** support any claim about the screening objective, about
  `_vs` versus `_rk`, or about HypSeek's trainability. The numbers above are a
  pretrained encoder with an untrained head.
- ✅ It does establish that **our reproduction of this training recipe is
  broken**, and it pins the failure to "no gradient reaches the parameters"
  rather than to data, leakage handling, or batch size.
- ✅ Effective batch size 72 vs 96 changed nothing — both produced the identical
  constant metric, so the earlier worry about running on 3 GPUs was unfounded.

Until an instrumented run shows a non-zero, decreasing loss, no trained-weight
result from this project should be quoted.
