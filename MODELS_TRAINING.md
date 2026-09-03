# Training HypSeek's screening weight

HypSeek releases only `_rk`, the checkpoint selected for affinity **ranking**.
That one weight leads every axis this benchmark measures, so the obvious question
is what the **screening** weight (`_vs`) would do. Training it took three
attempts; the first two produced a model that had never updated a single
parameter, and the reason is worth recording because nothing in the logs said so.

## The failure: every batch was silently dropped

Two 50-epoch runs finished and scored near-random (DUD-E EF1% 1.21 against the
released weight's 56.39, AUROC 0.573 against 0.967). Before treating that as
evidence about the training objective, we checked whether training had occurred:

- Both encoder towers were **byte-identical to their pretrained initialisation**
  — 0 of 193 tensors changed in each.
- `curv` still sat at `log(1.0)`; all three `alpha` parameters at `log(128^-0.5)`.
- `checkpoint_best.pt`, `checkpoint_last.pt` and `checkpoint41.pt` were the same
  weights and scored identically to four decimals.
- `valid_bedroc` printed **0.034 at every epoch of both runs** — constant, not flat.
- The fp16 loss scale climbed to 524,288, which only happens when no gradient
  ever overflows.

The cause: **`--batch-size 24` does not fit in 24 GB** for this model (129M
parameters, an ESM sequence tower, up to 16 ligands per pocket). Unicore catches
CUDA OOM in the forward/backward pass, logs a warning, and **skips the batch**.
The official run logged **94,000 OOM recoveries** — every batch, for 50 epochs.
Nothing else failed, so the run "completed" and wrote checkpoints of the
untouched initialisation.

The stock logging hid it: `loss=-1`, `bsz=0` and `gnorm=0` are placeholders that
appear whenever no batch survives, and they look identical to a quiet, healthy
run. A single-batch probe outside the trainer produced a gradient norm of 130
with all encoders receiving gradients, which is what ruled out the loss, the
data, and the leakage filtering as causes.

## The fix

`--batch-size 4 --update-freq 6` on 4 GPUs = **effective batch 96, identical to
the official 24 × 4**. Batch 6 also fits on one GPU but still OOMs occasionally
under DDP, and unicore's OOM recovery is not DDP-safe — one rank skipping a batch
while the others proceed desynchronises the collective and aborts with SIGABRT.
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces fragmentation.

With that, the log finally carries real numbers:
`loss=1.246, loss_cone=0.092, R_ang=0.209, loss_hcc_poc=0.606`, and 0 OOM events.

## Configuration

Official `train.sh` otherwise unchanged. Four data files had to be created
because the two projects package the same data differently: `casf.lmdb` split
into `valid_lig.lmdb` + `valid_prot.lmdb`, `fep_assays.json` symlinked to our
`fep_assay_ids.json`, `test_datasets/` symlinked, and a `cache/` directory that
`PairDataset` writes into without creating. `mode=CASF` — one of the script's own
two options — selects the best checkpoint by `valid_bedroc`, which is what makes
the result a screening weight; `mode=FEP` reproduces the released `_rk`.

Leakage removal is visible in the log: 26,748 assays → 26,729 after dropping FEP
assays → **18,316** after dropping every protein appearing in DUD-E, DEKOIS,
LIT-PCBA or CASF.

## Results, seed 1

With training actually running, the weight moved (molecule tower now 68.9 away
from its initialisation, against 0.000000 in the broken runs) and the numbers are
meaningful:

| | our `_vs` | released `_rk` | gap |
|---|---|---|---|
| DUD-E EF1% | 43.36 | **56.39** | −23% |
| DUD-E BEDROC | 0.684 | **0.884** | −23% |
| DUD-E AUROC | 0.922 | 0.967 | −4.7% |
| DEKOIS EF1% | 23.35 | **28.83** | −19% |
| DEKOIS AUROC | 0.921 | 0.964 | −4.5% |

The released weight is `checkpoint_avg_41-50_rk.pt` — an average of the last ten
epochs — so we averaged ours the same way before comparing. **It changed almost
nothing** (DUD-E EF1% 43.29 → 43.36), which rules that difference out.

**Where the gap sits is more informative than its size.** AUROC differs by 4–5%,
so overall ranking ability is close. EF1% and BEDROC differ by ~20%, and both are
early-recognition metrics: the gap is concentrated in the very top of the ranked
list, which is the part a screening campaign acts on.

## Resolved: the gap was our own negative pool

A collaborator trained the same weight independently and shared the checkpoint.
Run through **our** pipeline — same code, same data, same metrics — it lands
close to the paper where ours does not:

| | paper | collaborator `_vs` | ours (seed 1) | released `_rk` |
|---|---|---|---|---|
| DUD-E AUROC | 0.9435 | **0.9388** (−0.5%) | 0.922 (−2.3%) | 0.967 (+2.5%) |
| DUD-E BEDROC | 0.7892 | **0.7589** (−3.8%) | 0.684 (−13.3%) | 0.884 (+12.0%) |
| DUD-E EF1% | 51.44 | **49.34** (−4.1%) | **43.29 (−15.8%)** | 56.39 (+9.6%) |
| LIT-PCBA EF1% | 6.81 | **6.76** (−0.7%) | 4.44 (−34.8%) | 8.34 (+22.5%) |

**Our evaluation is not the variable.** The paper reports LigUnity-pocket as a
baseline, and our independent measurement of it matches to four decimal places
(AUROC 0.8922 vs 0.8922, EF5% 13.57 vs 13.57, BEDROC 0.6528 vs 0.6526, EF1%
42.57 vs 42.63). So the 15.8% is a training deficit on our side.

### What differed

Every hyperparameter matches the official `train.sh` — learning rate, schedule,
warmup, epochs, clip norm, fp16 settings, `max-pocket-atoms`, `max-lignum`,
`learn-curv`, `protein-similarity-thres` — and the task code loads the same
PocketAffDB files. One thing differed:

| | per-GPU batch | update-freq | GPUs | optimiser batch | **negatives per step** | DUD-E EF1% |
|---|---|---|---|---|---|---|
| official | 24 | 1 | 4 | 96 | **24** | 51.44 |
| collaborator | 12 | 1 | **1** | 12 | **12** | 49.34 |
| **ours** | 4 | **6** | 4 | 96 | **4** | 43.29 |

We could not fit batch 24 in 24 GB (471,107 OOM events in the first run), so we
dropped the batch and raised `update-freq` to keep the optimiser's effective
batch at 96 — reasoning that is correct for ordinary supervised training and
**wrong for contrastive learning**.

`three_hybrid_loss` builds its similarity matrix from the ligands in the current
forward pass, and there is no cross-GPU `all_gather` anywhere in the model or
the loss. Gradient accumulation sums the gradients of six small-batch losses; it
does not construct one large similarity matrix. **The optimiser saw a batch of
96; the contrastive loss saw 4 negatives instead of 24.**

The three configurations are monotone in the negative pool (24 → 12 → 4 giving
51.44 → 49.34 → 43.29) and unrelated to the optimiser batch (96 → 12 → 96). The
collaborator's run had **one eighth** of the official optimiser batch and still
reproduced the paper.

It also explains the *shape* of the deficit. AUROC moves 2.3% while EF1% moves
15.8%: fewer negatives still teach coarse separation, but not discrimination
among the hardest, most similar candidates — which is the top of the ranked list.

### The second mistake, which is the more transferable one

Batch 24 OOM'd and batch 6 crashed, so we retreated to 4. We never tried 8, 12
or 16, and never tried a single GPU. Both failures were measured **under 4-GPU
DDP**, where each rank carries extra memory and — worse — unicore's OOM recovery
is unsafe: one rank skipping a batch while the others keep communicating
desynchronises NCCL and aborts the process. The collaborator fit batch 12 on a
single 4090, the same 24 GB.

**The limit we measured was our parallelism, not our memory**, and we treated a
DDP-specific crash as a hardware ceiling. Before shrinking a batch, ask what a
different parallel setup allows.

## What this does and does not establish

**Withdrawn.** An earlier version of this page, and finding 9 in the README,
said that training the screening-selected weight yields a materially weaker
screener. That was our training deficit, not a property of the objective, and
the claim is retracted.

**What survives, with the collaborator's paper-faithful weight in place of
ours.** The released `_rk` still leads on screening at every T3 layer
([`results/T3_hypseek_three_way.csv`](results/T3_hypseek_three_way.csv)):

| Layer | `_rk` | collaborator `_vs` | ours |
|---|---|---|---|
| L1 EF1% | **36.63** | 30.70 | 22.15 |
| L2 EF1% | **23.61** | 19.39 | 13.32 |
| L3 EF1% | **13.56** | 11.11 | 7.11 |
| L4 EF1% | **7.34** | 5.75 | 4.63 |
| L1 AUROC | 0.923 | 0.900 | 0.888 |
| L4 AUROC | 0.710 | 0.683 | 0.682 |

So **a checkpoint selected on FEP ranking is also the better screener** — the
observation that motivated this whole exercise. The margin is 16% at L1, not the
40% our own weight suggested. And the L1→L4 decay is indifferent to which weight
is used (−50% / −54% / −53%), which is worth noting on its own: **the benchmark's
headline finding does not depend on getting the training right.**

**A separate anomaly, which we did not go looking for.** The weight published on
HuggingFace scores *above* the paper's own reported screening numbers — DUD-E
EF1% 56.39 against 51.44 (+9.6%), LIT-PCBA 8.34 against 6.81 (+22.5%), with our
pipeline validated against the paper's LigUnity baseline to four decimals. **The
released checkpoint is therefore not the model behind Table 1.** The paper never
mentions two checkpoints at all; the `_vs`/`_rk` distinction exists only in the
release. Anyone downloading the weight is not getting the published model — in
this instance a better one on screening, but the two are not interchangeable.

**One correction.** We had flagged the curvature parameter never moving during
training as a possible bug. The paper states κ is **fixed to 1** by design, so
that observation is expected. The three `alpha` parameters sitting at their
initial values across all three independently trained weights remains
unexplained.

## The transferable lesson

A run that consumes GPU-hours, writes checkpoints, and exits cleanly is not
evidence that it trained. The cheap check is to compare the saved weights against
the initialisation — it takes seconds and would have caught this before the first
50 epochs finished, let alone the second. That check is now a gate in the
post-training chain: distance 0 stops the pipeline instead of feeding empty
results into evaluation.
