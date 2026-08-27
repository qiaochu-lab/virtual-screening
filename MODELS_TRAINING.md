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

## Status

Training is running with the corrected configuration. **Nothing from the first
two runs should be quoted** — those numbers describe a pretrained encoder with an
untrained head, not a training outcome. Results will be reported with at least
two seeds, since a single run cannot settle anything about a model whose
published weights are known to be seed-sensitive.

## The transferable lesson

A run that consumes GPU-hours, writes checkpoints, and exits cleanly is not
evidence that it trained. The cheap check is to compare the saved weights against
the initialisation — it takes seconds and would have caught this before the first
50 epochs finished, let alone the second.
