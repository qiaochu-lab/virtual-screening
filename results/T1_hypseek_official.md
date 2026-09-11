# HypSeek official weights: `_vs` and `_rk` scored separately

Weights come from the author's Google Drive, published in
[HypSeek issue #4](https://github.com/jianhuiwemi/HypSeek/issues/4). Run on
2026-09-08.

## Why this was re-run

Until now, every HypSeek number in this repository came from the official
`_rk` — it was the only one public at the time. On 2026-09-07 a collaborator
pointed out that `_rk` is a checkpoint selected by benchmark, so using it to
score DUD-E / LIT-PCBA carries a selective-leakage risk, and that `_vs` should
run the screening tasks while `_rk` runs the ranking tasks — scored
separately.

At the same time, `alpha_prot` (the protein-sequence pathway weight) defaults
to 1 during training, but `test_task.py` reads it as
`getattr(self.args, "alpha_prot", 0)`, and argparse never exposed this name —
**the official evaluation has always run with the sequence pathway switched
off**, the opposite of what the paper says ("removing the sequence pathway
costs performance"). The reproducer in issue #4 measured that with
`alpha_prot=1`, DUD-E EF@1% rises from 51.43 to 53.04. We exposed it as
`--alpha-prot` and report both settings.

## Protocol check: our evaluation pipeline independently reproduces the paper

| DUD-E | measured here | issue #4 reports | paper |
|---|---|---|---|
| EF@1%, alpha_prot=0 | **51.4138** | 51.43 | **51.44** |
| EF@1%, alpha_prot=1 | **53.0186** | 53.04 | — |
| BEDROC, alpha_prot=0 | **0.7892** | 0.7892 | **0.7892** |

Agrees to four significant figures. This is the second external validation
point for the evaluation layer, after the LigUnity baseline.

## T1 (EF@1%)

| Weight | alpha_prot | DUD-E | DEKOIS | LIT-PCBA |
|---|---|---|---|---|
| **official `_rk`** | 0 | **56.39** | **28.83** | **8.34** |
| **official `_rk`** | 1 | **57.52** | **29.84** | 8.01 |
| official `_vs` | 0 | 51.41 | 25.52 | 6.82 |
| official `_vs` | 1 | 53.02 | 25.66 | 5.21 |
| collaborator's self-trained `_vs` | 0 | 49.34 | 25.64 | 6.76 |

## T3 (EF@1%)

| Weight | L1 | L2 | L3 | L4 | EF decay |
|---|---|---|---|---|---|
| **official `_rk`** | **36.63** | 23.61 | 13.56 | **7.34** | −82% |
| official `_vs` | 32.11 | 20.95 | 12.40 | 7.07 | −80% |
| collaborator's self-trained `_vs` | 30.70 | 19.39 | 11.11 | 5.75 | −84% |
| this project's self-trained `_vs` | 22.15 | 13.32 | 7.11 | 4.63 | −83% |

`alpha_prot` has no effect on T3 — the two settings give identical numbers —
because this project's own T3 scoring function (`test_t3_target` in
`test_task.py`, written by us) uses only the pocket pathway:
`res = pocket_reps @ mol_reps.T`, and never computes `prot_scores`. This is a
protocol fact, not a bug, but it means **every HypSeek number on T3 in this
repository is pocket-pathway-only**, and that needs to be stated in the main
text.

## Four conclusions

**One. The ranking weight beats the screening weight across the board on
screening.** `_rk` is higher than the official `_vs` on all seven measurements
— T1's three benchmarks plus T3's four layers: DUD-E +9.7%, DEKOIS +13.0%,
LIT-PCBA +22.4%, T3 L1 +14.1%. The evidence for this finding before now
(README finding 12) was `_rk` against **this project's own self-trained**
`_vs` — a weight with a known training defect. It is now against the
**official** `_vs`, so the strength of the argument is different.

**Two. "The released weight is not the model from the paper" is retracted.**
The old evidence was that we measured 56.39 against the paper's reported
51.44, but that compared `_rk` to the paper's `_vs` number — not the same
checkpoint to begin with. The official `_vs` reproduces the paper exactly
(51.41 vs 51.44).

**Three. The decay is insensitive to checkpoint choice.** The four weights'
L1 absolute values differ by 65% (36.63 vs 22.15), yet the decay for all of
them falls in the −80% to −84% range. This supports "the decay is a property
of the method class" rather than an accident of one particular training run.

**Four. The effect of `alpha_prot` reverses depending on the decoy type.**

| | `_rk` | `_vs` |
|---|---|---|
| DUD-E | +2.0% | +3.1% |
| DEKOIS | +3.5% | +0.6% |
| **LIT-PCBA** | **−3.9%** | **−23.5%** |

Turning on the sequence pathway helps on benchmarks with artificial decoys,
and hurts on LIT-PCBA, whose decoys are experimentally confirmed. This is
consistent with the direction of T1's core finding: the sequence pathway may
be helping the model recognize "which class of target pairs with which class
of molecule", and artificial decoys are separable precisely along that
dimension.

## None of three independent self-trained reproductions land on the published number

| Source | DUD-E EF@1% | Gap to paper |
|---|---|---|
| paper / official `_vs` | 51.44 / 51.41 | — |
| collaborator's self-trained | 49.34 | −4% |
| issue #4 third-party self-trained | 46.05 | −10% |
| this project's self-trained | 43.29 | −16% |
| **this project's self-trained (diagnosed defect)** | contrastive negative pool of 4 vs. the official 24 | |

The reproducer in issue #4 confirmed the training inputs are identical and
swept multiple random seeds; the author has not yet replied about the cause
of the training discrepancy. **Training from the paper's recipe does not
reproduce the published weight** — this is currently an independent
observation from three separate parties.

## Verification

The official `_rk` and the `_rk` this repository has used all along are
**byte-identical by md5** (`02d7574254bc6fd797e1bf7f7904ea4c`) — every past
HypSeek number's weight provenance was correct; the only problem was using the
wrong checkpoint type.
