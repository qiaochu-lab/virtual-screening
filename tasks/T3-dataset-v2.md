# T3 Dataset v2: ≥50 Actives per Target and Class Composition Matched to VSDS-vd

The advisor set two conditions on 2026-09-04: every target must have **at least
50 active molecules**, and target class composition should **reference VSDS-vd**.
This document records how this subset was derived, how much the metrics
changed, and which downstream analyses were affected.

**This is a filtering, not a rebuild.** Every target in the new set was already
among the 1,144 entries and had already been scored by the ten models, and each
target's EF/AUROC is computed within its own candidate pool, independent of
other targets — so switching subsets is just averaging over a different batch
of numbers, **no re-inference required**.

---

## 0. What this benchmark looks like

All numbers are counted directly from the data by
[`timesplit/analysis/benchmark_stats.py`](../timesplit/analysis/benchmark_stats.py)
(→ [`results/T3_benchmark_stats.csv`](../results/T3_benchmark_stats.csv)); none
are quoted from older values in any document.

| | Full set | **350-quota subset (used in the report)** |
|---|---|---|
| Entries / unique targets | 1,144 / 868 | **328 / 293** |
| Active molecules | 156,203 | **89,690** |
| Decoys | 7,789,429 | 4,463,779 |
| Total molecules | 7,945,632 | 4,553,469 |
| Active:decoy ratio | 1:50 | 1:50 |

⚠️ **These are the "designed pool", not the "pool actually scored".** The
numbers are taken from the eval-set jsonl, while the models read from
lmdb — building the lmdb drops a small number of molecules due to RDKit
parsing or conformer-generation failures. Of the 546 targets in L1+L4,
**413 (75.6%)** have mismatched lengths between the two, with jsonl always
larger, a median difference of 2 molecules and a maximum of 78, for a total of
2,495 molecules. That is 0.03% of the total, with no visible effect on any
metric — every EF/Recall cutoff is computed against the **model's array
length**, not the jsonl. But the descriptive statistics below report the
designed values, noted here for clarity.
| Actives per target (median) | 41 | **138** |
| Actives per target (IQR) | 18–115 | 81–328 |
| Actives per target (range) | 10–3,262 | 50–3,262 |

### Stratification

| Layer | Meaning | Full-set entries | Subset entries | Subset actives | Subset median actives |
|---|---|---|---|---|---|
| L1 | target seen · scaffold seen | 349 | 56 | 7,710 | 100 |
| L2 | target seen · scaffold novel | 488 | **178** | 59,783 | 172 |
| L3 | target novel · family seen | 53 | **19** | 3,533 | 103 |
| L4 | target novel · family novel | 254 | 75 | 18,664 | 152 |

⚠️ **L3 has only 19 targets**, the thinnest layer in the whole table; any
per-class breakdown involving L3 should not be reported.

### Protein class

| Class | Full-set share | Subset share | Change | Subset median actives |
|---|---|---|---|---|
| Kinases | 25.5% | **31.7%** | +6.2 pt | 164 |
| Other enzymes | 28.3% | 24.4% | −3.9 pt | 170 |
| GPCR | 9.9% | 12.5% | +2.6 pt | 101 |
| Proteases | 6.7% | 10.4% | +3.6 pt | 114 |
| Epigenetic | 6.3% | 8.8% | +2.5 pt | 159 |
| Unclassified | 13.0% | 6.1% | **−6.9 pt** | 108 |
| Ion channels | 5.2% | 2.7% | −2.5 pt | 351 |
| Nuclear receptors | 1.9% | 1.8% | −0.1 pt | 172 |
| Transporters | 2.0% | 0.9% | −1.1 pt | 109 |
| P450 | 1.0% | 0.6% | −0.4 pt | 85 |

### Structure source

| Source | Full set | Subset |
|---|---|---|
| A experimental structure | 77.7% | **83.5%** |
| B predicted · confidence sufficient | 8.0% | 7.3% |
| C predicted · confidence insufficient | 5.6% | 5.2% |
| C no confidence record | 8.7% | 4.0% |

### What the "≥50 actives" threshold filters out

Only **44.3%** of entries in the full set have ≥50 actives, so this threshold
alone removes more than half. The subset keeps 28.7% of entries, but keeps
**57.4%** of active molecules — what gets filtered out is the batch of targets
with sparse actives. There are two costs to this:

1. **Class composition is pushed toward the data-rich direction**: kinases
   +6.2 pt, "unclassified" −6.9 pt. Classes with few actives (ion channels,
   transporters, P450) are systematically weakened.
2. **Structure quality improves as a side effect**: experimental structures
   rise from 77.7% to 83.5%. Well-studied targets tend to have both more
   actives and a higher chance of a crystal structure — the two are
   correlated. So the subset's absolute metrics will run slightly higher than
   the full set's, and that is not the models getting stronger.

---

## 1. What the reference benchmark VSDS-vd is

Gu, Zhang, Shen et al., *Benchmarking AI-powered docking methods from the
perspective of virtual screening*, **Nature Machine Intelligence** 7(3):509–520
(2025), DOI `10.1038/s42256-025-00993-0` (Zhejiang University, Hou Tingjun /
Kang Yu group, the same group behind KarmaDock). Data at
<https://zenodo.org/records/13684010>.

Measured directly from the downloaded dataset by UniProt catalog (not quoted
from a secondhand description):

| Subset | Targets | Description |
|---|---|---|
| DTEBV-D (TrueDecoy) | **147** | actives paired with experimentally confirmed low-activity molecules |
| DRSM-D (RandomDecoy) | 68 | decoys drawn randomly from a commercial library |
| DLSCL-D | 8 | large-scale screening, paired with the TopscienceRefineSet library |

Median decoy:active ratio **40.5** (the paper states 1:40).

> **⚠️ A contradiction that must be recorded**: VSDS-vd's per-target active
> count has a **median of only 28, and only 24% of its targets reach 50 or
> more**. Filtering it by its own ≥50 rule leaves only 36 of the 147. So
> "actives ≥50" and "composition matched to VSDS-vd" are not mutually
> consistent in the data. Our own median after the ≥50 filter is 118 — 4×
> VSDS-vd's — so on this criterion we are stricter than the reference
> benchmark.

Class composition is relabeled using **exactly the same** ChEMBL
classification-tree criteria as `annotate_target_class3.py` — using the same
criteria on both sides is the only precondition under which this comparison is
meaningful.

---

## 2. How the subset was selected

`timesplit/build/select_vsds_matched.py --quota 350` → `results/T3_vsds_matched.csv`

**328 entries = 293 unique targets** (35 targets appear in more than one layer
at once, because L1/L2 are split by **ligand scaffold**, so the same target
can fall on both sides).

Stratification: **L1 56 · L2 178 · L3 19 · L4 75**

⚠️ **The "35 targets span multiple layers" fact has an operational
consequence**: any script that uses this CSV as a filter must key on
**(layer, target)**, not target alone. Filtering by target alone will also
pull in that target's records from other layers — this pitfall showed up in
the per-model analysis in §6, where `seen+unseen` came out to 417, 89 more
than the subset's own 328 entries.

⚠️ The balancing algorithm, class composition, and redundancy analysis
sections below were **written against the 250-quota version** (242 entries /
222 targets / L1 40 · L2 128 · L3 20 · L4 54). The methodology is unchanged,
but the specific numbers are from the old quota — the version finally adopted
is 350. See §0 for the current subset's full statistics.

### Balancing algorithm

**Iterative proportional fitting (IPF)** with caps: row margins = VSDS-vd's
class proportions, column margins = the candidate pool's layer proportions,
each cell capped at actual stock, with final rounding by largest remainder.
Within-cell sampling uses a fixed random seed — sampling by active count would
systematically bias toward well-studied, popular targets, and would further
widen the spread of per-target active counts (up to 3,262 in the pool).

**Two pitfalls hit along the way (both noted in the script comments)**:

1. Quota by class alone, filling L3/L4 preferentially within each class → all
   46 "other enzyme" slots were consumed by L4 alone, **this largest class
   dropped to 0 in L1/L2**, completely unbalancing composition across layers
2. After adding layer balancing, **L3 was squeezed down to 9** → L3's stock
   was only 20 to begin with, so this was changed to keep the entire layer and
   exclude it from proportional allocation

### Class composition comparison

| Class | VSDS-vd | This subset | Deviation |
|---|---|---|---|
| Kinases | 23.8% | 66 (27%) | +3.5 |
| Other enzymes | 18.4% | 51 (21%) | +2.7 |
| GPCR | 15.6% | 41 (17%) | +1.3 |
| Proteases | 15.6% | 34 (14%) | −1.6 |
| **Nuclear receptors** | 7.5% | **6 (2%)** | **−5.0** |
| Epigenetic | 6.8% | 19 (8%) | +1.0 |
| Ion channels | 2.0% | 6 (2%) | +0.4 |
| **P450** | 4.1% | **2 (1%)** | **−3.3** |
| Transporters | 1.4% | 3 (1%) | −0.1 |
| Unclassified | 4.8% | 14 (6%) | +1.0 |

Eight of the ten classes deviate by within ±3.5pp. **Nuclear receptors and
P450 fall short even taking every one available** (there are 6 and 2, against
17 and 9 needed to match proportion) — this is not a filtering-criteria issue:
these two families were characterized early and have few members, with almost
no new targets appearing after 2024-12, and **both L3+L4 hold zero of
either**. Recorded as a limitation.

### The scale is adjustable

| Quota | Actual | Deviation (excluding the two under-supplied classes) | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|---|
| 200 | 199 | 1.8pp | 32 | 104 | 20 | 43 |
| **250** | **242** | **3.5pp** | 40 | 128 | 20 | **54** |
| 300 | 284 | 6.1pp | 48 | 152 | 19 | 65 |
| 400 | 369 | 9.5pp | 63 | 201 | 19 | 86 |

Adjusted via `--quota`. **The version finally adopted is 350**: actual 328
entries, L1 56 · L2 178 · L3 19 · L4 75, with L4 rising from 54 at the 250
quota to 75 — an undersized L4 is the most expensive cost of this scheme (the
full ≥50 set has 116), so the larger quota was chosen.

---

## 3. Within-subset target redundancy: not removed

`timesplit/analysis/target_redundancy.py` → `results/T3_target_redundancy.csv`

(The section below was computed on the 250-quota version's 222 targets; the
conclusion does not change with the quota.)
All-to-all local alignment of the 222 targets (BLOSUM62, identity / min(len)),
24,531 pairs: median **2.9%**, **no pair ≥90%**, maximum 87.3%, only 9 pairs
(0.04%) ≥70%.

**Redundancy must be judged within-class** — cross-class pairs (kinase vs
GPCR) are near zero and pull the global median down:

| Class | Targets | Pairs | Median | Max | Pairs ≥40% |
|---|---|---|---|---|---|
| GPCR | 35 | 595 | **21.7%** | 66.8% | 16 |
| Nuclear receptors | 6 | 15 | **23.6%** | 37.1% | 0 |
| Kinases | 62 | 1,891 | 11.4% | 86.1% | 22 |
| Proteases | 28 | 378 | 2.8% | 77.9% | 6 |
| Other enzymes | 49 | 1,176 | 2.8% | 87.3% | 2 |
| *(cross-class)* | | 20,230 | *2.7%* | *33.2%* | *0* |

GPCR and nuclear receptors are highest (shared fold: 7-transmembrane /
ligand-binding domain), but **even the highest is far below the commonly used
40% redundancy-removal threshold**. cd-hit removes 0 at 90%, 8 at 70%, and 37
at 40%.

**Conclusion: not removed.** (The collaborator agreed on 2026-09-05.)

---

## 4. How much the metrics changed

`timesplit/analysis/score_subset.py` → `results/T3_main_vsds_subset.csv`

| | Full set (1,144 entries) | New subset, original layers | New subset, corrected layers |
|---|---|---|---|
| LigUnity-protein L4 | 8.83 | 12.38 | **9.38** |
| HypSeek L4 | 7.34 | 9.91 | **7.85** |
| EF decay range | −68 ~ −84% | −48 ~ −72% | −43 ~ −80% |

**L4 gets easier on the new subset**: class balancing removes a large number
of "unclassified" targets (41 → 4 within L4), and that batch happened to be
the hardest.

**Corrected layers** means reclassifying L4 targets that are ≥40% homologous
to the training set as L3 (see [T3 leakage audit](T3-leakage.md)). **Moving
just 6 targets (12% of 51) changes L4 by 20–25%, and restores the decay from
−69% to −78%** — L4 is thin enough that a handful of targets can swing the
headline conclusion.

### Main table (corrected layers)

| Model | L1 | L2 | L3 | L4 | EF decay |
|---|---|---|---|---|---|
| LigUnity-protein | 38.24 | 32.27 | 19.56 | **9.38** | **−78%** |
| LigUnity-pocket | 34.17 | 27.84 | 14.92 | 9.29 | −75% |
| HypSeek | 32.90 | 25.98 | 12.73 | 7.85 | −79% |
| LiTENCLIP | 32.46 | 22.68 | 9.21 | 9.20 | −74% |
| BindCLIP-hardneg | 18.67 | 12.66 | 8.60 | 5.44 | −75% |
| BindCLIP-randneg | 18.52 | 11.99 | 8.86 | 6.23 | −70% |
| DrugCLIP | 16.88 | 13.22 | 8.91 | 7.92 | −56% |
| ConGLUDe | 13.15 | 6.39 | 6.87 | 5.34 | −64% |
| ConPLex | 5.61 | 3.16 | 3.73 | 1.94 | −80% |
| SPRINT | 2.75 | 2.17 | 1.44 | 2.00 | −43% |

Target counts: L1 40 · L2 119 · L3 26 · L4 45. Decay is computed on the excess
over random.

---

## 5. What downstream analyses are affected

| Task | Affected? | Handling |
|---|---|---|
| **T1** standard benchmarks | **No** (uses DUD-E/DEKOIS/LIT-PCBA) | Untouched |
| **T2** ranking | Yes (the T3 column) | Recomputed, see `results/T2_on_T3_subset.csv` |
| **T5** three controls | Yes | Recomputed, see the T5 document |
| **T6** recall ceiling | Yes | Recomputed, L4 recall@50 goes from 17.5% → **9.5%** |
| **T6** docking rerank | **Yes, and it cannot be fixed by recomputing** | Only 5 of the 20 docked targets are in the new subset |

T6's docking scores were run for real (33 CPU-hours) — targets that were never
run simply have no score. Three options: ① state that this experiment was run
on the full L4 ② pick 20 new targets and re-run (~33 CPU-hours) ③ demote it to
an appendix. **Undecided.**

---

## Reproduction

```bash
python timesplit/build/select_vsds_matched.py --quota 250    # select subset
python timesplit/analysis/target_redundancy.py               # within-target redundancy
python timesplit/analysis/score_subset.py                    # T3 main table
python timesplit/analysis/score_t2_subset.py --subset results/T3_vsds_matched.csv
python t5_structure_source.py --models <ten models> --subset ...
python t5_threshold_curve.py --subset ...
python shortlist_recall.py --subset ...
```

`--subset` is provided by `timesplit/analysis/_subset.py`, and is shared by
all downstream scripts.
