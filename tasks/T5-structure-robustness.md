# T5 — Structure Robustness

**Question:** do the conclusions survive changing where the structure came from,
and how the pocket was cut?

**Status:** two controls complete, both informative. Apo/MD variants not done.

> 🔬 Physics collaborators: see "Where physics fits" at the bottom — the pocket
> definition result has direct consequences for any structure-based method.

---

## Why this task exists

Everything in T1/T3 depends on two upstream choices we made: which structure to
use, and how far from the ligand to cut the pocket. If the model rankings flip
when those change, the benchmark measures our preprocessing rather than the
models. Both were tested.

## Control 1 — structure source

Same targets, experimental holo structures against Boltz-2 predictions, on the
two layers where both sources occur
([`timesplit/analysis/t5_structure_source.py`](../timesplit/analysis/t5_structure_source.py),
[`results/T5_structure_source.csv`](../results/T5_structure_source.csv)).

⚠️ **An earlier version of this section reported only BindCLIP-hardneg and
BindCLIP-randneg and concluded "no significant difference anywhere". That was an
artifact**: the script iterated whatever models happened to be in
`results/t3/summary.json`, and that file is overwritten by each `score_t3.py`
run. With all ten models the picture changes. The script now requires an
explicit `--models` list and names anything missing
([`PATCHES.md`](../PATCHES.md)).

**L4** (n = 65–89 holo, 131–137 predicted):

| Model | holo | predicted | p |
|---|---|---|---|
| **ConGLUDe** | 6.40 ± 1.41 | **2.66 ± 0.59** | **0.0017** |
| **LigUnity-pocket** | 11.70 ± 1.77 | **6.29 ± 1.07** | **0.0086** |
| **SPRINT** | 1.90 ± 0.46 | **1.12 ± 0.27** | **0.0229** |
| **HypSeek** | 9.13 ± 1.65 | **6.20 ± 1.10** | **0.0295** |
| LiTENCLIP | 10.63 ± 1.68 | 7.08 ± 1.13 | 0.196 |
| BindCLIP-randneg | 6.11 ± 1.22 | 5.41 ± 0.99 | 0.284 |
| BindCLIP-hardneg | 6.34 ± 1.34 | 5.79 ± 0.97 | 0.324 |
| ConPLex *(sequence only)* | 1.89 ± 0.45 | 2.20 ± 0.52 | 0.660 |
| DrugCLIP | 6.73 ± 1.26 | 6.82 ± 1.05 | 0.882 |
| LigUnity-protein *(sequence only)* | 9.15 ± 1.67 | 8.64 ± 1.27 | 0.896 |

Nothing reaches significance at L3, where the predicted group is only 18 targets.

**What this supports, stated carefully.** Four models differ at L4 with p < 0.05,
all favouring experimental structures, and **8 of 10 point that way** (sign test
p = 0.109). But across all 20 comparisons **only ConGLUDe survives BH-FDR**
(threshold 0.0025 against its 0.0017). So this is a consistent direction with one
firmly established case, not four.

**The negative controls behave correctly, and that is what makes it interesting.**
Targets with crystal structures are not a random sample — well-studied targets
have them, and might simply be easier. If that were the whole story, the
sequence-only models would show the same gap, since they never see a structure.
Neither does: ConPLex p = 0.66, LigUnity-protein p = 0.90. **Target difficulty
does not explain the gap.**

**So the claim is now narrower.** Predicted structures are not a free
substitute: the largest measured cost is LigUnity-pocket losing 46% of its L4
EF1% (11.70 → 6.29). Where a model is weak to begin with the difference is
invisible, which is why DrugCLIP and BindCLIP show nothing. For a project
choosing between an AlphaFold model and waiting for a crystal, **the honest
summary is that predicted structures usually work but can cost up to half the
early enrichment, and which case you are in is not knowable in advance.**

## Control 2 — pocket cutoff

The models were trained with 6 Å pockets. We rebuilt the whole dataset at 4 Å
and 8 Å and re-ran three models. Full table:
[`results/T5_pocket_threshold.csv`](../results/T5_pocket_threshold.csv).

| Model | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| DrugCLIP 4 Å vs 6 Å | −34% | −44% | −42% | −50% |
| DrugCLIP 8 Å vs 6 Å | −62% | −47% | −65% | −52% |
| BindCLIP-randneg 4 Å / 8 Å | −32% / −72% | −46% / −60% | −62% / −75% | −38% / −64% |
| BindCLIP-hardneg 4 Å / 8 Å | −31% / −64% | −43% / −56% | −43% / −62% | −52% / −39% |

**6 Å wins 12 of 12 cells.** Both tightening and loosening hurt.

The finding is **not** "bigger pockets are better" but **"the cutoff must match
training"**. Practical consequence: deploying these models with a different
pocket convention can cost half the performance, and numbers from different
papers are hard to compare if their pocket conventions differ.

### Threshold choice was measured, not guessed

Sampling 120 targets first:

| Cutoff | Median atoms | Relative to 6 Å | Over the 511-atom model cap |
|---|---|---|---|
| 4 Å | 122 | 0.58× | 0% |
| 5 Å | 164 | 0.77× | 0% |
| **6 Å (training)** | 212 | 1.00× | 0% |
| 8 Å | 391 | 1.85× | 10.8% |
| 10 Å | 562 | 2.65× | **68.3%** |

5 Å was the original plan and was dropped: at 0.77× it is too close to 6 Å for a
null result to mean anything. 10 Å was excluded because two-thirds of pockets
would be truncated — that would measure truncation, not pocket size.

### Truncation artefact ruled out

8 Å has 10.8% of pockets over the model's 511-atom cap, where the code keeps a
**random** center-weighted sample. So part of the 8 Å drop could be truncation
rather than pocket size. Splitting the 8 Å results:

| Group | n | 6 Å | 8 Å | decay |
|---|---|---|---|---|
| Under cap (full pocket) | 2,796 | 12.86 | 5.18 | **−59.7%** |
| Over cap (randomly truncated) | 336 | 12.57 | 5.18 | −58.8% |

Nearly identical. The 8 Å degradation is a real pocket-scale effect. (Also, 4 Å
never hits the cap and still drops 31–62%, so that direction is clean by
construction.)

**Side note relevant to other groups:** the cap matters enormously depending on
configuration. At our setting (511 cap, 6 Å) only 1 pocket in 1,904 is
truncated. At 256 cap and 7 Å — the configuration a collaborator reported — the
median pocket is 289 atoms and **70.8%** get truncated, which is why truncation
strategy is a live issue there and a non-issue here.

## Control 3 — apo conformation

The first two controls both used **holo** pockets: cut from a complex, with side
chains already arranged around a ligand. Every public screening benchmark does
the same. A real campaign often starts from an **apo** structure, where nothing
has moved out of the way. If models degrade there, every benchmark — ours
included — overstates what these methods deliver in practice.

**Design.** For each target, superpose an apo entry onto the holo one by backbone
CA, then cut the 6 Å pocket in the superposed apo **using the holo ligand's
coordinates**. Both pockets therefore sit at the same place and the only
difference is conformation. (Finding a pocket independently in the apo structure
would measure pocket detection instead, which is a different question.)
Alignment is rejected below 30 matched residues or above 5 Å RMSD.

**Result** — 45 targets, paired
([`../timesplit/analysis/t5_apo_compare.py`](../timesplit/analysis/t5_apo_compare.py),
per-target [`../results/T5_apo.csv`](../results/T5_apo.csv)):

| Model | EF1% holo → apo | BEDROC | AUROC holo → apo |
|---|---|---|---|
| DrugCLIP | 7.60 → 6.45 (−15%, p=0.53) | −20% (p=0.064) | 0.704 → 0.627 (**−11%, p=0.0007**) |
| BindCLIP-randneg | 7.88 → 8.40 (+7%, p=0.88) | +3% (p=0.85) | 0.650 → 0.622 (−4%, p=0.32) |

**What holds and what does not.** DrugCLIP loses global ranking quality on apo
pockets and that loss is significant. BindCLIP's change is not distinguishable
from noise. Early-enrichment metrics (EF1%) are too noisy at 45 targets to
support any claim in either direction — their p-values are 0.53 and 0.88. So the
honest summary is: **apo conformation measurably hurts one of two models on
AUROC, and the benchmark cannot resolve the effect on the metric practitioners
actually use.**

⚠️ **How different are these apo structures, really?** Median side-chain
deviation in the pocket is 1.08 Å, with 56% of targets above 1 Å (75th pct
1.78 Å, 90th 2.61 Å). That is mild-to-moderate induced fit, **not** a collapsed
pocket. The result should be read as "models are somewhat sensitive to modest
conformational change" and cannot be extended to hard apo cases.

**A bug worth recording, because the first version of this table was wrong.**
The initial run reported apo being *better* (EF1 +8.8% / +35%). The check that
caught it was asking, before interpreting anything, how far the apo pockets
actually sat from the holo ones: side-chain deviation had a 90th percentile of
**29.8 Å**. Conformational change cannot be 30 Å — the pockets were in different
places. Cause: for homo-oligomers the ligand appears in several copies tens of
ångströms apart, and the apo extraction took the first copy while the holo
pipeline had used `pick_copy` (most contacts within the target's own chains).
Anchoring the copy choice to the existing holo pocket's centroid brought the
90th percentile to 2.61 Å. The sanity script
([`../timesplit/analysis/t5_apo_sanity.py`](../timesplit/analysis/t5_apo_sanity.py))
is now part of the procedure rather than a one-off.

**Coverage.** 486 of 631 targets with a holo structure also have an apo entry;
110 of those are in the eval set; 45 survive alignment, co-location and
minimum-pocket-size filters. Targets are L3/L4 only, since the holo pipeline only
ever needed structures for new targets.

## Not done

- MD-sampled conformers
- Top-k overlap / pocket RMSD as stability metrics

These were in the original plan and remain open.

## Where physics fits 🔬

❌ T5 is not a place to *add* a physics method — it is a check on whether the
retrieval-model conclusions are stable, not a model comparison.

✅ But **the pocket-cutoff result applies directly to physics workflows**: any
structure-based scoring depends on the same pocket definition, and we now have
the same 1,044 targets prepared at 4/5/6/8 Å with validated residue-level
extraction. If a docking or FEP setup needs a pocket definition, these are
already built and the sensitivity is quantified.

## Code

| What | File |
|---|---|
| Build pockets at 4 / 5 / 6 / 8 Å | [`timesplit/structure/extract_pocket.py`](../timesplit/structure/extract_pocket.py) (predicted), [`extract_pocket_pdb.py`](../timesplit/structure/extract_pocket_pdb.py) (experimental, chain-aware) |
| Re-run the models at the other thresholds | [`timesplit/runners/run_t3_thr.sh`](../timesplit/runners/run_t3_thr.sh), [`run_t3_unimol_5a.sh`](../timesplit/runners/run_t3_unimol_5a.sh) |
| Threshold curve: pocket sizes, cap-overflow rates, and the 4/6/8 Å table | [`timesplit/analysis/t5_threshold_curve.py`](../timesplit/analysis/t5_threshold_curve.py) |
| Truncation-artefact control (over-cap vs under-cap at 8 Å) | [`timesplit/analysis/t5_cap_stratify.py`](../timesplit/analysis/t5_cap_stratify.py) |
| Experimental vs predicted structure comparison | [`timesplit/analysis/t5_structure_source.py`](../timesplit/analysis/t5_structure_source.py) |
| Structure-confidence grading used to define "high quality" | [`timesplit/analysis/t3_target_quality.py`](../timesplit/analysis/t3_target_quality.py) |

Each script's docstring states **how to read a null result** before showing any
numbers — written that way deliberately, so the interpretation was fixed before
the output existed.
