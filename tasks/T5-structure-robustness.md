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

Same targets, experimental holo structures vs Boltz-2 predictions.

| Model | Layer | n(holo) | holo | n(pred) | predicted | p |
|---|---|---|---|---|---|---|
| BindCLIP-hardneg | L3 | 31 | 9.00±2.41 | 18 | 6.00±1.83 | 0.62 |
| BindCLIP-hardneg | L4 | 89 | 6.34±1.34 | 137 | 5.79±0.97 | 0.32 |
| BindCLIP-randneg | L3 | 31 | 8.33±2.09 | 18 | 8.00±2.18 | 0.74 |
| BindCLIP-randneg | L4 | 89 | 6.11±1.22 | 137 | 5.41±0.99 | 0.28 |

**No significant difference anywhere.** Predicted structures are usable
substitutes for targets without crystals.

⚠️ This is **not a randomised comparison** — whether a target has an experimental
structure is itself non-random (well-studied targets have them). The differences
are small and consistently in the same direction, which supports the reading,
but it is not proof.

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

## Not done

- Apo structures (unbound conformations)
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
