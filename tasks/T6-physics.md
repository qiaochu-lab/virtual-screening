# T6 — Physics Complementarity

**Question:** retrieval models enrich but cannot rank binding strength. Can a
physics-based method supply the missing half, and is retrieval → physics in
series better than either alone?

**Status:** premise established, one physics signal measured, head-to-head run
in flight.

> 🔬 **This is the task for physics collaborators.** T2 sets up the problem; T6
> is where physics methods enter as first-class competitors.

---

## The premise, restated after a correction

This task was set up on the claim that retrieval models cannot rank affinity at
all. **That claim was wrong** — it came from a molecule-ordering bug in our T2
analysis code (see [`PATCHES.md`](../PATCHES.md)). Corrected, the models do rank,
weakly:

| | Spearman |
|---|---|
| Retrieval models on post-cutoff targets (L1 → L4) | +0.09…+0.26 → +0.02…+0.10 |
| Retrieval models on congeneric FEP series | +0.28…+0.40 |
| **Boltz-2 on the same FEP ligands** | **+0.615** |

Two things the correction changed and one it did not:

- **Changed:** "the training objective is not the problem" is now false — the
  checkpoint selected for ranking (HypSeek `_rk`) *is* the best ranker at every
  layer, +0.260 at L1 against DrugCLIP's +0.091.
- **Changed:** "ranking collapses across chemical series" is withdrawn. On the
  14 targets present in both datasets, T3 and FEP ranking are statistically
  indistinguishable (+0.29 vs +0.41, p = 0.27).
- **Unchanged:** the physics side is still clearly ahead. Boltz-2's +0.615 on the
  FEP set is roughly 1.6× the best retrieval model on the same ligands, and its
  Kendall τ of 0.474 sits next to a published free-energy method's 0.503.

So T6's question survives in a sharper form: **the gap is quantitative, not
categorical** — which makes "does physics rerank the top of a retrieval list
usefully?" the experiment that matters, rather than "can physics do something
retrieval fundamentally cannot?".

## Evidence so far — Boltz-2 has affinity signal

Boltz-2's affinity module ran alongside T3 structure prediction, giving 928
predictions. Cross-target correlation against measured pAffinity:

| Output | Spearman | Pearson | n |
|---|---|---|---|
| `affinity_pred_value` (sign-flipped) | **+0.404** (p=8e-38) | +0.484 | 928 |
| `affinity_probability_binary` | +0.274 | +0.338 | 928 |

Monotone across quintiles: 6.06 → 6.12 → 6.18 → 6.70 → **8.09** measured pAff.

⚠️ **Three limits on this number, stated up front:**

1. **It is cross-target, not within-target.** Only one representative ligand per
   target was predicted, so this cannot be compared like-for-like with the
   within-target Spearman values from T2. It answers a cheaper prior question —
   *is there any signal at all?* — well enough to justify the expensive run.
2. **Range-restricted.** The representative ligand is each target's *strongest*,
   so measured pAff is truncated from below. This depresses the correlation
   systematically; the true value is likely higher.
3. **Ligands over 128 atoms are unsupported** by the affinity module, so
   macrocycles and peptides are systematically absent.

## In flight — the head-to-head

The three numbers currently on the table were measured under three different
protocols and are **not** comparable:

| Method | Value | What it actually measured |
|---|---|---|
| LigUnity / LiTENCLIP | ρ = 0.28–0.40 | within-target, 16 FEP systems, congeneric series |
| nine retrieval models | ρ ≈ 0 | within-target, T3, cross-database chemistry |
| Boltz-2 | ρ = +0.404 | **cross-target** absolute affinity |
| FEP+ | r ≈ 0.6–0.8 | literature, FEP benchmarks |

**Done — 461/461 complexes.** Boltz-2 scored per ligand on all 16 systems, the
same ligands the retrieval models were scored on, the same metric:

| System | n | **Boltz-2 ρ** | LigUnity-pocket | LigUnity-protein | LiTENCLIP |
|---|---|---|---|---|---|
| mcl1 | 42 | **+0.885** | +0.750 | +0.799 | +0.724 |
| cmet | 24 | **+0.883** | −0.072 | +0.537 | +0.512 |
| ptp1b | 23 | **+0.823** | +0.372 | +0.182 | +0.026 |
| jnk1 | 21 | **+0.806** | +0.662 | +0.338 | −0.307 |
| tyk2 | 16 | **+0.806** | +0.462 | +0.382 | +0.406 |
| cdk8 | 32 | **+0.799** | +0.510 | +0.314 | +0.475 |
| cdk2 | 16 | **+0.782** | −0.276 | +0.296 | +0.506 |
| syk | 44 | **+0.778** | +0.422 | +0.292 | +0.301 |
| tnks2 | 27 | **+0.737** | +0.496 | +0.327 | +0.708 |
| thrombin | 11 | +0.691 | **+0.782** | +0.700 | +0.318 |
| p38 | 34 | **+0.503** | +0.194 | +0.129 | −0.252 |
| eg5 | 28 | **+0.494** | +0.286 | +0.396 | +0.452 |
| hif2a | 41 | +0.433 | **+0.473** | +0.415 | +0.370 |
| shp2 | 26 | +0.332 | **+0.735** | +0.620 | +0.549 |
| pfkfb3 | 40 | +0.168 | **+0.505** | +0.158 | +0.111 |
| bace | 36 | **−0.081** | −0.032 | +0.444 | −0.488 |
| **mean** | | **+0.615** | +0.392 | +0.396 | +0.276 |

Per-system CSV: [`results/T6_FEP_boltz.csv`](../results/T6_FEP_boltz.csv).

**Boltz-2 mean Spearman +0.615, median +0.757, correct direction on 15 of 16.**
In Kendall τ — the metric the published physics reference reports — it is
**+0.474 mean, +0.569 median, against Uni-FEP's 0.503**. A co-folding model with
an affinity head lands within noise of a free-energy method, at a fraction of the
cost per ligand.

**But it does not dominate.** LigUnity-pocket beats it on 5 of 16 systems,
LigUnity-protein on 3, LiTENCLIP on 1 — and the systems where retrieval wins
(SHP-2, PFKFB3, HIF-2α, thrombin, BACE) overlap with the ones where the
published physics reference also loses to retrieval. Two families with different
failure modes is exactly the premise T6 was set up to test, and it now has direct
evidence rather than an argument from three incomparable numbers.

⚠️ Read with three caveats: the FEP systems are congeneric series, so this says
nothing about cross-series ranking; Boltz-2's affinity head was trained on
public affinity data whose overlap with these classic systems is not
characterised; and BACE at −0.081 shows the failure is not graceful when it comes.

## Cascade rerank — Boltz-2, run four times, no effect measured any time

| Mode | How | Cost |
|---|---|---|
| physics only | physics scores and ranks everything | high — per-ligand |
| **cascade rerank** | retrieval takes top-N → physics reorders | **low — only top-N** |
| rank fusion | weighted merge of both rankings | trivial — scores already on disk |

**The ceiling nobody measured first.** Reranking can only reorder what the
shortlist already contains, so recall@N *is* the cascade's upper bound. Measured
after the fact ([`../physics/shortlist_recall.py`](../physics/shortlist_recall.py),
LigUnity-protein):

| Layer | median actives | **recall@50** | recall@200 | recall@500 |
|---|---|---|---|---|
| L1 | 24 | **64.1%** | 79.5% | 88.8% |
| L2 | 68 | 36.1% | 56.3% | 70.6% |
| L3 | 37 | 32.3% | 48.8% | 63.3% |
| **L4** | 47 | **17.5%** | 34.0% | 50.9% |

Both experiments ran on **L4 — the layer with the lowest ceiling**. A top-50
shortlist there holds about 8 of a median 47 actives; 82% were never eligible
for reranking. That is a design error, and it is the main reason to redo this on
L1/L2 rather than to conclude anything.

**Run 1** (20 targets chosen for having ≥15 actives → shortlists 27% active,
10 usable): P@5 0.420 → 0.440, P@10 0.410 → 0.350, AUROC 0.478 → 0.617.
Baseline too strong for the wrong reason; superseded.

**Run 2** (19 targets with 1–6 actives in the top-50 → 5.8% active, 15 usable,
937/941 complexes scored; per-target CSV
[`../results/T6_rerank2.csv`](../results/T6_rerank2.csv)):

| Ordering | P@5 | P@10 | mean rank of actives | AUROC in shortlist |
|---|---|---|---|---|
| retrieval (baseline) | **0.093** | **0.087** | 27.5 | 0.446 |
| Boltz-2 rerank | 0.053 | 0.067 | **24.1** | **0.523** |
| rank fusion | 0.080 | 0.047 | 26.3 | 0.472 |

Paired over targets: rerank vs baseline P@5 −0.040 (p=0.52), P@10 −0.020
(p=0.58), AUROC +0.077 (p=0.45). Restricting to ligands inside Boltz-2's
≤56-heavy-atom training range (14 targets) does not change the picture.

**What can and cannot be said.**

- **The retrieval score is uninformative inside its own top-50** — AUROC 0.446,
  at or below chance. It enriches across the library, then goes flat. This is a
  clean, reusable finding, and it is why reranking looked promising.
- **Boltz-2 does carry signal there** (0.523–0.617 across the two runs) and lifts
  the average active a few places.
- **Neither run improved the top of the list**, which is what a screening
  campaign acts on.
- **Nothing is significant**, and with 1–6 actives per shortlist, P@5 can only
  take values 0, 0.2, 0.4 — the measurement is coarse before it is anything
  else. "No effect measured" is not "no effect exists".
- **Rank fusion sits between the two arms rather than above them**, so on this
  data the two error modes are not complementary in the way T6 assumed.

**Run 3 — the L1/L2 contrast, and it inverts the picture.** 12 targets on
known proteins with crystal structures, same top-50, same three arms, 749/749
complexes scored ([`../results/T6_rerank3.csv`](../results/T6_rerank3.csv)):

| Ordering | P@5 | P@10 | mean rank | AUROC in shortlist |
|---|---|---|---|---|
| retrieval (baseline) | **0.533** | **0.333** | 11.8 | **0.806** |
| Boltz-2 rerank | 0.333 | 0.225 | 15.5 | 0.720 |
| rank fusion | 0.433 | 0.308 | **11.0** | **0.822** |

Side by side with run 2 (L4, novel targets, largely predicted structures):

| | retrieval AUROC in shortlist | Boltz-2 AUROC | rerank helps? |
|---|---|---|---|
| **L1/L2** (known targets, crystals) | **0.806** | 0.720 | **no — it degrades** |
| **L4** (novel targets, predicted) | 0.446 | 0.523 | no — AUROC up, top-k flat |

**What the contrast settles.** The original question was whether cascade rerank
fails as an idea or only under novel-target conditions. The answer is neither:

1. **Whether the retrieval score is informative inside its own shortlist depends
   on target familiarity** — 0.806 on known targets, 0.446 (chance) on novel
   ones. Reading the L4 number alone as a property of retrieval models, as an
   earlier version of this document did, was an artifact of testing one layer.
2. **Boltz-2's discrimination is comparatively stable** (0.720 vs 0.523) but
   loses to the baseline where the baseline is good, and cannot lift the top of
   the list where the baseline is bad.
3. **Rank fusion never beats the better arm** in any of the three runs. The two
   families' errors are not complementary on this data, which is the opposite of
   T6's premise.

So: **reranking a retrieval model's top-50 with Boltz-2 gave no benefit under
any condition we tested.** That is a stronger statement than run 2 supported,
because "novel targets are just hard" is now excluded. It remains a statement
about *this* pairing — one physics method, top-50 depth, this scoring — not
about physics rescoring in general; the caveats in the previous section stand,
and n = 12–15 targets keeps every p-value above 0.10.

## Is one diffusion sample enough? A direct test

Boltz-2's structure stage samples the complex from a diffusion process, and that
sampling is stochastic. Our runs set `--diffusion_samples 1`, so the affinity
model received a single unfiltered draw. A fair objection to the rerank result
is that the pose was simply bad and the affinity score inherited the noise.

Two details matter for reading that objection:

- **The affinity stage was never single-sample.** `--diffusion_samples_affinity`
  defaults to **5** and we never overrode it. The affinity model runs its own
  diffusion with 5 samples on top of whatever structure it is given.
- **Extra structure samples do not reach the affinity model as multiple poses.**
  The structure stage ranks its N samples by confidence and writes only the
  rank-0 one as `pre_affinity_<id>.npz`
  ([`boltz/data/write/writer.py:177`](https://github.com/jwohlwend/boltz)). So
  raising N buys a *best-of-N structure selection*, not multi-pose rescoring —
  Boltz-2 cannot be handed externally generated poses at all. Our run confirms
  the mechanism directly: 101 records produced 505 PDB files (5 each) and
  exactly 101 `pre_affinity` files.

**The test.** Same 750 complexes, same YAMLs, same MSAs, same affinity settings
— only `--diffusion_samples` changed from 1 to 5, giving a strictly paired
comparison over the 749 complexes scored in both runs
([`physics/run_rerank4.sh`](../physics/run_rerank4.sh),
[`physics/export_rerank4.py`](../physics/export_rerank4.py),
[`results/T6_rerank4.csv`](../results/T6_rerank4.csv)).

| Ranking | P@5 | P@10 | mean active rank | AUROC |
|---|---|---|---|---|
| retrieval (baseline) | **0.533** | **0.333** | **11.8** | 0.806 |
| Boltz-2, N=1 | 0.333 | 0.225 | 15.5 | 0.720 |
| Boltz-2, **N=5** | 0.317 | 0.233 | 15.7 | **0.718** |
| rank fusion, N=1 | 0.433 | 0.308 | 11.0 | **0.822** |
| rank fusion, N=5 | 0.467 | 0.292 | 11.4 | 0.815 |

Paired Wilcoxon over the 12 targets, N=1 against N=5:

| | P@5 | P@10 | mean rank | AUROC |
|---|---|---|---|---|
| N=5 − N=1 | −0.017 (p = 1.000) | +0.008 (p = 1.000) | +0.122 (p = 0.915) | −0.002 (p = 0.983) |

**AUROC moves by 0.002 and every p-value sits above 0.9.** Best-of-5 structure
selection changes nothing. Boltz-2 reranking is still worse than the retrieval
order it was given (P@5 0.317 vs 0.533, p = 0.070), and fusion still fails to
beat the better arm.

This makes the rerank conclusion stronger rather than weaker: **structure
sampling quality is not the limiting factor.** Whatever prevents the affinity
score from reordering a retrieval shortlist is not fixed by giving it a better
pose to score. It also lowers the value of the obvious follow-up — supplying the
binding site as a `pocket` constraint is another way to improve the input
structure, and this result predicts it would land in the same place.

## A second physics method, deliberately unrelated

The three Boltz-2 runs share one scoring engine, one shortlist depth, and one
notion of what "physics" means. A conclusion resting on that alone is a
conclusion about Boltz-2. So the rerank was repeated with **smina docking** —
an empirical scoring function with no learned affinity head, no co-folding, and
no shared code with Boltz-2 — and two of the design choices were changed at the
same time:

| | Boltz-2 runs | smina run |
|---|---|---|
| Shortlist depth | top-50 | **top-200** (recall@200 at L4 is 34.0% vs 17.5% at 50) |
| Method | co-folding + learned affinity head | empirical scoring function |
| Poses | co-folded | docked into the retrieval pocket |

**Complete: 20 of 20 targets docked.** Nine had enough actives inside the
top-200 to score.

| Ranking | P@10 | P@20 | mean active rank | AUROC |
|---|---|---|---|---|
| retrieval (baseline) | 0.411 | 0.439 | 59.9 | 0.755 |
| **smina rerank** | **0.167** | **0.172** | 81.6 | **0.527** |
| rank fusion | 0.367 | 0.350 | 64.9 | 0.687 |

| vs baseline | ΔP@10 | ΔP@20 | ΔAUROC |
|---|---|---|---|
| smina rerank | −0.244 (**p = 0.031**) | −0.267 (**p = 0.039**) | −0.228 (p = 0.055) |
| rank fusion | −0.044 (p = 0.25) | −0.089 (p = 0.14) | −0.068 (p = 0.20) |

Same direction as Boltz-2 and larger — and this is the first rerank run where
the degradation reaches significance. Docking ranks the shortlist **worse than
the retrieval order it was handed**, dropping it to near chance (AUROC 0.527),
and fusion again fails to beat the better arm.

Three things to hold against these numbers before quoting them:

- **n = 9, from 20 targets docked.** Eleven were lost because fewer than 2
  actives fell inside the top-200 — the recall ceiling again, not a docking
  failure. The significant p-values come from a Wilcoxon signed-rank test over
  those 9 pairs; they say the direction is consistent, not that the effect size
  is well estimated.
- **Four targets timed out at 90 minutes** (coverage 36–90%) and hold a
  non-random subset of ligands — the fast ones are the small ones. They are kept
  here because both arms are scored on the *same* subset, so the comparison
  stays internally valid even though the absolute level is biased;
  `results/T6_dock.csv` carries a `coverage` column and `score_dock.py` also
  prints the complete-only summary (n = 5, same direction, no significance).
- **Docking into a pocket slice is not full-protein docking.** The box comes
  from the extracted pocket's bounding box, which is the same pocket definition
  the retrieval models saw — a fair comparison, but not the docking setup a
  docking paper would use.

What the second method buys is not significance — it is that the earlier result
no longer depends on one scoring engine or one shortlist depth. Two methods
sharing nothing but the label "physics" degrade the same retrieval ordering in
the same direction.

## Falsifiability, agreed in advance

If physics methods also land near zero under identical conditions, that is a
result about **the difficulty of predicting binding strength from structure**,
not a failed experiment. Both outcomes get reported. This was fixed before the
runs started, specifically so the reporting decision could not be made after
seeing the numbers.

## Where physics fits 🔬 — concrete entry points

| Entry point | What exists already | What a collaborator adds |
|---|---|---|
| **A. Within-target ranking, FEP benchmarks** | 16 systems, 461 ligands, 3 retrieval baselines, Boltz-2 running | FEP/TI or a docking score on the same 461 → direct comparison |
| **B. Cascade rerank on T3** | 9 models' full score arrays, 1,044 targets, pockets at 4/5/6/8 Å | docking on top-N per target → does rerank beat retrieval alone? |
| **C. Enrichment subset (crosses into T1/T3)** | 20–30 high-quality-structure L4 targets, ~1,250 molecules each | full docking → *is physics also better at enrichment, or only at ranking?* |
| **D. Physics score as an extra tower** | unified metric layer accepts any per-molecule score array | any scoring function, in the same metrics |

Everything is a plain float array per target plus a label array; the metric layer
([`eval/`](../eval/)) computes EF / BEDROC / AUROC / Spearman / Kendall from
that. Nothing about it is model-specific.

### Five known traps

1. **Sign convention.** Boltz-2's `affinity_pred_value` is lower-is-stronger;
   pAffinity is higher-is-stronger. Correlations come out inverted if this is
   missed.
2. **Range restriction** in the existing 928 predictions (above) — a real effect
   on the number, not a model deficiency.
3. **128-atom ligand limit** in the affinity module.
4. **Pocket definition must match.** Physics and retrieval must use the same 6 Å
   pockets or the comparison is not a comparison. All four thresholds are built.
5. **Official FEP scoring zeroes negative R².** The reference implementation
   reports R² only and clamps it to 0 when `corr < 0`, which collapses
   "systematically backwards" and "no relationship" into the same number. We
   report signed Spearman alongside.

## Scripts

All of it lives in [`physics/`](../physics/):

| What | File |
|---|---|
| Cross-target Boltz-2 affinity correlation (the +0.404 above) | [`t6_boltz_affinity.py`](../physics/t6_boltz_affinity.py) |
| Build Boltz-2 inputs for the 461 FEP complexes | [`prep_boltz_fep.py`](../physics/prep_boltz_fep.py) |
| Truncate systems over the 1170-residue limit to the binding domain | [`fep_truncate.py`](../physics/fep_truncate.py) |
| Launch the per-ligand run (3 shards) | [`run_boltz_fep.sh`](../physics/run_boltz_fep.sh) |
| Run the retrieval models on the same systems | [`run_fep.sh`](../physics/run_fep.sh), [`patch_fep_save.py`](../physics/patch_fep_save.py) |
| Score, and recover scores from embeddings where needed | [`score_fep.py`](../physics/score_fep.py), [`fep_recover_preds.py`](../physics/fep_recover_preds.py) |
| Paired test isolating ligand composition from target familiarity | [`fep_vs_t3_same_targets.py`](../physics/fep_vs_t3_same_targets.py) |
| Compare against the published physics reference | [`fep_compare_physics.py`](../physics/fep_compare_physics.py) |
| Pockets a physics method would consume, at four thresholds | [`timesplit/structure/extract_pocket*.py`](../timesplit/structure/) |
| Metrics any new scoring method plugs into | [`eval/metrics.py`](../eval/metrics.py) |
| Recall ceiling of a shortlist, by layer and depth | [`shortlist_recall.py`](../physics/shortlist_recall.py) |
| Build docking inputs from a retrieval shortlist | [`prep_dock.py`](../physics/prep_dock.py) |
| Run smina, one target at a time, 90-minute cap | [`run_dock.sh`](../physics/run_dock.sh) |
| Score the docking rerank, with per-target coverage | [`score_dock.py`](../physics/score_dock.py) |
| The 5-sample rerun and its paired comparison | [`run_rerank4.sh`](../physics/run_rerank4.sh), [`chain_rerank4.sh`](../physics/chain_rerank4.sh), [`export_rerank4.py`](../physics/export_rerank4.py) |
