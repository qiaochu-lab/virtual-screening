# T6 — Physics Complementarity

**Question:** retrieval models enrich but cannot rank binding strength. Can a
physics-based method supply the missing half, and is retrieval → physics in
series better than either alone?

**Status:** premise established, one physics signal measured, head-to-head run
in flight.

> 🔬 **This is the task for physics collaborators.** T2 sets up the problem; T6
> is where physics methods enter as first-class competitors.

---

## The premise is established, not assumed

The motivating claim — retrieval models cannot rank affinity — was tested hard
before building anything on it. Two obvious rescues were ruled out:

| Rescue | Test | Result |
|---|---|---|
| "It's the training objective" | HypSeek `_rk`, a checkpoint selected on a FEP validation set specifically for ranking | Spearman **+0.028** on T3 — same as the screening-optimised weight |
| "It's the geometry" | HypSeek uses hyperbolic rather than Euclidean embedding space | Best screening AUROC of all nine models (0.923 L1 / 0.710 L4), ranking still zero |

Changing the objective doesn't fix it. Changing the geometry doesn't fix it.
That is what turns "we could try physics" into "there is a specific gap physics
might fill".

**Refined premise** (see [T2](T2-affinity-ranking.md)): the models rank *within*
a chemical series (ρ ≈ 0.4 on FEP benchmarks) and not at all *across* series
(ρ ≈ 0 on T3). So T6's question sharpens to: **is physics better across series,
where retrieval fails completely?**

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

Running now: **Boltz-2 per-ligand on the same 16 FEP systems, 461 complexes**,
3 shards on GPUs 4/6/7. Roughly a third complete. When it lands, this table
becomes valid — same systems, same ligands, same metric:

| Method | Within-target ρ, 16 FEP systems |
|---|---|
| LigUnity-protein | +0.396 (16/16 correct sign) |
| LigUnity-pocket | +0.392 |
| LiTENCLIP | +0.276 |
| **Boltz-2** | **pending** |
| FEP+ | literature r ≈ 0.6–0.8 |

Preparation detail: systems exceeding Boltz-2's 1170-residue limit are truncated
to the kinase/binding domain (`fep_truncate.py`), which keeps the site intact.

## Three ways to combine, and which is worth doing

| Mode | How | Cost |
|---|---|---|
| physics only | physics scores and ranks everything | high — per-ligand |
| **cascade rerank** | retrieval takes top-N → physics reorders | **low — only top-N** |
| rank fusion | weighted merge of both rankings | trivial — scores already on disk |

**Cascade rerank is the one with practical value**, because it is what a real
screening campaign does (cheap filter, expensive confirm) and its cost scales
with N rather than library size. Rank fusion is nearly free and worth reporting
as a baseline.

**Open gap:** cascade rerank on T3 needs a physics score for each target's top-N
ligands — 1,044 targets × top-20 ≈ 20,000 complexes. Boltz-2 is too expensive at
that scale. **Docking (Vina / smina / GNINA) is the realistic choice**, and the
pockets are already extracted at four thresholds, so a collaborator can start
from prepared inputs rather than from PDB files.

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
