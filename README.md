# Virtual Screening Benchmark

A head-to-head benchmark of large-scale pocket–ligand retrieval models
(DrugCLIP-family), across six tasks, under one unified metric implementation.

Nine models are evaluated with their **own official code and official weights**.
Only the metric computation is shared, so differences in the tables are
attributable to the models.

**New here?** Start with [`tasks/`](tasks/) — one document per task, each stating
what question it asks, what data it uses, how it was run, and what came out.

---

## Task status

| Task | Question | Status | Doc |
|---|---|---|---|
| **T1** Enrichment | Do published enrichment numbers reproduce on standard benchmarks? | ✅ 9 models × DUD-E/DEKOIS; 2 cells queued | [T1](tasks/T1-enrichment.md) |
| **T2** Affinity ranking | Can these models rank binding strength, not just separate binders from non-binders? | ✅ answered — **weakly, and it decays with novelty** | [T2](tasks/T2-affinity-ranking.md) 🔬 |
| **T3** Time-split | Do they generalise to targets that appeared after training? | ✅ main result, 9 models × 4 layers | [T3](tasks/T3-time-split.md) |
| **T4** Target fishing | Run retrieval backwards: molecule → target | not started (deprioritised) | [T4](tasks/T4-target-fishing.md) |
| **T5** Structure robustness | Do the conclusions survive changing structure source and pocket definition? | ✅ two controls done | [T5](tasks/T5-structure-robustness.md) 🔬 |
| **T6** Physics complementarity | Can physics methods supply the ranking ability retrieval lacks? | ✅ ranking: yes (ρ 0.615 vs 0.40). ❌ cascade rerank: no benefit in any condition tested | [T6](tasks/T6-physics.md) 🔬 |

🔬 = has a **"where physics fits"** section with concrete entry points.

**Reviewing rather than running?** Three pages carry the caveats:

| | |
|---|---|
| [`LIMITATIONS.md`](LIMITATIONS.md) | every known reason a number here could be wrong, ordered by how much it moves the headline claims |
| [`PATCHES.md`](PATCHES.md) | what had to be changed in third-party code, and the bugs in **our own** code — including two that produced a wrong conclusion before being caught |
| [`MODELS.md`](MODELS.md) | exact checkpoints, why each variant, interface quirks |

## Headline findings

1. **All nine models lose 64–77% of EF1% on post-cutoff targets.** Absolute
   performance differs fivefold between the best and worst model; the *decay* is
   nearly identical. This is a property of the method class, not of any one
   model. → [T3](tasks/T3-time-split.md)

2. **Affinity ranking is weak but real, and it decays like enrichment does.**
   Per-target Spearman on post-cutoff data runs +0.09 to +0.26 at L1 and falls to
   +0.02 to +0.10 at L4; on congeneric FEP benchmarks it is ≈ +0.4, and on the
   14 targets shared by both the two are statistically indistinguishable
   (+0.41 vs +0.29, p = 0.27). The checkpoint selected upstream *for ranking*
   (HypSeek `_rk`) leads every layer. ⚠️ An earlier version of this README
   reported ranking as **zero** on T3 — that was a molecule-ordering bug in our
   analysis code, documented in [`PATCHES.md`](PATCHES.md).
   → [T2](tasks/T2-affinity-ranking.md)

3. **Model ranking reverses by target class.** Sequence-only models win on
   kinases; geometry-aware models win on other enzymes. Reporting only the
   overall mean is misleading. → [T3](tasks/T3-time-split.md)

4. **Insensitive to structure source, extremely sensitive to pocket
   definition.** Predicted structures substitute for experimental ones with no
   significant difference (p = 0.28–0.74); moving the pocket cutoff from 6 Å to
   4 Å or 8 Å costs 31–75%, with 6 Å winning 12 of 12 cells. → [T5](tasks/T5-structure-robustness.md)

5. **A co-folding model ranks affinity well, but reranking a retrieval
   shortlist with it does not help.** Three runs, two layers: on known targets
   the retrieval score is informative inside its own top-50 (AUROC 0.806) and
   Boltz-2 reranking *degrades* it (0.720); on novel targets the retrieval score
   is at chance (0.446) and Boltz-2 is slightly better (0.523) but cannot move
   the top of the list. Rank fusion never beat the better arm. Whether retrieval
   scores are usable inside their own shortlist turns out to depend on target
   familiarity — which is why testing one layer misled us. → [T6](tasks/T6-physics.md)

6. **A co-folding model ranks affinity where retrieval cannot.** On the 16 FEP
   systems, same ligands and same metric, Boltz-2 reaches Spearman +0.615
   (Kendall τ 0.474, against a published free-energy method's 0.503) while the
   retrieval models sit at +0.28 to +0.40. It is not a clean sweep — retrieval
   wins on 5 of 16 systems, mostly the ones where the physics reference also
   loses — which is what makes the two families worth combining rather than
   ranking. → [T6](tasks/T6-physics.md)

7. **Training data explains performance tiers better than architecture.** The
   three models trained on LigUnity's data all land at L1 EF1% 32–39; the three
   on DrugCLIP's data all land at 17–19 — across differences in retrieval
   augmentation and molecular encoder. → [T3](tasks/T3-time-split.md)

## Repository layout

**Task numbers appear only in `tasks/`.** Code directories are named after what
they operate on, so nothing pretends to be "task N" — a task document's Code
table is what maps a claim to the script behind it. This matters because several
tasks own no code of their own: T2 re-scores arrays that T1 and T3 already
produced, and T5 is the T3 pipeline re-run at other pocket cutoffs.

```
tasks/         ⭐ start here — one document per task
├── T1-enrichment.md            standard benchmarks (DUD-E / LIT-PCBA / DEKOIS)
├── T2-affinity-ranking.md      🔬 can they rank binding strength?
├── T3-time-split.md            main result: post-cutoff generalization
├── T4-target-fishing.md        planned, not started
├── T5-structure-robustness.md  🔬 structure-source and pocket-cutoff controls
└── T6-physics.md               🔬 physics complementarity — the collaboration task

standard/      DUD-E · LIT-PCBA · DEKOIS runs, and the training-similarity ablation
timesplit/     the self-built time-split benchmark — dataset and all model runs
├── build/         time split, difficulty layers, eval-set construction
├── structure/     PDB metadata, chain mapping, pocket extraction at 4/5/6/8 Å
├── runners/       per-model adapters and the patches each repo needed
└── analysis/      main table, class breakdown, robustness controls
physics/       🔬 FEP benchmark and Boltz-2 — the physics arm behind T2 and T6

eval/          unified metric layer (80 tests)
env/           per-model environment construction, with the version traps
results/       machine-readable CSVs
├── T1_main.csv              9 models × 3 standard benchmarks × 4 metrics
├── T3_main.csv              9 models × 4 layers × 5 metrics
├── T3_targets.csv           per-target detail (class, layer, structure source)
├── T2_on_T3.csv             affinity ranking on time-split data
├── T2_on_FEP.csv            affinity ranking on the 16 FEP systems
└── T5_pocket_threshold.csv  4 / 6 / 8 Å comparison
```

Which task each directory serves:

| Directory | Feeds |
|---|---|
| `standard/` | T1, and the T2 scores on standard benchmarks |
| `timesplit/` | T3 (build + run + analyse), T5 (same pipeline, other cutoffs), T2 (re-scores its outputs), T4 (would reuse them) |
| `physics/` | T2 (FEP benchmark) and T6 (Boltz-2, physics comparison) |
| `eval/` | every task — one metric implementation for all of them |

## Models evaluated

Nine retrieval models plus Boltz-2 as the physics arm, all with official code and
official weights. Which exact checkpoint, why that variant, and the interface
quirks of each: **[`MODELS.md`](MODELS.md)**.

| Model | Protein side | Training data |
|---|---|---|
| DrugCLIP, BindCLIP-randneg, BindCLIP-hardneg | 3D pocket | DrugCLIP set (4,098 UniProt) |
| LigUnity-pocket / -protein, LiTENCLIP, HypSeek | 3D pocket / sequence / hyperbolic | PocketAffDB (2,196 UniProt) |
| ConGLUDe | sequence + structure graph | own |
| ConPLex | sequence only — **negative control** | BindingDB |
| SPRINT | SaProt structure-aware sequence | own |
| Boltz-2 | co-folding + affinity head | own |

Only **two** distinct training sets cover the seven pocket-family models, and
that split predicts the performance tiers better than architecture does.

## Reading the numbers

**T3 absolute values are not comparable to published values.** The decoy
construction deliberately differs from DUD-E's property-matched scheme — that
scheme is precisely the bias under examination. What *is* comparable is the
**decay from L1 to L4 within this fixed setup**, which is what T3 measures.

**Every model's per-molecule scores are on disk**, so any new metric or any new
scoring method can be dropped into the same comparison without re-running
anything.

## A note on paths

Scripts contain a hardcoded working directory (`B = "/data/work/..."`)
from the machine they ran on. They are published as a record of what was
actually executed rather than as a turnkey package — change `B` at the top to
run them elsewhere. Every script's docstring states what the step is for and
what breaks without it, so the logic transfers even where the paths do not.

## License

MIT
