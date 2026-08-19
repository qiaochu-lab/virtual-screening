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
| **T1** Enrichment | Do published enrichment numbers reproduce on standard benchmarks? | 3 of 9 models | [T1](tasks/T1-enrichment.md) |
| **T2** Affinity ranking | Can these models rank binding strength, not just separate binders from non-binders? | ✅ answered — **no, across chemical series** | [T2](tasks/T2-affinity-ranking.md) 🔬 |
| **T3** Time-split | Do they generalise to targets that appeared after training? | ✅ main result, 9 models × 4 layers | [T3](tasks/T3-time-split.md) |
| **T4** Target fishing | Run retrieval backwards: molecule → target | not started (deprioritised) | [T4](tasks/T4-target-fishing.md) |
| **T5** Structure robustness | Do the conclusions survive changing structure source and pocket definition? | ✅ two controls done | [T5](tasks/T5-structure-robustness.md) 🔬 |
| **T6** Physics complementarity | Can physics methods supply the ranking ability retrieval lacks? | premise established, head-to-head running | [T6](tasks/T6-physics.md) 🔬 |

🔬 = has a **"where physics fits"** section with concrete entry points.

## Headline findings

1. **All nine models lose 64–77% of EF1% on post-cutoff targets.** Absolute
   performance differs fivefold between the best and worst model; the *decay* is
   nearly identical. This is a property of the method class, not of any one
   model. → [T3](tasks/T3-time-split.md)

2. **Ranking ability depends on chemical-series composition, not target
   familiarity.** On FEP benchmarks (congeneric analogues) Spearman ≈ +0.4; on
   T3 (cross-database chemistry) ≈ 0. A paired test on the *same 14 targets*
   gives +0.413 vs −0.004, p = 0.0001 — which rules out "the model knows these
   targets" and points at the data. The models learn local SAR, not absolute
   binding strength. → [T2](tasks/T2-affinity-ranking.md)

3. **Model ranking reverses by target class.** Sequence-only models win on
   kinases; geometry-aware models win on other enzymes. Reporting only the
   overall mean is misleading. → [T3](tasks/T3-time-split.md)

4. **Insensitive to structure source, extremely sensitive to pocket
   definition.** Predicted structures substitute for experimental ones with no
   significant difference (p = 0.28–0.74); moving the pocket cutoff from 6 Å to
   4 Å or 8 Å costs 31–75%, with 6 Å winning 12 of 12 cells. → [T5](tasks/T5-structure-robustness.md)

5. **Training data explains performance tiers better than architecture.** The
   three models trained on LigUnity's data all land at L1 EF1% 32–39; the three
   on DrugCLIP's data all land at 17–19 — across differences in retrieval
   augmentation and molecular encoder. → [T3](tasks/T3-time-split.md)

## Repository layout

```
tasks/                  ⭐ one document per task — the place to start
├── T1-enrichment.md         standard benchmarks (DUD-E / LIT-PCBA / DEKOIS)
├── T2-affinity-ranking.md   🔬 affinity ranking + where physics enters
├── T3-time-split.md         main result: post-cutoff generalization
├── T4-target-fishing.md     planned, not started
├── T5-structure-robustness.md 🔬 structure-source and pocket-cutoff controls
├── T6-physics.md            🔬 physics complementarity — the collaboration task
└── scripts/                 the analysis scripts behind every number above

results/                machine-readable results
├── T3_main.csv              9 models × 4 layers × 5 metrics
├── T3_targets.csv           per-target detail (class, layer, structure source)
├── T2_on_T3.csv             affinity ranking on T3 data
├── T2_on_FEP.csv            affinity ranking on the 16 FEP systems
└── T5_pocket_threshold.csv  4 / 6 / 8 Å comparison

t3/                     dataset construction pipeline → t3/README.md
├── build/                   time split, difficulty layers, eval set
├── structure/               PDB metadata, chain mapping, pocket extraction
├── runners/                 per-model adapters
└── analysis/                stratified controls, robustness checks

eval/                   unified metric layer → eval/README.md
```

## Models evaluated

| Model | Representation | Training data | Status |
|---|---|---|---|
| DrugCLIP | pocket (3D) | DrugCLIP set | ✅ |
| BindCLIP-randneg / -hardneg | pocket (3D) | DrugCLIP set | ✅ |
| LigUnity-pocket / -protein | pocket / sequence | LigUnity set | ✅ |
| LiTENCLIP | pocket (3D) | LigUnity set | ✅ |
| HypSeek (`_rk`) | hyperbolic embedding | LigUnity set | ✅ |
| ConGLUDe | sequence + graph | own | ✅ |
| ConPLex | sequence | own | ✅ (negative control) |
| SPRINT | SaProt structure-aware sequence | own | ⚠️ L3/L4 only |
| Boltz-2 | co-folding + affinity head | own | physics arm |

## Reading the numbers

**T3 absolute values are not comparable to published values.** The decoy
construction deliberately differs from DUD-E's property-matched scheme — that
scheme is precisely the bias under examination. What *is* comparable is the
**decay from L1 to L4 within this fixed setup**, which is what T3 measures.

**Every model's per-molecule scores are on disk**, so any new metric or any new
scoring method can be dropped into the same comparison without re-running
anything.

## A note on paths

Scripts contain a hardcoded working directory (`B = "/data/yicheng/xqc/..."`)
from the machine they ran on. They are published as a record of what was
actually executed rather than as a turnkey package — change `B` at the top to
run them elsewhere. Every script's docstring states what the step is for and
what breaks without it, so the logic transfers even where the paths do not.

## License

MIT
