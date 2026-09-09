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
| **T1** Enrichment | Do published enrichment numbers reproduce on standard benchmarks? | ✅ complete — 10 models × 3 benchmarks | [T1](tasks/T1-enrichment.md) |
| **T2** Affinity ranking | Can these models rank binding strength, not just separate binders from non-binders? | ✅ answered — **weakly, and it decays with novelty**; the CASF/T3 gap is explained | [T2](tasks/T2-affinity-ranking.md) 🔬 |
| **T3** Time-split | Do they generalise to targets that appeared after training? | ✅ main result, 10 models × 4 layers | [T3](tasks/T3-time-split.md) |
| **T3 v2** Dataset revision | ≥50 actives, class composition matched to VSDS-vd | ✅ 242 entries / 222 targets; every downstream analysis re-aggregated | [T3 v2](tasks/T3-dataset-v2.md) |
| **T5** Target swap | Do the models use the protein at all, or only the ligand? | ✅ 10 models × L1/L4 × 3 rounds — **AUROC collapses to 0.500** | [Target swap](tasks/T5-target-swap.md) |
| **T3** Leakage audit | Is the benchmark actually solvable without the protein? Are the "novel" targets novel? | ✅ three diagnostics — **L1 is largely a memorisation test, L3/L4 are clean** | [Leakage](tasks/T3-leakage.md) |
| **T4** Target fishing | Run retrieval backwards: molecule → target | not started (deprioritised) | [T4](tasks/T4-target-fishing.md) |
| **T5** Structure robustness | Do the conclusions survive changing structure source, pocket definition, and apo conformation? | ✅ three controls done | [T5](tasks/T5-structure-robustness.md) 🔬 |
| **T6** Physics complementarity | Can physics methods supply the ranking ability retrieval lacks? | ✅ ranking: yes (ρ 0.615 vs 0.40). ❌ cascade rerank: no benefit; two independent physics methods, one significant | [T6](tasks/T6-physics.md) 🔬 |

🔬 = has a **"where physics fits"** section with concrete entry points.

**New to virtual screening?** [`WALKTHROUGH-zh.md`](WALKTHROUGH-zh.md) (Chinese)
explains the whole project from first principles — what virtual screening is, how
these models work, what each task did, and what the results mean — with no
assumed background. It is the recommended entry point for readers who are not
already working in this area.

**Reviewing rather than running?** Three pages carry the caveats:

| | |
|---|---|
| [`LIMITATIONS.md`](LIMITATIONS.md) | every known reason a number here could be wrong, ordered by how much it moves the headline claims |
| [`PATCHES.md`](PATCHES.md) | what had to be changed in third-party code, and the bugs in **our own** code — including two that produced a wrong conclusion before being caught |
| [`MODELS.md`](MODELS.md) | exact checkpoints, why each variant, interface quirks |

## Headline findings

1. **All ten models lose 68–84% of their above-random enrichment on post-cutoff
   targets.** Absolute performance differs fifteenfold between the best and worst
   model; the *decay* is nearly identical. This is a property of the method
   class, not of any one model.
   ⚠️ Decay is measured on the excess over a random ranking — EF@1% has a floor
   at 1.0, so a raw ratio understates the loss for weak models. Under the raw
   ratio the range is 46–80%, with SPRINT the lone outlier at 46% purely because
   its L1 is 2.52; on the excess it sits at 76%, mid-pack. AUROC decay in the
   same tables already used this convention.
   → [T3](tasks/T3-time-split.md)

2. **A ligand-only baseline reaches 98.7% of the theoretical ceiling — at every
   layer.** Scoring each candidate purely by its 2D fingerprint similarity to the
   target's other actives (no protein at all) gives EF@1% ≈ 50.4 against a ceiling
   of 51.0, at L1 *and* L4. Actives of one target are largely one congeneric
   series; cross-target decoys are not. The baseline sees the target's actives and
   the models do not, so it is an **upper bound**, not an indictment — but it
   reframes the decay: the ceiling is flat across layers while models fall from 38
   to 9, so **what the models lose on novel targets is access to a memorisable
   chemical series**, not chemistry ability. Normalised against that ceiling, the
   best model extracts **76% of the available signal at L1 and 19% at L4**.

   The matching *lower* bound closes the argument. A classifier that sees only
   ECFP4 and never learns which target it is scoring (GroupKFold by uniprot —
   grouping by molecule would split a congeneric series across the fold boundary)
   lands **below random at every layer**: mean AUROC 0.513 / 0.428 / 0.442 / 0.379
   for L1–L4, and EF@1% of exactly 0 on 259 of 328 targets. So every bit of T3's
   signal comes from ligand-to-known-ligand similarity for that target, and none
   from drug-likeness of the molecule alone — which is what cross-target real
   actives as decoys were chosen to guarantee.
   → [Leakage audit](tasks/T3-leakage.md)

3. **On genuinely novel chemistry against a novel target, the best model
   enriches 4.5-fold — not the 9.4 the main table reports.** Splitting each
   layer's actives by their Tanimoto distance to the training set and computing
   enrichment per tier: at L4, LigUnity-protein reaches **27.1 on chemistry it
   has seen (≥0.7) and 4.5 on chemistry it has not (<0.35)**. DrugCLIP falls
   *below random* (0.6) on the novel tier at L3. The two novelties also
   compound: changing the target alone (L1→L4, familiar chemistry) costs 1.9×,
   changing the chemistry alone (within L4) costs 6.0× — and the same chemistry
   change costs only 2.0× at L1. **Ligand novelty hurts more than target
   novelty, and hurts far more once the target is novel too.**
   → [Leakage audit](tasks/T3-leakage.md)

4. **Swapping the target to an unrelated protein collapses every model to
   chance.** Holding the candidate pool fixed — same molecules, same labels, same
   order — and replacing only the target identity drops EF@1% by **83–99%** and
   lands AUROC at **0.471–0.522, median 0.500** across eight models at L4. Not
   degraded: indistinguishable from a coin flip. This is what rules out the
   reading that finding 2 invites — the models are *not* ignoring the protein.
   The dependence also scales with capability: the strongest model loses 99.3%
   of its enrichment, while SPRINT, whose AUROC barely clears chance to begin
   with (0.589/0.558), is the only model that does not significantly degrade
   (p = 0.077). **Weak models are not more robust; they were never using the
   target.** → [Target swap](tasks/T5-target-swap.md)

5. **Ligand-side leakage is entirely confined to L1.** 32.1% of L1 actives are
   exact InChIKey matches to a training ligand (decoy background: 3.7%), and their
   median Tanimoto to the training set is 0.727 with 53.9% above 0.7. L2/L3/L4
   actives are *more* novel than the decoys (median 0.37–0.42 vs 0.419). Models
   also preferentially retrieve the familiar: HypSeek's top-1% actives at L1 have
   a median similarity of **0.969** to training ligands.
   → [Leakage audit](tasks/T3-leakage.md)

6. **The L3/L4 split had a fall-through bug: 24% of L4 targets were never
   checked.** Family membership came from a precomputed CD-HIT file, and a target
   *absent* from that file fell through to L4 — "not found" was treated as "no
   homologous family". A sensitive mmseqs search against the training set finds
   ≥70% homologues for 10% of the supposedly novel targets, including 100%
   full-length cross-species orthologs. Relabelling at the 40% threshold moves 30
   targets and restores the decay from −69% to −78%.
   → [Leakage audit](tasks/T3-leakage.md)

7. **Affinity ranking is weak but real, and it decays like enrichment does.**
   Per-target Spearman on post-cutoff data runs +0.09 to +0.26 at L1 and falls to
   +0.02 to +0.10 at L4; on congeneric FEP benchmarks it is ≈ +0.4, and on the
   14 targets shared by both the two are statistically indistinguishable
   (+0.41 vs +0.29, p = 0.27). The checkpoint selected upstream *for ranking*
   (HypSeek `_rk`) leads every layer. ⚠️ An earlier version of this README
   reported ranking as **zero** on T3 — that was a molecule-ordering bug in our
   analysis code, documented in [`PATCHES.md`](PATCHES.md).
   → [T2](tasks/T2-affinity-ranking.md)

   **The T3-vs-CASF gap is our filter, not the models.** The same models score
   ρ ≈ 0.42–0.55 within CASF targets. T3's `pAff ≥ 6` cut halves the within-target
   affinity spread (SD 0.783 vs 1.576), and correcting for that restriction of
   range recovers 75–91% of the difference. Report the observed T3 number, but
   do not read it as these models ranking worse on post-cutoff data than on CASF.

8. **Model ranking reverses by target class.** Sequence-only models win on
   kinases; geometry-aware models win on other enzymes. Reporting only the
   overall mean is misleading. → [T3](tasks/T3-time-split.md)

9. **Mildly sensitive to structure source, extremely sensitive to pocket
   definition.** Moving the pocket cutoff off the 6 Å the models were trained on
   costs 31–75%, with 6 Å winning 12 of 12 cells. Swapping experimental
   structures for Boltz-2 predictions costs less but not nothing: **8 of 10
   models drop at L4** (sign test p = 0.109), four individually significant and
   ConGLUDe surviving BH-FDR across all 20 comparisons, with LigUnity-pocket
   losing 46% (11.70 → 6.29). The two sequence-only models — which never see a
   structure — show no gap, so target difficulty does not explain it.
   ⚠️ An earlier version of this finding said predicted structures substitute
   with *no* significant difference. That rested on two models, selected by
   accident rather than design ([`PATCHES.md`](PATCHES.md)).
   → [T5](tasks/T5-structure-robustness.md)

10. **A co-folding model ranks affinity well, but reranking a retrieval
   shortlist with it does not help.** Three runs, two layers: on known targets
   the retrieval score is informative inside its own top-50 (AUROC 0.806) and
   Boltz-2 reranking *degrades* it (0.720); on novel targets the retrieval score
   is at chance (0.446) and Boltz-2 is slightly better (0.523) but cannot move
   the top of the list. Rank fusion never beat the better arm. Whether retrieval
   scores are usable inside their own shortlist turns out to depend on target
   familiarity — which is why testing one layer misled us.
   **A second, unrelated physics method reproduces this, and there it reaches
   significance**: smina docking of a top-200 shortlist at L4 drops P@10 from
   0.411 to 0.167 (p = 0.031, n = 9) — worse than the retrieval order it was
   handed. Two methods sharing no code, no scoring idea and no shortlist depth,
   same direction. **Nor is it a pose-quality artifact**: raising Boltz-2's
   structure sampling from 1 to 5 (best-of-5 by confidence, strictly paired over
   749 complexes) moves AUROC by 0.002, every p-value above 0.9. Structure
   quality is not the limiting factor. → [T6](tasks/T6-physics.md)

11. **A co-folding model ranks affinity where retrieval cannot.** On the 16 FEP
   systems, same ligands and same metric, Boltz-2 reaches Spearman +0.615
   (Kendall τ 0.474, against a published free-energy method's 0.503) while the
   retrieval models sit at +0.28 to +0.40. It is not a clean sweep — retrieval
   wins on 5 of 16 systems, mostly the ones where the physics reference also
   loses — which is what makes the two families worth combining rather than
   ranking. → [T6](tasks/T6-physics.md)

12. **Sequence and pocket trade places by benchmark — neither representation
   wins consistently.** LigUnity ships a pocket branch and a sequence branch from
   one release — same training set, same ligand encoder, same checkpoint scheme.
   Paired per target, the sequence branch wins DEKOIS and T3's L1/L2 (60–69% of
   decided targets, p ≤ 0.02), the pocket branch wins **LIT-PCBA on every
   early-enrichment metric** (10 of 10 decided targets on EF1%, p = 0.005), and
   DUD-E is a tie under pairing despite a 5.89 EF1% gap in the means. On novel
   targets (L4) the two are indistinguishable — 48–50%, p ≥ 0.53. This is
   finding 3 one level down: not only does model ranking reverse by benchmark,
   so does the ranking of two branches of one model.
   → [T3](tasks/T3-time-split.md)

13. **Training data explains performance tiers better than architecture.** The
   models trained on PocketAffDB all land at L1 EF1% 32–39; the three on
   DrugCLIP's data all land at 17–19 — across differences in retrieval
   augmentation and molecular encoder. The same split holds on DUD-E, where the
   four PocketAffDB models take the top four places. Visible directly in
   [`figures/`](figures/) fig 1 and fig 2, which are coloured by training set
   rather than architecture. → [T3](tasks/T3-time-split.md)
   *Finding 18 sharpens this: only one of the two training sets confers a
   target-specific advantage.*

14. **A checkpoint selected for affinity ranking is also the better screener.**
   HypSeek ships two weights from one run: `_vs` selected on screening, `_rk` on
   FEP ranking. Both are now public (the author released them in
   [issue #4](https://github.com/jianhuiwemi/HypSeek/issues/4)), and `_rk` beats
   `_vs` on **all seven screening measurements** — DUD-E 56.39 vs 51.41,
   DEKOIS 28.83 vs 25.52, LIT-PCBA 8.34 vs 6.82, and every T3 layer
   (L1 36.63 vs 32.11, L4 7.34 vs 7.07). An earlier version of this finding
   compared `_rk` against a *self-trained* `_vs` with a known training defect;
   the official weight makes the comparison sound.
   → [`results/T1_hypseek_official.md`](results/T1_hypseek_official.md)

   A collaborator's paper-faithful `_vs` reproduction also trails it at
   L1→L4 decay is indifferent to which weight is used (−50% vs −54%).
   ⚠️ An earlier version of this finding said retraining the screening weight
   makes a model *worse*; that was a deficit in our own training (a contrastive
   negative pool of 4 against the official 24) and is retracted.
   → [`MODELS_TRAINING.md`](MODELS_TRAINING.md)

15. **Three independent groups cannot reproduce HypSeek's published weight from
    its published recipe.** Following the paper, this project's `_vs` lands 16%
    below the reported DUD-E EF@1% (43.29 vs 51.44), a collaborator's 4% below
    (49.34), and a third party in
    [issue #4](https://github.com/jianhuiwemi/HypSeek/issues/4) 10% below
    (46.05) — the last having verified identical training inputs and swept
    random seeds. The author's own released weight reproduces the paper exactly
    (we measure 51.41), so the gap is in the recipe, not the evaluation. Ours has
    a diagnosed cause (contrastive negative pool of 4 against the official 24);
    the other two do not.
    ⚠️ An earlier version of this finding claimed the released weight was not
    the published model, on the grounds that we measured 56.39 against a
    published 51.44. That compared `_rk` to the paper's `_vs` number — different
    checkpoints. **Retracted.**
    → [`results/T1_hypseek_official.md`](results/T1_hypseek_official.md)

16. **The official evaluation runs with the protein-sequence pathway switched
    off, contradicting the paper.** `alpha_prot` defaults to 1 in training but
    `test_task.py` reads it as `getattr(self.args, "alpha_prot", 0)` and argparse
    never exposed the name, so every published evaluation number was produced
    with that pathway disabled — while the paper states that removing it costs
    performance. Turning it on raises DUD-E EF@1% from 51.41 to 53.02, but
    *lowers* LIT-PCBA from 6.82 to 5.21. The pathway helps on synthetic decoys
    and hurts on experimentally confirmed ones, which is the same axis finding 3
    is about. → [`results/T1_hypseek_official.md`](results/T1_hypseek_official.md)

17. **One model's headline DUD-E number is 74% target leakage, and the standard
    benchmark cannot show it.** Contrastive training on DUD-E decoys is
    ConPLex's method rather than an ablation — `contrastive: True` is the shipped
    default — and it draws negatives from 40 of the 102 targets our T1 scores.
    Dividing out each target's difficulty by the other nine models' median gives
    a clean monotone gradient for ConPLex alone: r = 1.034 on the targets it
    trained on, 0.615 on targets its split files name but hold out, 0.100 on the
    45 they never mention (Kruskal–Wallis p = 1e-5). The other nine models are
    flat. It is not BindingDB overlap: the effect survives on the 14 training
    targets BindingDB never contained (5.09×, p = 0.001) and vanishes on the 27
    BindingDB saw that no split names (0.72×, p = 0.71). Rescored on the 45
    untouched targets, **ConPLex falls from EF1% 18.70 to 4.83 and AUROC 0.683 to
    0.577** while every other model moves between −10% and +7%. T3 is unaffected
    — its training set covers 19.4% of T3 targets, the same band as the rest.
    → [`tasks/T1-enrichment.md`](tasks/T1-enrichment.md#conplex-trained-on-dud-e)

18. **Only one of the two training sets confers a target-specific advantage, and
    the layering is not what produced the decay.** Re-cutting L1–L4 per model
    using each model's *own* training set separates the two families further,
    not less: 51–58% decay for the DrugCLIP-set models against 68–72% for the
    PocketAffDB models, where the shared labels had them overlapping at 45–69%
    and 69–72%. Every model then scores higher on targets it trained on — but
    ranking the ten models *within each target*, which cancels target difficulty,
    splits them cleanly. The four PocketAffDB models gain 0.9–1.8 rank places on
    their own targets; the three DrugCLIP-set models gain **−0.10 to −0.13**,
    i.e. nothing. A four-cell crossover confirms it: on targets only PocketAffDB
    contains, all four of its models rank better and all six other models rank
    worse, with no exception (group difference 2.41 places, exact p = 0.0048) —
    and the DrugCLIP-set models' value (+0.90) is indistinguishable from that of
    models that never saw those targets (+1.03). **PocketAffDB membership is
    worth ~2.4 rank places; `train_no_test_af` membership is worth nothing
    measurable.** Caveat: the A-only cell holds 13 targets, so the per-model
    significance is weak and the evidence is at group level.
    → [`tasks/T3-leakage.md`](tasks/T3-leakage.md)

## Repository layout

**Task numbers appear only in `tasks/`.** Code directories are named after what
they operate on, so nothing pretends to be "task N" — a task document's Code
table is what maps a claim to the script behind it. This matters because several
tasks own no code of their own: T2 re-scores arrays that T1 and T3 already
produced, and T5 is the T3 pipeline re-run at other pocket cutoffs.

```
WALKTHROUGH-zh.md   从零理解整个项目（中文）— 无需背景知识的完整导览

MODELS_TRAINING.md  ⚠️ our attempt to train HypSeek's screening weight: two runs
               that never updated a parameter, then two that did — and what the
               19–23% shortfall against the released weight does and does not show

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
├── T1_main.csv                 10 models × 3 standard benchmarks × 4 metrics
├── T3_main.csv                 10 models × 4 layers × 5 metrics
├── T3_main_clean.csv           the same, with training-set contamination removed
├── T3_main_ci.csv              bootstrap confidence intervals
├── T3_targets.csv              per-target detail (class, layer, structure source)
├── T3_actives_gradient.csv     results at actives floors of 10/20/30/50, four metrics
├── T3_seq_vs_pocket_per_target.csv  the sequence/pocket pair on the time split
├── T1_seq_vs_pocket_per_target.csv  the same pair on the three standard benchmarks
├── T2_on_T3.csv                affinity ranking on time-split data
├── T2_on_FEP.csv               affinity ranking on the 16 FEP systems
├── T2_range_restriction.csv    why CASF and T3 disagree — spread, not models
├── T5_apo.csv                  apo vs holo pockets
├── T5_structure_source.csv     experimental vs predicted structures, all ten models
├── T5_pocket_threshold.csv     4 / 6 / 8 Å comparison
├── T6_FEP_boltz.csv            Boltz-2 affinity on the FEP systems
├── T6_rerank{,2,3}.csv         three cascade-rerank runs
├── T6_rerank4.csv              the same shortlist at 5 diffusion samples
└── T6_dock.csv                 smina docking rerank, with per-target coverage
figures/       four summary figures, and the scripts that rebuild them
```

Which task each directory serves:

| Directory | Feeds |
|---|---|
| `standard/` | T1, and the T2 scores on standard benchmarks |
| `timesplit/` | T3 (build + run + analyse), T5 (same pipeline, other cutoffs), T2 (re-scores its outputs), T4 (would reuse them) |
| `physics/` | T2 (FEP benchmark) and T6 (Boltz-2, physics comparison) |
| `eval/` | every task — one metric implementation for all of them |
| `figures/` | T1, T3, T6 — regenerated from `results/`, never hand-edited |

## Models evaluated

Nine retrieval models plus Boltz-2 as the physics arm, all with official code and
official weights. Which exact checkpoint, why that variant, and the interface
quirks of each: **[`MODELS.md`](MODELS.md)**.

| Model | Protein side | Training data |
|---|---|---|
| DrugCLIP, BindCLIP-randneg, BindCLIP-hardneg | 3D pocket | DrugCLIP set (4,098 UniProt) |
| LigUnity-pocket / -protein, LiTENCLIP, HypSeek | 3D pocket / sequence / hyperbolic | PocketAffDB (2,196 UniProt) |
| ConGLUDe | sequence + structure graph | own |
| ConPLex | sequence only — **negative control on T3 only** ⚠️ | BindingDB + DUD-E contrastive |
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
