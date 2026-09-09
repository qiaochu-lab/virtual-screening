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

0. **Read in compounds rather than percentages, a novel-target campaign still
   beats random tenfold.** Ordering the top 100 molecules buys **59 real actives
   on a familiar target and 20 on a novel one** (LigUnity-protein), against a base
   rate near 2%. Switching the cutoff from "top 1% of the pool" to "top 100,
   whatever the pool" moves every model's decay by at most 8 percentage points
   and reorders nothing, so the decay below is not an artefact of the metric.
   → [`results/T3_recall_at_k.csv`](results/T3_recall_at_k.csv)

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

2. **A chemical-series oracle reaches 98.7% of the theoretical ceiling — at every
   layer, and 100% on DUD-E.** Scoring each candidate purely by its 2D fingerprint similarity to the
   target's other actives (no protein at all) gives EF@1% ≈ 50.4 against a ceiling
   of 51.0, at L1 *and* L4. Actives of one target are largely one congeneric
   series; cross-target decoys are not.

   ⚠️ **This is not a "ligand-only baseline", and calling it one would overstate
   it.** It is conditioned on the target — it reads that target's known actives,
   which no model here is given. The accurate name is a **target-conditioned
   ligand-similarity oracle**, and what it measures is a **ceiling**: how far
   pure chemical similarity could go if you already knew what binds.

   **The same oracle on the conventional benchmarks: DUD-E 100.0% of ceiling,
   DEKOIS 95.5%, LIT-PCBA 39.8%.** So this is not a quirk of our dataset — every
   benchmark here except LIT-PCBA falls to a protein-blind similarity lookup. The
   mechanism differs: DUD-E's decoys are topology-mismatched by protocol
   (actives→actives 0.65–0.76 vs decoys→actives ~0.17), T3's actives are
   congeneric series (0.78–0.89 vs ~0.16). ⚠️ **T3's gap is the largest of the
   three**, so this is a limitation of our benchmark too. LIT-PCBA is the only one
   where the two overlap (0.29–0.42 vs 0.19–0.26) — and the only one where every
   model collapses to near-random. Those are the same fact.
   → [`tasks/T1-enrichment.md`](tasks/T1-enrichment.md#what-these-three-benchmarks-are-made-of)

   It reframes the decay: the ceiling is flat across layers while models fall from 38
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
   *below random* (0.6) on the novel tier at L3. **Ligand novelty costs more than
   target novelty** — changing the target alone (L1→L4, familiar chemistry) costs
   1.9×, changing the chemistry alone costs 4–6× — and re-cutting the same
   analysis per model, with each model's own training set deciding which targets
   are novel, puts the realistic cell (novel target, novel chemistry) at 3.8–7.9
   against 53.1 for the easy one: a 12-fold span within one model and one metric.
   The two novelties are approximately **independent, not compounding**: the
   novel/familiar ratio has median 0.26 on seen targets and 0.26 on unseen. An
   earlier version of this finding claimed they amplified each other; that came
   from LigUnity-protein alone (3.1× amplification, off an 18-target cell) and
   three of the other four models run the other way. **Corrected.**
   → [Leakage audit](tasks/T3-leakage.md)

4. **Swapping the target to an unrelated protein collapses every model to
   chance.** Holding the candidate pool fixed — same molecules, same labels, same
   order — and replacing only the target identity lands AUROC at **0.475–0.527
   at L4** and **0.471–0.570 at L1**, across all ten models, three substitute
   draws each. Not degraded: indistinguishable from a coin flip. Every one of
   the seven pocket models drops by 19–42% with p < 2e-4. This is what rules out
   the reading that finding 2 invites — the models are *not* ignoring the
   protein. The dependence scales with capability: HypSeek falls 0.918 → 0.570
   at L1, while SPRINT, whose correct-target AUROC barely clears chance to begin
   with (0.584/0.516), does not significantly degrade at L1 at all (p = 0.22),
   and ConPLex does not at L4 (p = 0.28). **Weak models are not more robust;
   they were never using the target.**
   → [Target swap](tasks/T5-target-swap.md),
   [`results/T3_target_swap.csv`](results/T3_target_swap.csv)

   ⚠️ The collapse is to chance, not *below* it, and two L1 cells sit visibly
   above: HypSeek 0.570 and SPRINT 0.567. Something survives the swap at L1 that
   is not target identity — most likely the ligand-side signal that finding 5
   localises there.

4b. **What they use is family membership, not target identity.** Repeating the
   swap with a *homologous* substitute — same CD-HIT-40% cluster, same candidate
   pool, verified zero self-pairs (HDAC3→HDAC1, carbonic anhydrase I→II,
   JAK2→JAK3) — costs essentially nothing. Against the 19–42% AUROC collapse and
   93.5–99.3% EF collapse an unrelated protein causes, a homologue moves AUROC by
   **−4.1% to +9.8%** and EF@1% by **−18% to +12%**, and **not one of the ten
   models degrades significantly**. HypSeek's L1 AUROC goes 0.918 → 0.570 on an
   unrelated target and 0.918 → 0.907 on a relative.

   So the protein signal is real — finding 4 stands — but **its resolution stops
   at the family**. For screening that means a model transfers within a family
   that already has known ligands, and there is no evidence here that it can tell
   two members of one family apart. ⚠️ L4 has only **7** substitute pairs, which
   is structural rather than a sampling shortfall (L4 targets are by definition
   from families the training set never saw), so this rests on L1's 32–34 pairs.
   → [Target swap](tasks/T5-target-swap.md),
   [`results/T3_target_swap_family.csv`](results/T3_target_swap_family.csv)

5. **Ligand-side leakage is confined to L1 — and only for the four models
   trained on the affinity half.** Against that half's 428,767 ligands, 32.1% of
   L1 actives are exact InChIKey matches to a training ligand (decoy background:
   3.7%), median Tanimoto 0.727 with 53.9% above 0.7, while L2/L3/L4 actives are
   *more* novel than the decoys (median 0.37–0.42 vs 0.419). Those four models
   also preferentially retrieve the familiar: HypSeek's top-1% actives at L1 have
   median similarity **0.969**.

   **Measured against the structure half's own 13,590 ligands the same L1 actives
   look completely different — median 0.371, only 9.0% above 0.7, and 42.2%
   outright novel.** So "L1 is close to a memorisation test" is a statement about
   the affinity-trained models, and it is part of why the two groups differ
   roughly twofold at L1: half of L1 is chemistry one group has seen and the
   other has not. The preference itself survives the correction for the
   structure-only models, but weakly — DrugCLIP's retrieved L1 actives sit at
   16.1% above 0.7 against a 9.0% pool, a 1.8× lift that is gone by L4.
   → [Leakage audit](tasks/T3-leakage.md)

6. **The L3/L4 split had a fall-through bug: 24% of L4 targets were never
   checked.** Family membership came from a precomputed CD-HIT file, and a target
   *absent* from that file fell through to L4 — "not found" was treated as "no
   homologous family". A sensitive mmseqs search against the training set finds
   ≥70% homologues for 10% of the supposedly novel targets, including 100%
   full-length cross-species orthologs. Relabelling at the 40% threshold moves 30
   targets and restores the decay from −69% to −78%.
   → [Leakage audit](tasks/T3-leakage.md)

7. **Affinity ranking is weak but real — and it is carried entirely by
   familiar chemistry.** Per-target Spearman on post-cutoff data runs +0.09 to
   +0.26 at L1 and falls to +0.02 to +0.10 at L4; on congeneric FEP benchmarks it
   is ≈ +0.4, and on the 14 targets shared by both the two are statistically
   indistinguishable (+0.41 vs +0.29, p = 0.27). The checkpoint selected upstream
   *for ranking* (HypSeek `_rk`) leads every layer.

   **Splitting each target's actives by ligand novelty changes what that decay
   means.** Paired within target, familiar chemistry (max Tanimoto to training
   ≥ 0.5) against novel (< 0.5): at L1 the three strongest models score +0.16 to
   +0.28 on the familiar half and **+0.01 to +0.08 on the novel half**
   (p = 0.0001–0.0013, surviving BH-FDR over 14 tests). At L4 **every** model has
   p > 0.19 — the effect is gone because the familiar half has fallen to meet the
   novel one. The novel half does not decay from L1 to L4 (+0.01…+0.08 →
   +0.05…+0.09): **it is already at floor at L1**. So the layer-wise decay in T2
   is the familiar-chemistry half falling, not ranking ability degrading with
   target novelty. Confounder checked — L1's familiar half also has a 35% wider
   affinity spread, and the effect survives the Thorndike correction.
   ⚠️ An earlier version of this README reported ranking as **zero** on T3 — that
   was a molecule-ordering bug in our analysis code, documented in
   [`PATCHES.md`](PATCHES.md).
   → [T2](tasks/T2-affinity-ranking.md)

   **Half of CASF-2016 is in these models' training files, by construction.**
   148 of 285 CASF PDB IDs (51.9%) appear verbatim in the PocketAffDB training
   labels — identity, not similarity. The training code drops DUD-E, LIT-PCBA and
   DEKOIS targets from training and **does not drop CASF**, and the PDBbind half
   of the labels is unfiltered in the released weight. So of the four benchmarks
   these weights are evaluated on, CASF is the one where the test proteins were
   seen. Its measurable cost is narrower than the overlap suggests: splitting
   CASF into fully-contaminated and fully-clean targets and correcting for the
   clean targets' 35% narrower affinity spread, only **HypSeek `_rk`** keeps a
   gap outside the bootstrap interval (+0.800 → +0.399 corrected) — which is
   awkward precisely because it is the model leading that table.
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

   **Finding 18 sharpens what "training data" means here.** The two sets are not
   rivals — the four leading models trained on *exactly* the same 16,744 PDB
   structures as the other three, plus an affinity-labelled half covering 2,196
   targets. So the claim is narrower than "more data wins": **the same
   structures, plus the affinity half, take L1 EF@1% from 17–19 to 32–39.**

   ⚠️ **What the affinity half adds is not only labels.** Its 428,767 unique
   ligands against the structure half's **13,590** — a **31.6×** larger chemical
   space — arrive together with the pAff values. How much of the gap is the
   labels and how much is the ligand diversity cannot be separated from these
   two training sets alone; it would take retraining on one with the other held
   fixed. The claim as stated is about the *half*, not about labels per se.

   The correct reference set also matters for reading ligand novelty. Scored
   against its *own* 13,590 training ligands rather than the affinity half's,
   **72.6% of T3's molecules are novel chemistry to DrugCLIP and only 0.9% reach
   the "very close" tier** (against 23.6% and 8.8% under the affinity half). But
   the familiar-chemistry effect survives that correction: at L1, DrugCLIP still
   enriches **9.7 on its novel tier against 25.4 on its familiar one**, and the
   BindCLIP pair 11.5→30.9 and 11.9→26.8 — a 2.3–2.7× gap where the wrong
   reference had shown 5.2×. **Smaller, but the same shape as the four
   affinity-trained models.**
   → [`tasks/T3-leakage.md`](tasks/T3-leakage.md)

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

18. **The two "different training sets" are the same structures plus affinity
    labels, and the labelled half is where the target-specific advantage lives.**
    LigUnity-family training reads *two* label files
    (`train_task.py:523-524`): a structure half (`train_label_pdbbind_seq.json`,
    16,744 PDB entries) and an affinity half (`train_label_blend_seq_full.json`,
    2,196 UniProt with pAff). **The structure half is byte-for-byte the same PDB
    set as DrugCLIP's `train_no_test_af`** — 16,744 on each side, intersection
    16,744, neither exclusive. So the seven pocket models are not trained on
    rival corpora: four of them trained on what the other three saw, *plus*
    affinity labels for 2,196 targets.

    Re-cutting L1–L4 per model against the corrected union still separates the
    two families, slightly more than before: **51–58% decay for the three
    structure-only models against 69–74% for the four with affinity labels**,
    where the shared L1→L4 labels had them overlapping. Ranking the ten models
    *within each target*, which cancels target difficulty, the four gain
    **+0.70 to +1.86 rank places** on targets in their own training set; the
    three structure-only models gain **−0.10 to −0.13**, i.e. nothing — though
    that null is weak evidence, since all seven trained on those same targets
    and none can stand out. Splitting by which half a target came from, the four
    rank **1.54 places better** on targets only the affinity half contains
    (exact permutation p = 0.0095). Repeated on all 635 common targets rather
    than the 350-quota subset — which grows the thin structure-only cell from 7
    targets to 28 — the gap is 1.61 places at the **identical p = 0.0095**, and
    all four models fall on the same side, so the one exception in the subset
    (HypSeek, +0.30) does not survive the larger sample.
    → [`tasks/T3-leakage.md`](tasks/T3-leakage.md)

    ⚠️ **This finding replaces an earlier version that was wrong.** It read
    "PocketAffDB membership is worth ~2.4 rank places; `train_no_test_af`
    membership is worth nothing measurable", from a crossover that assumed the
    two training sets were disjoint. What is nested is not the two label files
    (they share only 817 UniProt) but what each *group of models* saw. The
    measurements were real; the attribution was not. Both the corrected numbers
    and their full-sample replication were independently reproduced by a second
    agent along a separate code path before this was rewritten.

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
| DrugCLIP, BindCLIP-randneg, BindCLIP-hardneg | 3D pocket | 16,744 PDB structures |
| LigUnity-pocket / -protein, LiTENCLIP, HypSeek | 3D pocket / sequence / hyperbolic | the **same** 16,744 + pAff for 2,196 UniProt |
| ConGLUDe | sequence + structure graph | own |
| ConPLex | sequence only — **negative control on T3 only** ⚠️ | BindingDB + DUD-E contrastive |
| SPRINT | SaProt structure-aware sequence | own |
| Boltz-2 | co-folding + affinity head | own |

The seven pocket-family models share **one** structure corpus; four of them
additionally get affinity labels, and that difference predicts the performance
tiers better than architecture does. Detail in [`MODELS.md`](MODELS.md).

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
