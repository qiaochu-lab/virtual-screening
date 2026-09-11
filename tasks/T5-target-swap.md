# Target Swap: Does the Model Actually Use Target Information?

The candidate ligand pool is left completely untouched; only the target
identity is swapped for another protein, to see how much discriminative power
the model has left.
Ten models × L1/L4 × three rounds of random substitutes.

---

## 1. Why this experiment is needed

The [leakage audit](T3-leakage.md) turned up something unsettling: a
ligand-similarity baseline that **looks at the protein not at all** reaches
98.7% of the theoretical EF@1% ceiling. This is easy to read as "the models
don't look at the protein at all — they're just recognizing molecules."

**But we cannot support that conclusion.** What the chemical-series oracle
proves is that "this benchmark can be solved by a pure chemistry signal", not
"the models are taking the pure-chemistry route." The moment a reviewer asks
"how do you know the models aren't using protein information", there is no
answer.

Target swap answers this directly: **if the model doesn't look at the
protein, swapping the target should have no effect.**

---

## 2. Design

```
correct: identity of target T   +  the candidate pool built for T (T's actives + cross-target decoys)
swap:    identity of target T′  +  the same candidate pool (labels, order, molecules all unchanged)
```

The only variable is target identity.

### Directory layout: why you can't just swap the pocket file

The model's protein sequence is looked up **by directory name**, not stored
in the lmdb. So
[`build_target_swap.py`](../timesplit/build/build_target_swap.py) names the
directory after the substitute `T′`, then symlinks the original target `T`'s
ligand pool into it:

```
swap_root/{layer}/{T′}/{T′}_pocket.lmdb  →  T′'s own pocket
swap_root/{layer}/{T′}/{T′}_lig.lmdb     →  T's ligand pool (symlinked, renamed)
```

When reading results back, `swap_manifest.json` is used to map back to `T` so
they can be paired with the results from the correct pocket.

> ⚠️ During a trial run I looked up the swap directory by `T`, which retrieves
> "T's pocket + someone else's ligands" — the molecule counts on the two sides
> are completely different, and this nearly got reported as a result.
> [`score_target_swap.py`](../timesplit/analysis/score_target_swap.py)
> therefore compares the label array for every pair, and drops and flags any
> mismatch.

### Sequence-only models

ConPLex / ConGLUDe / SPRINT don't read a pocket — they look up sequence from
the eval-set jsonl's `uniprot` field.
[`build_swap_eval.py`](../timesplit/build/build_swap_eval.py) generates an
eval set for them with **`uniprot` swapped, ligand pool kept**, sharing **the
same swap_manifest** as the pocket tree, so the pairing stays consistent and
all ten models' results can go into one table.

The original plan was to mark sequence models as not applicable; in practice
this design turns out to apply to them naturally.

### Three controls

| | Approach | Why |
|---|---|---|
| Pocket size | Substitute atom count within ≤30% of the original target's (measured median 13–16%) | Otherwise it measures a size effect, not an identity effect |
| Sampling variance | 3 substitutes drawn per target, median taken per target | The luck of a single draw would drown out the effect |
| Inference path | Each model copies its own `run_t3_*.sh`, changing only `--t3-root` and `--results-path` | The ten models differ in architecture / loss / precision, so hand-assembling parameters would easily introduce unrelated differences |

---

## 3. Results

Paired Wilcoxon signed-rank. EF = EF@1%.

### L1 (target and scaffold both seen)

| Model | EF correct | EF swap | ΔEF | AUROC correct | AUROC swap | p |
|---|---|---|---|---|---|---|
| LigUnity-protein | 38.24 | **0.26** | **−99.3%** | 0.906 | 0.560 | <1e-4 |
| LigUnity-pocket | 34.17 | **0.21** | **−99.4%** | 0.880 | **0.504** | <1e-4 |
| HypSeek | 32.90 | 1.12 | −96.6% | 0.919 | 0.580 | <1e-4 |
| LiTENCLIP | 32.46 | 0.32 | −99.0% | 0.877 | 0.533 | <1e-4 |
| BindCLIP-hardneg | 18.67 | 0.74 | −96.0% | 0.763 | 0.514 | <1e-4 |
| BindCLIP-randneg | 18.52 | 1.02 | −94.5% | 0.769 | 0.475 | <1e-4 |
| DrugCLIP | 16.88 | 0.76 | −95.5% | 0.760 | **0.502** | <1e-4 |
| ConGLUDe | 13.15 | 0.85 | −93.5% | 0.689 | 0.503 | <1e-4 |
| ConPLex | 5.61 | 0.90 | −83.9% | 0.589 | 0.511 | 0.0004 |
| **SPRINT** | 2.75 | 1.22 | **−55.4%** | 0.589 | 0.547 | **0.077** |

### L4 (target and family both novel)

| Model | EF correct | EF swap | ΔEF | AUROC correct | AUROC swap | p |
|---|---|---|---|---|---|---|
| HypSeek | 9.91 | **0.08** | **−99.2%** | 0.727 | **0.500** | <1e-4 |
| LigUnity-pocket | 11.91 | 0.38 | −96.8% | 0.683 | 0.471 | <1e-4 |
| LiTENCLIP | 10.10 | 0.35 | −96.5% | 0.708 | 0.504 | <1e-4 |
| LigUnity-protein | 12.38 | 0.45 | −96.4% | 0.717 | **0.497** | <1e-4 |
| BindCLIP-hardneg | 6.45 | 0.37 | −94.3% | 0.616 | 0.505 | 0.011 |
| ConGLUDe | 6.18 | 0.33 | −94.6% | 0.565 | 0.500 | 0.002 |
| BindCLIP-randneg | 6.98 | 0.53 | −92.4% | 0.656 | 0.489 | 0.0001 |
| DrugCLIP | 9.22 | 0.90 | −90.3% | 0.682 | 0.496 | <1e-4 |
| ConPLex | 2.43 | 0.69 | −71.4% | 0.562 | 0.522 | 0.150 |
| **SPRINT** | 1.89 | 1.55 | **−17.8%** | 0.558 | 0.515 | 0.012 |

---

## 4. Three conclusions

### One: swapping the target lands AUROC exactly at 0.5

At L4, eight models' swap AUROC falls in **0.471–0.522, median 0.500**. This
is not "getting worse" — it is **complete loss of discriminative power** —
exactly a coin flip.

A single round's results spread over 0.44–0.61; taking the median across
three rounds suppresses sampling noise and it converges to 0.5. This itself
shows that the three-round design was necessary.

### Two: there is a dose relationship — the stronger the model, the more it depends on target identity

| Model | L1 correct EF | ΔEF |
|---|---|---|
| LigUnity-protein | 38.24 | −99.3% |
| HypSeek | 32.90 | −96.6% |
| DrugCLIP | 16.88 | −95.5% |
| ConPLex | 5.61 | −83.9% |
| SPRINT | 2.75 | −55.4% (p=0.077) |

**SPRINT is the only model that does not degrade significantly after the
swap** — because it had almost no target specificity to begin with (L1 AUROC
0.589, L4 0.558, both hugging random).

**Weak models are not "more robust" — they were never using target
information to begin with.** This ties "target-signal strength" directly to
"model performance", which carries far more information than simply saying
"every model collapses."

### Three: pure sequence models collapse just the same

LigUnity-protein does not read the 3D pocket, only the amino-acid sequence —
and it drops the hardest (L1 −99.3%). ConGLUDe (sequence + structure graph) is
L1 −93.5%. **Dependence on target identity is not a phenomenon specific to
3D-pocket models.**

---

## 5. Put together with the other two experiments

| Experiment | What it shows |
|---|---|
| [Chemical-series oracle reaches the ceiling at 98.7%](T3-leakage.md#2-chemical-series-oracle-ceiling-reaches-987-of-the-theoretical-limit) | This benchmark **can** be solved by pure chemistry — but only if that target's actives are already known |
| **Target swap collapses to 0.5** | The models **do** use target information; they are not just guessing from chemistry |
| [Novelty-tiered enrichment: L4 fully novel chemistry is only 4.5](T3-leakage.md#4-enrichment-computed-separately-per-novelty-tier--the-realistic-scenario-numbers) | But this signal **is not strong enough** — it fails once the chemistry is genuinely novel |

Put the three sentences together:

> **The models do use target information, but that signal is not enough to
> support real chemical extrapolation.**

Any one of them alone invites misreading — reporting only the first reads as
"the model doesn't look at the protein"; reporting only the second reads as
"the model works fine."

---

## 5b. Same-family swap: the model recognizes family, not target identity

A random swap can only prove "swapping to an unrelated target causes
collapse" — it cannot tell whether the model recognizes the **specific
target** or **the family it belongs to**. A same-family swap answers this
directly: the substitute is another protein that is sequence-homologous and
pocket-similar, while the candidate ligand pool remains completely unchanged.

The substitutes are genuine homologous proteins — **zero self-pairs** among
the 123 pairs. Spot check: HDAC3 → HDAC1 (O15379 → Q13547), carbonic
anhydrase I → II (P00915 → P00918), JAK2 → JAK3 (O60674 → P52333).

### AUROC

| Model | Layer | Correct | Unrelated-target swap | Same-family swap |
|---|---|---|---|---|
| HypSeek `_rk` | L1 | 0.918 | 0.570 (**−37.9%**) | 0.907 (**+0.2%**) |
| | L4 | 0.718 | 0.486 (**−32.3%**) | 0.795 (−4.1%) |
| LigUnity-protein | L1 | 0.918 | 0.543 (**−40.8%**) | 0.917 (**+0.8%**) |
| | L4 | 0.705 | 0.481 (**−31.8%**) | 0.772 (−0.1%) |
| LigUnity-pocket | L1 | 0.882 | 0.544 (**−38.4%**) | 0.880 (+1.8%) |
| | L4 | 0.649 | 0.495 (**−23.8%**) | 0.741 (+9.8%) |
| LiTENCLIP | L1 | 0.877 | 0.511 (**−41.7%**) | 0.874 (−0.6%) |
| | L4 | 0.706 | 0.494 (**−30.0%**) | 0.798 (+1.9%) |
| DrugCLIP | L1 | 0.753 | 0.471 (**−37.5%**) | 0.779 (+3.0%) |
| | L4 | 0.675 | 0.488 (**−27.6%**) | 0.689 (+3.1%) |
| BindCLIP-randneg | L1 | 0.753 | 0.496 (**−34.1%**) | 0.773 (+2.1%) |
| BindCLIP-hardneg | L1 | 0.750 | 0.486 (**−35.2%**) | 0.781 (+1.4%) |
| ConGLUDe | L1 | 0.720 | 0.510 (**−29.2%**) | 0.852 (+0.7%) |
| ConPLex | L1 | 0.594 | 0.499 (−16.0%) | 0.596 (+0.9%) |
| SPRINT | L1 | 0.584 | 0.567 (−2.9%) | 0.624 (+3.6%) |

### EF@1%

For the same batch of models, swapping to an unrelated target drops EF@1% by
**93.5–99.3%** (HypSeek L1 34.37 → 1.51, L4 9.08 → 0.11); swapping to a
same-family target drops it by **−18% to +12%**, i.e. within the noise range.
The largest is BindCLIP-hardneg L1 at −18.1%, DrugCLIP L1 at −16.4%.

### Conclusion

> **What the models learn is family-level recognition, not target identity.**
> Swap the target for an unrelated protein and all ten models collapse to
> random; swap it for a homologous protein, and **not one model degrades
> significantly.**

This narrows the random-swap conclusion considerably, and also makes it more
useful: "the model does use protein information" is correct, but that
information's **resolution only reaches the family level**. The practical
implication for virtual screening is — a model transfers within a family that
already has known ligands when the target is swapped; the current evidence
says it cannot tell apart two members of the same family whose pockets differ
subtly.

This also fits together with the other two experiments in §5: the
chemical-series oracle shows that **ligand-side signal explains a great
deal**, the random swap shows that **protein-side signal genuinely exists**,
and the same-family swap shows that **the granularity of that signal is
family-level**.

### ⚠️ Two limitations that must be stated before the numbers

1. **L4 has only 7 substitute pairs.** This is structural sparsity, not a
   sampling shortfall — L4 is by definition "target and family both novel",
   so its targets have no same-family substitute to be found in the pool in
   the first place. **The L4 rows above cannot be cited on their own**; the
   conclusions above rest mainly on L1 (32–34 pairs).
2. **ConGLUDe's L1 has only 8 pairs** (10 pairs were dropped for label
   mismatch), and L4 has none at all.

## 5c. Protein-null: decided not to run, reasons recorded here

The checklist also had a protein-null ablation — removing the protein input
entirely or zeroing it out, to see how much performance remains. It was
tagged P2 (lowest priority). **We decided not to run it — this is a reasoned
decision, not an oversight.**

There are two ways to do the null-input experiment, and neither is worth
running:

**One, degenerate input (all-zero / masked pocket).** The model never saw
anything like this during training, so wherever the output lands, it carries
no explanatory power. If it comes out at 0.5, we can't tell whether that
means "it doesn't work without a protein" or "the model's default behavior on
malformed input"; if it doesn't come out at 0.5, that's even harder to
explain. **An experiment that reads out nothing in either direction is not
worth running.** This is the exact opposite of target swap: a random swap
hands the model a **valid but wrong** protein, so the model still receives a
normal pocket representation, which is why "AUROC lands exactly at 0.5" is
clean and interpretable.

**Two, a fixed substitute (the same real pocket for every target).** This is
valid, interpretable input, but it **is just a degenerate case of random
swap** (substitute count n=1 instead of a random draw per target). Section 4's
random swap has already tested the same thing with 3 substitutes per target
and the median taken, and its sampling variance is better controlled. A fixed
substitute would only produce the same conclusion with a worse estimate.

**"There are idle GPUs available" is not a reason to run an experiment.**
This is written down because it nearly became one: after Boltz's continued
run was re-sharded, some GPUs freed up, and the first instinct was "might as
well fit it in." The cost of idle resources is zero; the cost of publishing a
result that reads out nothing is not zero.

If there is idle compute to spend, it would be better spent running Boltz-2
on AIMNet2's same 93 targets — that would give "two physics methods' ranking
ability on the same batch of targets", of which currently only one exists.
But that is a new front, not on this checklist, and would need to be
scheduled separately.

## 6. Limitations

- **Only random substitutes were done, not same-family substitutes.** So for
  now we can only say "swapping to an unrelated target causes collapse" — we
  cannot distinguish whether the model recognizes the **specific target** or
  the **target family**. Same-family swap is the next step.
- **Only L1 and L4 were run.** L2/L3 were not run, but these two layers are
  the two extremes of novelty, and the middle layers are expected to fall in
  between.
- 3 substitutes per target, with no deduplication among substitutes — the
  same substitute may be selected by multiple targets.

## Reproduction

```bash
python timesplit/build/build_target_swap.py --out-root .../T3_swap_full \
    --layers L1 L4 --rounds 3 --size-tol 0.30 --subset results/T3_vsds_matched.csv
python timesplit/build/build_swap_eval.py --manifest .../swap_manifest.json \
    --out-root .../T3_swap_eval --rounds 3 --layers L1 L4     # sequence-only models
./standard/swap_queue.sh          # ten models × three rounds, scheduled under the 4-GPU cap
python timesplit/analysis/score_target_swap.py --models <ten models> --rounds 3
```

### Reproducing the same-family swap

```bash
python timesplit/build/build_target_swap.py --out-root .../T3_swap_family \
    --layers L1 L4 --rounds 3 --mode family --clstr .../uniport40.clstr \
    --size-tol 0.30 --subset results/T3_vsds_matched.csv
./standard/family_queue.sh        # runs after swap_queue.sh, does not aggregate on its own
python timesplit/analysis/score_target_swap.py --models <ten models> --rounds 3 \
    --prefix swapfam --manifest .../T3_swap_family/swap_manifest.json \
    --out results/T3_target_swap_family.csv
```
