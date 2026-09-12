# T3 Leakage Diagnostics

This started from Mattsson & Walters, *Identifying and Addressing Systematic Data Leakage
in Protein-Ligand Affinity Benchmarks*, bioRxiv 2026-06-30,
DOI `10.64898/2026.06.29.735309`. We checked its three claims against our own data, one by one:

1. **Splitting by sequence identity is not enough to prevent leakage** ("target mirroring":
   homologous proteins remain correlated in binding profile even when overall identity is
   very low; on 6,000+ pairs from ChEMBL 36, leakage persists down to 0.2)
2. **A chemical-series oracle** (blind to the protein, but reading that target's known
   actives) reaches r = 0.66 on FEP+
3. Results should be **reported stratified by ligand novelty** (the hardest tier being
   Tanimoto < 0.35)

The conclusion up front: **for the four models trained on the affinity half, L1 is
essentially a memorisation test, and L3/L4 are clean** (this subject is necessary — see
§3c); along the way we also found a bug in our own layer-assignment logic.

---

## 1. Target homology: targets the model has seen leaked into L4

`timesplit/analysis/target_mirroring.py` → `results/T3_target_mirroring.csv`

Using a sensitive mmseqs2 search (`-s 7.5`), we compared all 866 T3 targets against the
training set, **requiring the alignment to cover ≥50% of each sequence, with E ≤ 1e-3**.

> ⚠️ Without gating on coverage, the search returns a pile of fragment hits with
> `fident=1.00` but `qcov=1%` — the first-round results were all this kind of noise;
> "100% identity" turned out to be just a handful of residues lining up.

⚠️ **The reference set was changed once.** Originally we only compared against the
affinity half's **2,196** targets, but LigUnity-family training reads **two** label files
(see §6), so the actual training targets are the union of **4,847**. So the earlier
homology hit rate was **systematically underestimated**. The two versions side by side:

| 307 "novel targets" in L3/L4 | Old (vs. 2,196) | **New (vs. union of 4,847)** |
|---|---|---|
| ≥70% has a homologue | 30 (10%) | **45 (15%)** |
| ≥50% | 61 (20%) | **80 (26%)** |
| ≥40% | 80 (26%) | **106 (35%)** |
| ≥30% | — | 146 (48%) |
| ≥20% (Walters reports leakage down to this level) | 153 (50%) | **180 (59%)** |
| Median identity | — | 28.8% |

**And 35 of the 307 "novel targets" (11.4%) have a 100% self-match in the training
set** — they are simply *in* the training set, not "have a homologue". These include
O00443, P05067 (APP), P08519, P09391, P10636 (tau), among others.

As a comparison, for the 559 targets in L1/L2: median identity 76.7%, 313 at ≥70% — this
layer is supposed to be in the training set, as expected.

**The direction is the same as that earlier fall-through bug: seen targets leaked into
L4 → the decay is underestimated → fixing it can only make the headline conclusion
stronger.**
([`results/T3_target_mirroring_union.csv`](../results/T3_target_mirroring_union.csv))

The highest-homology cases are mostly **cross-species orthologs**:

| Identity | T3 "novel target" | Training-set match | qcov / tcov |
|---|---|---|---|
| **100.0%** | I6WXK4 | P96830 | 100% / 100% |
| 99.6% | P38158 | P07265 | 100% / 100% |
| 97.9% | P20648 | P19156 | 100% / 100% |
| 96.6% | P05622 (PDGFRB) | Q05030 | 100% / 100% |
| 90.7% | P43220 (GLP1R) | P32301 | 100% / 100% |

### Root cause: a fall-through bug

`timesplit/build/build_t3.py`:

```python
f = fam.get(up)
layer = "L3" if (f is not None and f in train_fams) else "L4"
```

`fam` comes from `uniport40.clstr` (CD-HIT 40% clustering) provided by LigUnity.
**If a target is not in this file, `fam.get()` returns None, and it falls straight
through to L4** — "family not found" was treated as "has no homologous family".

**Of the 254 targets in L4, 61 (24%) are simply absent from that file**, and 10 of them
have ≥70% homology to the training set.

**Correcting under the 40% criterion: L4 254 → 224, 30 targets should be relabelled L3.**
After the fix, the main-table decay moves from −69% to −78% (see
[T3 dataset v2](T3-dataset-v2.md)).

---

## 2. Chemical-series oracle ceiling: reaches 98.7% of the theoretical limit

`timesplit/analysis/ligand_only_baseline.py` → `results/T3_ligand_only.csv`
(the script's filename is a historical holdover — **this quantity is not a "ligand-only
baseline"**, see below)

Using only ECFP4 to compute "each candidate molecule's maximum Tanimoto to that target's
known actives" (leave-one-out for actives — a molecule is never allowed to match itself),
then computing the metrics as usual:

⚠️ **The name needs to change, and this is not wordplay.** Although it never looks at
protein structure, it **reads that target's known active molecules** — information no
evaluated model is given. Calling it a "ligand-only baseline" would lead a reader to
think "you can do this well without looking at the protein at all", when the correct
reading is "**if you already know what binds this target, how far can pure chemical
similarity go**".

The accurate name is a **target-conditioned ligand-similarity oracle**, or for short a
**chemical-series oracle ceiling**. The real "ligand-only baseline" is the one in §2b —
it doesn't even know which target it's scoring, and it comes out **below random**. The
two differ by two orders of magnitude, and conflating them under one name would flip the
conclusion.

| Layer | EF1% | BEDROC | AUROC | PR-AUC |
|---|---|---|---|---|
| L1 | 48.04 | 0.889 | 0.940 | 0.836 |
| L2 | 50.47 | 0.981 | 0.988 | 0.968 |
| L3 | 50.07 | 0.962 | 0.978 | 0.946 |
| L4 | **50.36** | 0.973 | 0.987 | 0.962 |
| *Random* | *1.08 ± 1.44* | | *0.500* | *~0.020* |
| **Theoretical ceiling** | **51.00** | | | |

**All four layers reach 98.7% of the maximum score.** The reason is that a given target's
actives are mostly one congeneric series, while decoys are actives of other targets and
sit far away in chemical space.

### How to read this number

This oracle **has more information than the models do** — it can see that target's true
actives, while the models see none of them at L3/L4. So it is a **ceiling**: it
quantifies how tightly clustered this set of actives is in chemical space, not what a
model can achieve.

But it gives a sharper explanation for T3's decay:

> The chemical-series oracle stays at ~50 across all four layers, almost unchanged, while
> the models fall from 38 to 9.
> **The decay is not the model "getting dumber" — it's the model losing the memorisable
> chemical series.**

### Is the ceiling an artefact of our own decoy rule?

The decoy builder refuses any candidate sharing a Bemis-Murcko scaffold with one
of the target's actives (`build_t3_eval.py`). That rule makes the pool easier to
separate by fingerprint, so part of the oracle's ~99% could be something we wrote
into the construction rather than something we found. The way to settle it is to
turn the rule off and rebuild.

Rebuilt on the 350-quota targets with `--no-scaffold-exclusion`, the oracle
barely moves ([`results/T3_ligand_only_noscaf.csv`](../results/T3_ligand_only_noscaf.csv)):

| Layer | EF@1% with the rule | without it | % of the 51.00 ceiling |
|---|---|---|---|
| L1 | 50.77 | 50.43 | 99.6% → 98.9% |
| L2 | 50.86 | 50.77 | 99.7% → 99.5% |
| L3 | 50.97 | 50.97 | 99.9% → 99.9% |
| L4 | 50.91 | 50.81 | 99.8% → 99.6% |

AUROC is unchanged to three decimals at every layer. **So the ceiling is not an
artefact of the rule** — but the reason matters, and it is not "scaffold
collisions are harmless".

**The rule is close to vacuous.** Counting, in the rebuilt sets, how many decoys
actually share a scaffold with one of that target's actives: a median of **7 out
of ~5,000 at L1 (0.12%), and exactly zero at L2, L3 and L4**. Cross-target
actives almost never share a Bemis-Murcko scaffold with a given target's actives,
so the filter had nothing to remove. What holds the ceiling up is the other fact
— a target's own actives are a congeneric series, which leave-one-out maximum
Tanimoto reads directly.

⚠️ One thing this comparison does *not* isolate. Decoys are sampled at random
from the candidate pool, so turning the rule off and resampling replaces 94–97%
of them regardless. The stability above is therefore across two nearly disjoint
random draws as well as across the rule — reassuring for the ceiling being a
property of the actives, but it means the rule's own contribution is pinned down
by the 0.12% collision count, not by the difference between the two runs.

### Normalised metric

`results/T3_normalized_by_ceiling.csv`. Model EF1% ÷ the chemical-series oracle ceiling
on the same set of targets:

| Model | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| LigUnity-protein | **79%** | 63% | 38% | **17%** |
| LigUnity-pocket | 69% | 54% | 29% | 16% |
| LiTENCLIP | 64% | 45% | 16% | 18% |
| HypSeek `_vs` | 63% | 45% | 21% | 15% |
| DrugCLIP | 34% | 24% | 17% | 16% |
| ConGLUDe | 25% | 13% | 13% | 10% |
| ConPLex | 12% | 7% | 7% | 4% |
| SPRINT | 5% | 4% | 3% | 3% |

**One sentence for the whole of T3**: the best model extracts 79% of the available
signal on familiar targets, and only 17% on entirely novel ones.

⚠️ Regenerated 2026-09-12 from the current subset table; the previous version of this
table predated both the EF tie-handling fix and the `_vs` switch. HypSeek's row is the
screening weight, as everywhere else on the screening side — on `_rk` it reads
68 / 50 / 25 / 15.

---

## 2b. The real ligand-only baseline: below random

The 50.36 figure in §2 is an **oracle ceiling**, not a model: it reads that target's true
actives. Take that information away — train a classifier that only sees ECFP4 and **has
no idea which target it's scoring at all** — and *that* is what deserves the name
"ligand-only baseline" — the result flips
([`timesplit/analysis/ligand_only_learned.py`](../timesplit/analysis/ligand_only_learned.py)
→ [`results/T3_ligand_only_learned.csv`](../results/T3_ligand_only_learned.csv)):

`HistGradientBoostingClassifier` + ECFP4, with **GroupKFold grouped by uniprot**. Grouping
must be by target rather than by molecule: a given target's actives are mostly one
chemical series, and splitting by molecule would put the two halves of a series on either
side of the train/validation boundary — leaking the answer to yourself.

Task-list item 4.2 called for **two implementations** — a GBDT and a 2-layer MLP — on the
grounds of "confirming this isn't an artefact of one particular model". Tree models and
neural nets have very different inductive biases on sparse binary fingerprints, so
agreement between the two is what counts:

| Layer | Targets | GBDT EF1% | GBDT AUROC | AUROC&lt;0.5 | MLP EF1% | MLP AUROC | AUROC&lt;0.5 |
|---|---|---|---|---|---|---|---|
| L1 | 56 | 0.50 | 0.513 | 52% | 0.70 | 0.511 | 50% |
| L2 | 178 | 0.33 | 0.428 | 68% | 0.28 | 0.422 | 68% |
| L3 | 19 | 0.18 | 0.442 | 68% | 0.21 | 0.414 | 74% |
| L4 | 75 | 0.21 | 0.379 | 76% | 0.16 | 0.374 | 75% |
| *Random* | | *1.00* | *0.500* | | *1.00* | *0.500* | |

**All four layers come out below random, and the two model families give almost
identical numbers** — weighted-average AUROC is 0.4321 for GBDT and 0.4256 for MLP, a
difference of 0.0065. In the GBDT run, 259 of 328 targets (79%) have an EF1% of 0.

The MLP is a two-layer ReLU `(512, 128)`; `early_stopping` carves off a further 10%
**inside the training fold** for validation, never touching the test fold (otherwise it
would amount to tuning the stopping point on test data). `--clf {gbdt,mlp}` switches
between them, and output filenames follow automatically.

### Why this is good news

This is exactly what cross-target real-active decoys should look like. Decoys are real
actives of other targets, so "drug-likeness" carries no discriminative power between
actives and decoys — a model that looks only at chemistry not only fails to learn
anything, it gets biased below random by the class imbalance in the training set.

**Put together, the two baselines are what makes the complete statement**:

> Knowing what that target's known actives look like → reaches 98.7% of the ceiling (§2)
> Not knowing, and looking only at the chemistry itself → below random (this section)
>
> So **all** of T3's signal comes from "how similar is this molecule to this target's
> known ligands", and none of it comes from "how drug-like the molecule itself looks".
> The decoy design is clean.

It also pins down the explanation for the L1→L4 decay: the model is not getting dumber
at L4 — L4 simply has no known ligand to match against, and chemistry alone carries no
signal at all.

## 2c. Chemical memorisation preference (CMP): once decomposed, only a small residue remains

[`chemical_memory_pref.py`](../timesplit/analysis/chemical_memory_pref.py)
→ [`results/T3_chemical_memory_pref.csv`](../results/T3_chemical_memory_pref.csv)

Let S be a molecule's maximum ECFP4 Tanimoto to the training ligands; S ≥ 0.7 counts as
"familiar":

```
CMP = P(familiar | ranked into top 1%) − P(familiar | whole candidate pool)
```

### ⚠️ This number cannot be read directly as "preference"

**The novelty distributions of actives and decoys are simply different to begin
with** — in the §3 table, 53.9% of L1 actives fall in the ≥0.7 tier, against only 8.7%
of decoys. So an **accurate** model will get a positive CMP purely from ranking actives
up, even with zero preference for familiar chemistry. The first version of this script's
docstring got this backwards, stating "looking at all molecules in the top 1% is
independent of accuracy" — which is exactly the wrong way round.

Subtract out the part accuracy alone can explain. Let π be the fraction of actives in the
top 1%, a the fraction of that target's actives that are familiar chemistry, d the
corresponding fraction among decoys, and q the familiar fraction in the whole pool:

```
CMP_pred = [π·a + (1−π)·d] − q        what "ranking actives up" alone would earn
CMP_exc  = CMP − CMP_pred             the residual — this is the actual preference
```

### After decomposition

| Model | Layer | CMP | CMP_pred | **CMP_exc** | Within-actives control |
|---|---|---|---|---|---|
| HypSeek `_rk` | L1 | +0.460 | +0.322 | **+0.138** | +0.185 |
| | L4 | +0.083 | +0.017 | **+0.066** | +0.055 |
| LigUnity-pocket | L1 | +0.385 | +0.303 | **+0.082** | +0.136 |
| | L4 | +0.021 | +0.007 | **+0.014** | +0.022 |
| LigUnity-protein | L1 | +0.406 | +0.330 | **+0.077** | +0.112 |
| | L4 | +0.092 | +0.029 | **+0.063** | +0.140 |
| LiTENCLIP | L1 | +0.360 | +0.283 | **+0.077** | +0.125 |
| DrugCLIP | L1 | +0.178 | +0.156 | **+0.022** | +0.185 |
| BindCLIP-hardneg | L1 | +0.198 | +0.153 | **+0.044** | +0.103 |
| SPRINT | L1 | +0.067 | +0.012 | **+0.055** | +0.135 |

**Of that +0.46 at L1, +0.32 is purely "the model is accurate".** The true preference
residual is only +0.02 to +0.14 — an order of magnitude smaller than the raw number.

### Three readings

**One: the preference is real, but small.** All eight models' CMP_exc are positive (the
−0.004 for LigUnity-protein at L3 is the sole exception, and that cell has only 16
targets). An independent corroborating check points the same way: comparing **within a
target's actives** only (retrieved actives vs. all actives, naturally immune to the
active/decoy distribution difference) gives +0.02 to +0.19, matching CMP_exc in both
magnitude and ranking.

**Two: the preference barely changes across layers, even though the raw CMP collapses by
an order of magnitude.** Take HypSeek: raw CMP falls from +0.460 at L1 to +0.083 at L4
(an 82% drop), while CMP_exc only falls from +0.138 to +0.066. The three DrugCLIP-family
models' CMP_exc has a range of only 0.009–0.019 across all four layers — essentially a
flat line. **So the impression that "the model stops preferring familiar chemistry on
novel targets" is false — it has the same preference throughout, it simply stops being
accurate on novel targets.**

**Three: the strongest model has the heaviest preference.** HypSeek's CMP_exc is highest
at every layer (+0.138 / +0.088 / +0.075 / +0.066), and the DrugCLIP family is lowest
(+0.014–0.022). That agrees in direction with §6's finding that it "improves rank the
most on its own training set", but the samples and criteria differ between the two, so
this is not enough to chain into one conclusion.

### ⚠️ The table above uses the wrong reference set for the three DrugCLIP-family models

Novelty was computed against **the affinity half's 428,767 ligands**. But the training
ligands for DrugCLIP and the two BindCLIPs are the molecules in the 66,164
`train_no_test_af` pairs, only **13,590** after deduplication (the other side of §6's
31.6× gap). Recomputing with group A's own ligands
(`--novelty data/t3/ligand_novelty_drugclip.json`
→ [`results/T3_chemical_memory_pref_Aref.csv`](../results/T3_chemical_memory_pref_Aref.csv)):

| Reference set | Median T3-molecule similarity | ≥0.7 | ≥0.5 | **<0.35** |
|---|---|---|---|---|
| Affinity half, 428,767 ligands | 0.419 | 8.8% | 31.2% | 23.6% |
| **Group A, 13,590 ligands** | **0.303** | **0.9%** | **5.3%** | **72.6%** |

**Measured against DrugCLIP's own training ligands, 72.6% of T3's molecules are novel
chemistry, and only 0.9% qualify as "very close".** Its CMP is accordingly close to
zero:

| Model | Layer | Pool familiar% | CMP | CMP_exc | Within-actives control |
|---|---|---|---|---|---|
| DrugCLIP | L1 | 1.0% | +0.043 | **+0.012** | +0.178 |
| | L2 | 0.8% | −0.000 | **+0.001** | −0.000 |
| | L4 | 0.8% | −0.001 | **−0.001** | +0.037 |
| BindCLIP-randneg | L1 | 1.0% | +0.050 | **+0.019** | +0.194 |
| BindCLIP-hardneg | L1 | 1.0% | +0.041 | **+0.011** | +0.128 |

(Compare against +0.022 / +0.025 / +0.044 above, computed with the wrong reference set.)

### ⚠️ A conclusion drawn from this was wrong and has been retracted

After seeing "72.6% of the pool is novel to group A", this document once stated:

> DrugCLIP's enrichment at L1 was achieved on almost entirely novel chemistry, so
> "relying on memorised familiar chemistry" does not apply to group A — it's simply a
> property of the four [affinity-trained] models.

**That inference was wrong.** "Most of the pool is novel" does not imply "the tiers carry
no discriminative power" — quite the opposite: switching to the correct reference set
actually widens the dynamic range of novelty. Re-running the tiered enrichment with group
A's own ligands
([`results/T3_novelty_tiered_ef_Aref.csv`](../results/T3_novelty_tiered_ef_Aref.csv)):

| Model | Layer | Novel <0.35 | Distant .35–.5 | Close .5–.7 | Very close ≥0.7 | Very-close/novel |
|---|---|---|---|---|---|---|
| DrugCLIP | L1 | 9.7 (42) | 18.6 (43) | 29.6 (36) | **25.4 (26)** | **2.6×** |
| BindCLIP-randneg | L1 | 11.5 (42) | 20.9 (43) | 26.5 (36) | **30.9 (26)** | **2.7×** |
| BindCLIP-hardneg | L1 | 11.9 (42) | 20.3 (43) | 26.2 (36) | **26.8 (26)** | **2.3×** |

**At L1 it rises monotonically, so the effect is real** — just smaller than under the
wrong reference set (DrugCLIP 5.2× → 2.6×). So "models enrich better on familiar
chemistry" is **a shared property of all seven pocket models**, not a property specific
to the four [affinity-trained] models.

⚠️ **Only L1 is legible.** Under the correct reference set the "very close" tier at
L2/L3/L4 shrinks to just 1–8 targets (a 0.9% pool share can't support more), so those
rows are non-monotonic and not citable.

### Two methodological lessons

**One: drawing a conclusion from the distribution alone is the same class of mistake as
drawing one from the mean alone.** The wrong inference above was made straight from the
pool composition (72.6% novel) without waiting for the tiered enrichment to come out.

**Two: the two groups have different usable statistics and cannot be placed side by
side.** Under the 428,767 reference set, the two extreme tiers (≥0.7 vs <0.35) drop to
only 5–9 targets with enough sample past L2, so only the "halves, ≥0.5 vs <0.5" split can
be used; under the 13,590 set, the halves are diluted to 5.3% and lose power, while the
two extreme tiers instead have 26–42 targets. **Placing group A's and group B's tier
numbers side by side in the same table invites the reader to directly compare two
incomparable quantities.**

(The original CMP observation that "group A's residual is near zero" still holds, but it
measures the **composition** of the top 1%, which is a different thing from the tiered
enrichment here: familiar molecules are only 0.9% of the pool, so the composition is
naturally thin; the tiered enrichment instead asks "are actives that fall in the familiar
tier more easily retrieved" — and the answer is yes. Group A's within-actives control of
+0.13 to +0.19 at L1 points the same way, but variance is high on a 0.9% baseline, so it
should not be cited on its own.)

### Another known defect (fixed; the numbers are unaffected)

Section B of `ligand_novelty.py` (the second table in §3) used to treat
`actives + decoys` directly as the molecule order, **validating only the length, not the
labels** — exactly the pitfall this document keeps coming back to. A different agent has
since added a hard validation check and re-verified cell by cell: all six cells match
what was already published exactly (HypSeek L1 0.969 / 74.1%, LigUnity-protein 0.814 /
66.4%, DrugCLIP 0.818 / 65.2%, all three L4 rows match too), and the skipped-target counts
of 46/31/47/32 also fall within the range the document states. **So that table was in
fact produced with validation already applied; the numbers don't need to change — what
was missing was reproducibility** — the version in the repository has no validation, and
re-running it as-is would silently produce unvalidated numbers. Fixed (`97e0f06`).

---

## 3. Ligand novelty: leakage is entirely confined to L1

### 3a. Exact overlap (InChIKey)

| Layer | Actives that appear in the training set | Decoys (background) |
|---|---|---|
| **L1** | **32.1%** (5,479/17,066) | 3.7% |
| L2 | 0.0% (19/102,005) | 3.7% |
| L3 | 2.5% (110/4,358) | 3.8% |
| **L4** | **3.2%** (1,042/32,774) | 3.7% |

At L1, a third of the actives are literally the **same molecules** as ones in the
training set (8.7× the decoy background). **L4's 3.2% is level with the decoy background
of 3.7% — L3/L4 have no exact leakage.**

(What the time split guarantees is that the *record* is new; a molecule that was tested
against target A in the training set and tested against target B in T3 is still a
molecule the model already knows.)

#### Switching to the structure half's own ligands: absolute magnitude two orders smaller, relative bias larger

The table above is computed against the affinity half's 428,767 ligands. Switching to
the structure half's 13,590 (→ 12,555 InChIKeys):

| | Affinity half, 428,767 | **Structure half, 13,590** |
|---|---|---|
| L1 actives | **32.1%** (5,479/17,066) | **3.2%** (542/17,066) |
| L1 decoys (background) | 3.70% (31,566/853,300) | **0.25%** (2,144/853,300) |
| **L1 actives/decoys** | **8.7×** | **12.6×** |
| L4 actives | 3.2% | 0.2% (64/32,774) |
| L4 decoys | 3.7% | 0.25% |
| **L4 actives/decoys** | **0.86×** | **0.8×** |

⚠️ **"3.2% against 32.1%" cannot be read as "the structure half has no exact leakage".**
It must be compared against **its own decoy background** — the training ligand pool is
31.6× smaller, and the decoy background falls correspondingly from 3.70% to 0.25%. **The
relative enrichment is the same on both sides, and if anything higher for the structure
half** (12.6× vs 8.7×).

So the accurate statement takes two sentences, and neither can be dropped:

> **In relative terms**, L1 actives in both groups are more likely to be training-set
> molecules than the decoy background is (8.7× / 12.6×) — an inherent property of the
> time split, regardless of which training set is used; at L4 both groups sit near 1
> (0.86× / 0.8×), i.e. genuinely no exact leakage.
>
> **In absolute terms**, a third of L1 actives are the same molecules for the
> affinity-half group, versus only 3% for the structure-half group. **So "L1 is
> essentially a memorisation test" holds only for the former** — for the latter it is
> "relatively biased, small in absolute magnitude".

This, together with §3c (structure-half L1 retrieved-into-very-close tier 16.1% vs. pool
9.0%, 1.8×), §4 (tiered enrichment, L1 very-close/novel 2.6×), and §2c (CMP residual
+0.012), points to the same shape in all four places: **relatively biased, small in
absolute magnitude.**

### 3b. Continuous novelty tiers

`timesplit/analysis/ligand_novelty.py` → `results/T3_ligand_novelty.csv`

Computing the maximum Tanimoto against the training set's 428,767 deduplicated ligands
(once per each of T3's 146,919 unique molecules):

| Layer | Active median | Novel <0.35 | Distant .35–.5 | Close .5–.7 | Very close ≥0.7 |
|---|---|---|---|---|---|
| **L1** | **0.727** | 2.1% | 14.9% | 29.1% | **53.9%** |
| L2 | 0.417 | 24.1% | 47.4% | 25.4% | 3.1% |
| L3 | 0.372 | 37.8% | 39.9% | 16.8% | 5.5% |
| L4 | 0.380 | 35.5% | 46.4% | 11.8% | 6.4% |
| *Decoys (identical across layers)* | *0.419* | *23.6%* | *45.3%* | *22.3%* | *8.8%* |

**At L1, more than half the actives are near-copies of training molecules. L2/L3/L4
actives are even more novel than the decoys** — a strong endorsement of the cross-target
real-active decoy design.

⚠️ **The reference set for this entire table is the affinity half's 428,767 ligands, and
it only holds for those four models.** Measuring the same L1 actives against the
structure half's own 13,590 ligands:

| L1 actives | Affinity-half reference | **Structure half's own reference** |
|---|---|---|
| Median similarity | 0.727 | **0.371** |
| Novel <0.35 | 2.1% | **42.2%** |
| Very close ≥0.7 | **53.9%** | **9.0%** |

L4 goes the other way even more extremely: under the structure-half reference, **80.1%**
of L4 actives are novel chemistry (versus 35.5% under the affinity-half reference).

**So the statement "L1 is a memorisation test" must carry a subject.** For the three
DrugCLIP-family models, 42.2% of L1's actives are novel chemistry, and only 9% are
near-copies — their L1 is not a memorisation test.

This incidentally explains **why the two groups' L1 scores differ by nearly twofold**
(32–39 vs. 17–19): half of L1 is molecules the affinity half has seen and the structure
half has not. Part of the gap is not a difference in ability — it's this layer's
composition favouring one group.

### 3c. Model preference: systematically retrieving chemistry it has seen

The novelty-tier composition of actives models rank into the top 1%. **The reference set
for the table below is again the affinity half; see the correction at the end of this
section for the two DrugCLIP rows.**

| Model | Layer | Retrieved median | Novel <0.35 | Very close ≥0.7 |
|---|---|---|---|---|
| HypSeek | **L1** | **0.969** | 1.1% | **74.1%** |
| LigUnity-protein | L1 | 0.814 | 1.1% | 66.4% |
| DrugCLIP | L1 | 0.818 | 2.0% | 65.2% |
| *(L1 pool composition)* | | *0.727* | *2.1%* | *53.9%* |
| HypSeek | L4 | 0.448 | 22.1% | 12.9% |
| LigUnity-protein | L4 | 0.452 | 25.5% | 14.9% |
| DrugCLIP | L4 | 0.415 | 22.8% | 7.5% |
| *(L4 pool composition)* | | *0.380* | *35.5%* | *6.4%* |

**Models systematically favour retrieving chemistry they've seen.** HypSeek's retrieved
actives at L1 have a median similarity of **0.969** — near-copies of training ligands —
against a pool median of only 0.727. At L4, 35.5% of the pool is novel molecules, but the
model retrieves only 22%.

### ⚠️ The reference set for the two DrugCLIP rows is wrong; the preference is much weaker after recomputing

The novelty in the table above is computed against the affinity half's 428,767 ligands,
while the DrugCLIP family's training ligands number only 13,590. Recomputing with their
own ligands (→ `results/T3_ligand_novelty_Agroup.csv`):

| DrugCLIP | Retrieved median | Novel <0.35 | Very close ≥0.7 |
|---|---|---|---|
| **L1 retrieved** | 0.442 | 25.6% | **16.1%** |
| *L1 pool* | *0.371* | *42.2%* | *9.0%* |
| **L4 retrieved** | 0.305 | 65.9% | 0.6% |
| *L4 pool* | *0.291* | *80.1%* | *0.6%* |

**At L1 the preference is real but much weaker**: the very-close tier is 16.1% against
the pool's 9.0%, a **1.8×** lift; the novel tier is 25.6% against 42.2% — systematically
under-retrieved. Compare against the "retrieved median 0.818, very-close 65.2%" figures
in the table above — those were measured against someone else's training set.

**At L4 the preference is essentially gone**: the very-close tier goes from 16.1% →
0.6%, level with the pool's 0.6%, leaving only the "under-retrieving the novel tier"
effect (65.9% vs. 80.1%).

This points the same way as §2c's CMP residual and §4's tiered enrichment re-run under
the A reference set (L1 very-close/novel 2.6×) in all three places: **the preference is
a shared property of all seven pocket models, but for the three structure-half models it
is only legible at L1, and its magnitude is about half that of the four affinity-half
models.**

> ⚠️ **The first version of this section was wrong**, and it must be recorded: I assumed
> the molecule order the model saw was "actives + decoys" (jsonl order), and the
> distribution that came out of that was almost identical to the pool background. That
> is exactly the signature of **misaligned molecule order** — the model reads from lmdb,
> whose cursor order is lexicographic (`0, 1, 10, 100, …`), not jsonl order. **Equal
> length does not mean equal order.** The table above was only obtained after adding a
> hard validation check (the position labelled 1 must genuinely be an active, otherwise
> the target is skipped). 31–47 targets fail validation under both orderings and were
> skipped. This is the third occurrence of the same pitfall — see
> [PATCHES.md](../PATCHES.md).

---

## 4. Enrichment computed separately per novelty tier — the realistic-scenario numbers

`timesplit/analysis/novelty_tiered_ef.py` → `results/T3_novelty_tiered_ef.csv`

Section 3 answered "how novel are the test ligands" and "does the model retrieve toward
the familiar or the novel side", but not the most direct question: **what is the model's
enrichment specifically on the novel-chemistry tier**. This is exactly the number the
Novelty-Tiered Benchmark in the Walters paper calls for.

Metric definition (comparable directly across tiers, unaffected by how many actives a
tier holds):

```
recall_t = number of that tier's actives that land in the top 1% / total actives in that tier
EF_t     = recall_t / 0.01          # EF_t = 1 means the same as random
```

Computed per target and then averaged; only targets with at least 3 actives in that tier
are counted.

| Model | Layer | Novel <0.35 | Distant .35–.5 | Close .5–.7 | Very close ≥0.7 |
|---|---|---|---|---|---|
| **LigUnity-protein** | L1 | 26.7 (n=18) | 19.1 (106) | 33.7 (184) | **52.4** (245) |
| | L2 | 18.6 (234) | 26.5 (373) | 44.2 (330) | 55.7 (152) |
| | L3 | 7.0 (20) | 13.3 (33) | 19.4 (22) | 32.6 (11) |
| | **L4** | **4.5** (103) | 7.5 (169) | 14.8 (111) | **27.1** (49) |
| LigUnity-pocket | L4 | 7.0 | 8.2 | 14.4 | 15.0 |
| HypSeek | L4 | **3.8** | 5.6 | 11.2 | 19.5 |
| LiTENCLIP | L4 | 5.5 | 6.8 | 13.6 | 20.0 |
| DrugCLIP | L3 | **0.6** | 6.3 | 6.2 | 17.5 |
| DrugCLIP | L4 | 5.3 | 6.8 | 11.0 | 11.9 |

(Numbers in parentheses are the target counts included in the statistic.)

⚠️ **The reference set for the two DrugCLIP rows is wrong.** Novelty is computed against
the affinity half's 428,767 ligands, while DrugCLIP's training ligands number only
13,590 (see §2c). Measured against its own ligands, **72.6%** of T3's molecules fall
into "novel <0.35" and only **0.9%** reach "very close ≥0.7" — the "very close tier
17.5" shown for DrugCLIP L1 in the table isn't "very close" by its own standard at all.
**This entire tiering scheme does not hold for the three DrugCLIP-family models** —
under the correct reference set, the "familiar" tier can't even muster 3 actives per
target. The two DrugCLIP rows are kept in the table only so the shape can be compared
side by side with the other models; do not cite its tier values on their own.

### Two conclusions

**One: the realistic-scenario number is an order of magnitude smaller than the main
table's.** The main table reports L4 EF1% of 9.38, which is the mean over all actives
pooled together. Split apart — **novel target + novel chemistry, i.e. the genuine
drug-discovery scenario, is only 4.5**; novel target + familiar chemistry is 27.1. (Both
numbers are from LigUnity-protein, which is trained on the affinity half, so the tier
reference set is correct for it.)

⚠️ **This document previously also stated here that "DrugCLIP's novel tier at L3 is 0.6,
below random." That sentence has been deleted.** That tier was cut against the affinity
half's 428,767 ligands, while DrugCLIP's training ligands number only 13,590 — by its
own reference set, its "novel" tier is **9.7**, not 0.6. The correct tiers for the three
DrugCLIP-family models are in the correction table at the end of this section.

**Two: both ligand novelty and target novelty are at work, but they do not amplify each
other.**

First looking at LigUnity-protein alone:

| | Very close ≥0.7 | Novel <0.35 | Novel/familiar ratio |
|---|---|---|---|
| L1 | 52.4 | 26.7 | 0.51 |
| L4 | 27.1 | **4.5** | **0.17** |

- Changing only the target (L1 → L4, familiar chemistry both times): 52.4 → 27.1, a
  **1.9×** drop
- Changing only the chemistry (within L4): 27.1 → 4.5, a **6.0×** drop
- The same change in chemistry costs 2.0× at L1 but 6.0× at L4 — amplified **3.1×**

⚠️ **These are the full 1,144-target numbers, and this particular comparison cannot be
repeated on the 350-quota subset the paper otherwise reports.** The L4 half transfers and
gets stronger there — 39.6 on chemistry the model has seen against **4.9** on chemistry it
has not, an eightfold span rather than sixfold
([`results/T3_novelty_tiered_ef_subset.csv`](../results/T3_novelty_tiered_ef_subset.csv)).
The L1 half does not: the subset's "novel <0.35" tier at L1 holds **7 targets**, and reads
50.8 — above its own "very close" tier, which is not a result but a seven-target
fluctuation. The amplification ratio needs both layers, so it stays a full-set statement,
and the tier table is the reason rather than an afterthought.

⚠️ **This amplification only holds for this one model.** An earlier version of this
document wrote, on this basis, "when the target is novel, the cost of ligand novelty is
sharply amplified" — that was generalised from a single model, and **the other four
models do not support it**:

| Model | L1 novel/familiar | L4 novel/familiar | Amplification factor |
|---|---|---|---|
| LigUnity-protein | 0.51 | 0.17 | **3.09** |
| HypSeek `_rk` | 0.20 | 0.19 | 1.03 |
| LiTENCLIP | 0.22 | 0.27 | 0.81 |
| LigUnity-pocket | 0.30 | 0.47 | 0.65 |
| DrugCLIP | 0.19 | 0.44 | 0.43 |
| **Median** | **0.22** | **0.27** | **0.81** |

Three of the five models go the opposite direction (the cost of changing chemistry at L4
is actually **smaller**). And LigUnity-protein's L1 novel tier has only **18 targets** —
the thinnest cell in the entire table.

Section 6 redid the same thing against each model's own training set, with the same
conclusion: the novel/familiar ratio's median is **0.26** in both the seen and unseen
groups — no amplification.

**So the correct statement is**: the L1→L4 decay is indeed "novel target" and "novel
ligand" stacking together, but the two are **approximately independent** (multiplicative,
no interaction term) — the latter is not amplified by the former. Ligand novelty does
carry more weight — changing chemistry costs 4–5×, changing the target costs about 2× —
and that half of the conclusion holds up.

### Data trail

46 / 30 / 9 / 31 targets (L1/L2/L3/L4) fail the molecule-order validation and were
skipped. The same set of targets is skipped across every model, which shows this is a
property of those targets (neither the jsonl nor the lmdb ordering lines up with the
labels), not an issue with any one model.

---

## 5. Which training set the layer labels are drawn from

The L1–L4 definition is "whether PocketAffDB has seen this target", and PocketAffDB is
the training set for only four of the ten models. Using this labelling to compare "models
trained on PocketAffDB" against "models trained on other sets", and then concluding
"training data matters more than architecture", is circular — the labels themselves were
drawn from one group's own training set.

First laying out the coverage of the three training sets against each layer
([`timesplit/analysis/per_model_audit.py`](../timesplit/analysis/per_model_audit.py),
350-quota subset, 328 entries / 293 targets):

| Layer | Targets | A seen | B seen | C seen | Seen by none of the three |
|---|---|---|---|---|---|
| L1 | 56 | 43 (77%) | 56 (**100%**) | 28 (50%) | 0 (0%) |
| L2 | 178 | 124 (70%) | 178 (**100%**) | 90 (51%) | 0 (0%) |
| L3 | 19 | 4 (21%) | 0 (**0%**) | 5 (26%) | 12 (63%) |
| L4 | 75 | 17 (23%) | 0 (**0%**) | 9 (12%) | 52 (69%) |

- **A** `train_no_test_af` (16,744 PDB → 4,098 UniProt) — DrugCLIP, BindCLIP ×2
- **B** the LigUnity family's training set — LigUnity ×2, LiTENCLIP, HypSeek
- **C** ConPLex's BindingDB training sequences (reverse-looked-up via mmseqs, identity
  ≥95%, bidirectional coverage ≥50%)
- **D** SPRINT's MERGED (336 UniProt)
- ConGLUDe's list is still unobtained

⚠️ **The B column above is computed against the affinity half (2,196 UniProt), which is
the layer definition itself.** Group B models actually train on the **union of 4,847**
(affinity half + structure half, and the structure half is exactly A). Recomputed against
the union, B's coverage is 21% at L3 and 11% at L4, not 0 — see §6. This table is kept
as-is because what it is meant to show is precisely "whose training set the layer
definition was drawn from".

The B column's 100/100/0/0 is the layer definition itself, not a measurement. What
actually matters is how much the other two columns deviate from it: **A is only 77% at
L1, but 21–23% at L3/L4.** In other words, for the three DrugCLIP-family models, a
quarter of what we call "L1" targets is something they haven't seen, and a fifth of what
we call "L4" targets is something they have. Their L1→L4 decay therefore also has a
"label mismatch" component mixed in.

C's shape (50/51/26/12), by contrast, is consistent with the layering intent, so
ConPLex's decay reading is unaffected by this issue.

**This does not overturn any existing conclusion** — the L1→L4 decay holds within every
model. As for whether the mismatch pushes decay up or down, the next section measures it
directly: **it is not one-directional** — DrugCLIP's own decay is 13.5 percentage points
higher than L1→L4, while BindCLIP-hardneg's is 13.4 points lower. So this is noise, not
bias — but when comparing absolute L4 values across models, it still needs to be kept in
mind that these labels were not drawn for them.

`T3_per_model_audit.csv` retains the raw per-layer counts.

---

## 6. Per-model layering: what does the work is the affinity-labelled half

⚠️ **This section was rewritten once on 2026-09-09.** The first version treated A
(`train_no_test_af`) and B (PocketAffDB) as two mutually exclusive training sets, and
concluded "PocketAffDB membership is worth about 2.4 rank places; `train_no_test_af`
membership brings no measurable advantage." **That premise was wrong** — see "the
training sets are actually nested" below. The measurement was real; the attribution was
not.

Taking the previous section's problem head-on: judging seen/unseen for each model against
**its own** training set
([`timesplit/analysis/per_model_layers.py`](../timesplit/analysis/per_model_layers.py)).

### The training sets are actually nested, not parallel

`unimol/tasks/train_task.py:523-524` reads **two** label files at once:

```python
pair_label_1 = json.load(open(".../train_label_pdbbind_seq.json"))     # structure half
pair_label_2 = json.load(open(".../train_label_blend_seq_full.json"))  # affinity half
```

We had previously only counted the latter. The former is not some separate thing — it
covers **16,744 PDB IDs**, and is **exactly the same** as DrugCLIP's `train_no_test_af`:

| | PDB ID count | Intersection | Each exclusively |
|---|---|---|---|
| `train_label_pdbbind_seq.json` | 16,744 | **16,744** | **0** |
| `train_no_test_af` | 16,744 | | **0** |

So the seven pocket-family models did not "use two different datasets":

- **Group A** (DrugCLIP, BindCLIP ×2) trained on those 16,744 structures
- **Group B** (LigUnity ×2, LiTENCLIP, HypSeek) trained on **the same 16,744**, plus the
  affinity half's 2,196 targets

⚠️ **What is nested is "the targets each group of models has seen", not "the two
files".** The two files intersect in only 817 UniProt at the UniProt level, with 1,379
exclusive to the affinity half and 2,651 exclusive to the structure half — neither
contains the other. So the "L-only" cell below is not empty — 36 in the 350 subset, 135
in the full set — and the analysis holds.

This also narrows finding 13's statement by one turn: not the broad claim "training data
matters more than architecture", but **the same batch of structures, once the affinity
half is added, takes L1 EF@1% from 17–19 up to 32–39**.

⚠️ **But what the "affinity half" brings is not only labels.** Its 428,767 unique
ligands, against the structure half's **13,590** — a **31.6×** larger chemical space —
arrive together with the pAff values. How much of the gain comes from the labels and how
much from ligand diversity cannot be separated using these two training sets alone;
separating them would require retraining with one variable held fixed. So this point is
about "that **half**", not "the labels themselves".

The four training sets in this section are: **A** the structure half, **B** the union of
the two files (4,847 UniProt), **C** ConPLex's BindingDB (reverse-looked-up via mmseqs),
**D** SPRINT's MERGED (336). ConGLUDe's list remains unobtained.

### Only done at the target level, not across all four layers

The L1/L2 boundary is whether the ligand scaffold has been seen; L3/L4 is whether the
family has been seen. The former needs each model's **training-ligand** list; the latter
needs re-running the clustering against each model's own training set — of the ten
models, only two or three have an obtainable ligand list. So the per-model version only
makes the one cut of "is the target in its training set or not".

⚠️ **This table's subset filter had a bug that was fixed once on 2026-09-09.** The 350
subset was originally filtered by **uniprot**, but the subset is actually **(layer,
target) pairs** — 328 entries / 293 unique uniprot, with 35 uniprot appearing in multiple
layers. Filtering by uniprot alone pulls in "that target's record in another layer" too —
seen+unseen was measured coming out to **417**, 89 more than the subset's own 328
entries. The table below uses the correct values after filtering by (layer, target).
**The conclusion is unchanged; the numbers shift slightly.**

| Model | Set | seen | unseen | EF1 seen | EF1 unseen | Own decay | L1→L4 decay |
|---|---|---|---|---|---|---|---|
| DrugCLIP | A | 187 | 128 | 15.53 | 7.07 | 58% | 51% |
| BindCLIP-hardneg | A | 187 | 128 | 14.64 | 7.49 | 52% | 71% |
| BindCLIP-randneg | A | 187 | 128 | 14.24 | 7.78 | 49% | 62% |
| HypSeek `_vs` | B | 234 | 75 | 24.26 | 8.46 | **68%** | 70% |
| LigUnity-pocket | B | 234 | 75 | 28.56 | 9.28 | **70%** | 73% |
| LigUnity-protein | B | 234 | 75 | 32.86 | 11.46 | 67% | 73% |
| LiTENCLIP | B | 234 | 75 | 24.25 | 8.81 | 66% | 73% |
| ConPLex | C | 128 | 170 | 5.34 | 2.32 | 70% | 75% |
| SPRINT | D | 22 | 241 | 2.11 | 2.02 | 8% *(p=0.85)* | 54% |

⚠️ Every column here now matches
[`results/T3_per_model_layers.csv`](../results/T3_per_model_layers.csv) cell for cell.
It did not before: the right-hand column used to read 45 / 69 / 59 / 72 / 69 / 70 / 69 /
67 / 29 in this document while the committed CSV held the values above — the table had
drifted from its own source at some earlier point, the same way the main tables had. Only
HypSeek's row changed for a reason connected to this update (it is now the `_vs` weight);
the rest were simply brought back into sync.

**After switching labels, groups A and B separate further apart.** Under the current
L1→L4 labels, group A sits at 51–71% and group B at 70–73%, overlapping; computed against
each group's own training set, **group A is 49–58% and group B is 66–70%**, not
overlapping. **The decay conclusion is not an artefact of the layer labels.**

### Target difficulty must be divided out, and it cannot be a ratio

Targets a model has seen are mostly well-studied ones, which are easy to begin with. The
first version normalised using the ratio `EF(model,t) ÷ median(EF(other models,t))`, and
**that criterion cannot be reported**: the ratio blows up when the denominator
approaches zero — DrugCLIP's median fold is 1.85 but its mean fold is 0.57, opposite
directions.

Switching to **within-target rank** — ranking the ten models by EF1 within each target;
rank is a within-target relative quantity, so target difficulty is automatically divided
out ([`per_model_seen_effect.py`](../timesplit/analysis/per_model_seen_effect.py)):

| Model | Set | P(seen&gt;unseen) | Rank seen | Rank unseen | Rank improvement |
|---|---|---|---|---|---|
| DrugCLIP | A | 0.693 | 5.66 | 5.77 | **+0.10** |
| BindCLIP-randneg | A | 0.665 | 5.95 | 5.91 | **−0.04** |
| BindCLIP-hardneg | A | 0.695 | 5.84 | 6.03 | **+0.19** |
| LigUnity-pocket | B | 0.816 | 2.97 | 4.94 | **+1.98** |
| LigUnity-protein | B | 0.823 | 2.74 | 4.47 | **+1.73** |
| HypSeek `_vs` | B | 0.795 | 4.17 | 5.64 | **+1.47** |
| LiTENCLIP | B | 0.776 | 4.16 | 5.03 | **+0.87** |
| ConPLex | C | 0.625 | 7.48 | 7.25 | **−0.23** |
| SPRINT | D | 0.450 | 7.00 | 7.92 | +0.92 *(p=0.78)* |

**The two criteria conflict, and the conflict itself is the answer.** Absolute EF says
group A is clearly higher on targets it has seen (P=0.67~0.70, p&lt;1e-4); within-target
rank says group A's relative position hasn't moved at all. **Group A's "higher when
seen" is because those targets are simply easier — every model scores higher there.**

⚠️ **But this null result for group A is weak evidence.** Since A ⊂ B, the targets group
A has seen are ones **all seven pocket models have seen**, so none of them can possibly
stand out there. This null is more likely a failure to detect than "structure data is
useless". Separating the two requires looking at targets covered by only one half — the
next subsection.

### Cutting cells by the two label files

([`train_set_crossover.py`](../timesplit/analysis/train_set_crossover.py), **237**
(layer, target) records where all ten models have a result)

| Cell | Meaning | Record count |
|---|---|---|
| P∩L | Both halves | 128 |
| **P-only** | Structure half only — **both A and B trained on it** | **7** |
| **L-only** | Affinity half only — **only B trained on it** | 42 |
| Neither | | 60 |

| Model | Group | P-only | L-only | L-only minus P-only | p |
|---|---|---|---|---|---|
| HypSeek `_vs` | B | 5.79 | 3.76 | **−2.02** | 0.039 |
| LiTENCLIP | B | 5.93 | 4.23 | **−1.70** | 0.044 |
| LigUnity-protein | B | 3.71 | 2.81 | **−0.90** | 0.150 |
| LigUnity-pocket | B | 3.64 | 2.79 | **−0.86** | 0.084 |
| ConGLUDe | ? | 7.14 | 7.19 | +0.05 | 0.49 |
| BindCLIP-randneg | A | 6.21 | 6.70 | +0.49 | 0.76 |
| ConPLex | C | 6.57 | 7.17 | +0.60 | 0.71 |
| SPRINT | D | 6.79 | 7.75 | +0.96 | 0.69 |
| BindCLIP-hardneg | A | 4.50 | 6.00 | +1.50 | 0.94 |
| DrugCLIP | A | 4.71 | 6.61 | +1.89 | 0.94 |

The between-group mean difference is **2.29 places** (group B −1.37, non-B +0.92). Exact
permutation test: of all C(10,4)=210 ways of picking four of the ten models to call
"group B", only **1** gives a between-group difference ≥ the observed value, **p =
0.0048** — the four affinity-trained models are exactly the most extreme split of the
ten. Reproducible with
[`finding18_permutation.py`](../timesplit/analysis/finding18_permutation.py), which also
re-derives the pre-2026-09-12 values from the archived CSVs.

> **On targets covered only by the affinity half, models trained on that half rank about
> 2.3 places stronger relative to the others.** On the structure half, which both groups
> trained on, no group stands out — which is exactly what should happen.

⚠️ **These are the `_vs` numbers.** HypSeek's row is the official screening weight, as in
the main tables since 2026-09-12; ranking is a within-target position among the ten
benchmark models, so `_rk` is excluded rather than ranked alongside it — keeping both
would rank eleven and count HypSeek twice. On `_rk` this table read −1.88 / −1.25 / −1.13
/ **+0.12** for group B and a between-group 1.73 places at p = 0.0095.

### Robustness: the full set of shared targets confirms the subset

The "P-only" cell holds only 7 records in the 350 subset, thin enough that the result
should not rest on it alone. Widening the scope to the **840 records shared by all ten
models** grows that cell to 28 (`train_set_crossover.py --subset all` →
[`results/T3_train_set_crossover_full.csv`](../results/T3_train_set_crossover_full.csv)):

| | 350 subset (n=237) | Full common set (n=840) |
|---|---|---|
| Four cells: P∩L / P-only / L-only / neither | 128 / **7** / 42 / 60 | 419 / **28** / 191 / 202 |
| Group B's "L-only minus P-only" | −2.02, −1.70, −0.90, −0.86 | −1.60, −1.30, −1.10, −0.67 |
| Is group B entirely negative | **Yes** | **Yes** |
| Between-group difference | **2.29 places** | 1.95 places |
| Exact permutation p | **0.0048 (1/210)** | **0.0048 (1/210)** |

**Across a threefold change in sample size the separation stays complete and p does not
move**; the effect size settles from 2.29 to 1.95 places as the thin cell fills out.
Nothing here rests on the small cell any more.

⚠️ **This section used to argue the other way round.** On `_rk`, group B was not entirely
negative on the subset — HypSeek sat at +0.12, and the full-set run was what removed that
exception (1.73 → 1.85 places, 2/210 → 1/210). Moving HypSeek's row to the screening
weight `_vs` puts it at −2.02, the strongest of the four, so the subset no longer needs
rescuing. The conclusion is unchanged; the argument for it got shorter.

### This set of numbers has been independently verified

Per the project's new rule (rewriting or retracting a published conclusion requires an
independent re-run by another agent), both the subset and full-set versions were
recomputed by a second agent from `saved_preds.npy` along a separate code path: the four
cell counts, the per-model values for all ten models, and the permutation p all matched.

### Three things that must go into the limitations

1. **The "P-only" cell has only 7 records in the 350 subset.** After the full-set version
   widens it to 28, but 28 is still thin. Group B is entirely negative on both, and
   the exact permutation p is 0.0048 on each.
2. **The layer labels themselves are contaminated.** In the 350 subset, 4 (21%) L3
   targets and 8 (11%) L4 targets were in fact seen by the four group-B models through
   the structure half. The direction is the same as that earlier fall-through bug: seen
   targets leaked into L4 → **decay is underestimated** → fixing it can only make the
   headline conclusion stronger.
3. **This section settles "whose training set was used", not "at what level similarity
   should be defined".** Seen/unseen uses exact UniProt matches and sequence identity,
   and sequence is the weakest layer of protein similarity — pocket-level and
   interaction-pattern-level similarity can still be high when sequence identity is
   &lt;30%. That is a separate, orthogonal follow-up experiment.

### One more instance of the same problem, unfixed

Ligand novelty (`ligand_novelty.json`) is computed against the **affinity half's 428,767
ligands**, while group A's training ligands are the molecules in those 66,164 pairs — **a
different set entirely**. So the novelty tiers in §4 and §6b are only approximate for the
three group-A models. Fixing it would require recomputing a novelty cache against group
A's ligands (CPU-only, roughly half a day) — not yet done.

---

## 6b. Target seen/unseen × ligand novelty: the two axes are approximately independent

Crossing §6's target-side criterion with §4's ligand-side criterion, judging seen/unseen
for each model against **its own** training set
([`novelty_tiered_ef.py --target-groups`](../timesplit/analysis/novelty_tiered_ef.py)
→ [`results/T3_novelty_by_own_train.csv`](../results/T3_novelty_by_own_train.csv)).

EF_t, where 1.0 = random; numbers in parentheses are the target counts included:

| Model | Group | Novel <0.35 | Distant .35–.5 | Close .5–.7 | Very close ≥0.7 | Novel/familiar ratio |
|---|---|---|---|---|---|---|
| LigUnity-protein | seen | 19.2 (119) | 26.4 (177) | 44.8 (173) | **53.1** (120) | 0.36 |
| | unseen | 5.2 (56) | 9.4 (69) | 19.0 (46) | 36.8 (29) | 0.14 |
| LigUnity-pocket | seen | 14.0 (119) | 23.4 (177) | 39.4 (173) | 48.3 (120) | 0.29 |
| | unseen | **7.9** (56) | 9.1 (69) | 15.6 (46) | 17.7 (29) | 0.44 |
| HypSeek `_rk` | seen | 11.2 (119) | 18.7 (176) | 36.8 (171) | 50.8 (118) | 0.22 |
| | unseen | 4.2 (56) | 6.5 (69) | 13.8 (46) | 23.2 (29) | 0.18 |
| LiTENCLIP | seen | 8.0 (119) | 19.4 (177) | 34.6 (173) | 44.5 (120) | 0.18 |
| | unseen | 4.7 (56) | 7.0 (69) | 12.4 (46) | 22.6 (29) | 0.21 |
| DrugCLIP | seen | 7.6 (99) | 13.1 (145) | 20.8 (143) | 21.4 (104) | 0.35 |
| | unseen | 4.9 (78) | 6.6 (106) | 6.9 (80) | 11.2 (47) | 0.44 |
| BindCLIP-randneg | seen | 5.7 (99) | 12.0 (145) | 20.3 (143) | 21.6 (104) | 0.26 |
| | unseen | 3.9 (78) | 6.5 (106) | 10.9 (80) | 15.1 (47) | 0.26 |
| BindCLIP-hardneg | seen | 4.6 (99) | 12.8 (145) | 21.2 (143) | 22.0 (104) | 0.21 |
| | unseen | 3.8 (78) | 7.0 (106) | 11.3 (80) | 10.8 (47) | 0.35 |
| SPRINT | seen | 0.2 (14) | 2.9 (19) | 1.6 (17) | 5.6 (12) | 0.04 |
| | unseen | 1.1 (153) | 2.2 (225) | 1.9 (207) | 2.9 (145) | 0.38 |

**Three readings.**

**One: the two axes are approximately independent, with no interaction term.** The
novel/familiar ratio's median is 0.26 in the seen group and 0.26 in the unseen group too
(excluding SPRINT, whose seen group has only 14–19 targets and is noise). The per-model
direction is also inconsistent — four models have a larger ratio in the unseen group,
three have a smaller one. So "novel target" and "novel ligand" are two approximately
**multiplicative** independent things, and the "amplification" in §4 was a single-model
phenomenon.

**Two: the worst cell is the genuinely prospective screening scenario.** Target unseen +
chemistry fully novel: the best model reaches 7.9, the mainstream sits at 4.2–5.2, the
DrugCLIP family at 3.8–4.9. The best cell (target seen + chemistry very close) is 53.1.
**Same model, same metric, a 12-fold span.**

**Three: SPRINT goes the other way.** Its novel tier is 0.2 on the 26 targets it has
seen, and 1.1 on the 302 it hasn't. This is consistent with its p=0.84 null effect in §6
— its training set is too small (336 proteins), and the sample size falling within the
subset is not enough to support any conclusion.

⚠️ **The ligand-side reference set is wrong for the three group-A models.** Novelty
tiers are computed against the affinity half's 428,767 ligands, while DrugCLIP /
BindCLIP ×2's training ligands number only 13,590. Measured against their own ligands,
72.6% of T3's molecules are "novel <0.35" and only 0.9% reach "very close ≥0.7" (see
§2c), so **the tier labels for group A's six rows in this table do not hold** — what
they call "very close" is still novel chemistry by their own standard. This also
implies: **the 21–22-fold enrichment group A gets in the "target seen + chemistry very
close" cell was in fact achieved on near-purely novel chemistry.**

The tiers for group B's four models are correct (the reference set is exactly their
training ligands). ConPLex has BindingDB's SMILES and SPRINT has 1.39 million ligand
IDs — neither has been done yet.

**ConPLex is absent**: its raw T3 scores are not under the `results/t3_raw/conplex/`
directory layout — this cell is empty, not because it has no result.

---

## 6c. NEW-4 formal interaction test: the two axes are additive, no interaction term

§6b used a descriptive 2×2 to say "the two axes are approximately independent", but that
cannot establish whether the interaction term is statistically zero. Task-list item 3.2
calls for a formal test:

```
Performance ~ TargetNovelty + LigandNovelty + TargetNovelty × LigandNovelty
```

### Why a binomial GLM rather than regressing on EF

EF_t has a large number of exact zeros (a tier's actives with none landing in the top
1%), and taking a log turns those into −inf; adding an epsilon then makes the result
depend on how epsilon is chosen. Going back to the definition of EF itself — it is
simply "the fraction of that tier's actives that land in the top 1% ÷ 0.01" — so instead
treat **whether each active lands in the top 1%** directly as a Bernoulli response and
fit a logit:

```
logit P(in top 1%) = β₀ + β₁·TargetSeen + β₂·LigandTier + β₃·(TargetSeen × LigandTier)
```

Zeros are handled naturally, with no transformation needed.

**Standard errors use target-level bootstrap (1,000 draws) rather than the GLM's own.** A
given target's actives are not independent (same series), and a GLM's independence
assumption would understate the standard error. The task list allows bootstrap or
mixed-effects; here, resampling with replacement is done by target, absorbing
within-target correlation into the resampling unit.

**The reference set is per-model.** Ligand novelty is computed against each model's own
training ligands — four separate caches, with training-ligand counts ranging from 3,814
(ConPLex) to 1,390,031 (SPRINT), a two-order-of-magnitude spread. Mixing them would make
the interaction term spurious.

### Results

([`novelty_interaction.py`](../timesplit/analysis/novelty_interaction.py) →
[`results/T3_novelty_interaction.csv`](../results/T3_novelty_interaction.csv))

| Model | Set | Cells | Novelty IQR width | β target | β ligand tier | **β interaction** | 95% CI | p |
|---|---|---|---|---|---|---|---|---|
| LigUnity-protein | B | 789 | 0.179 | +1.596 | +0.978 | −0.339 | [−0.881, +0.173] | 0.20 |
| LiTENCLIP | B | 789 | 0.179 | +1.299 | +0.917 | −0.162 | [−0.703, +0.378] | 0.55 |
| HypSeek `_rk` | B | 784 | 0.178 | +1.069 | +0.850 | −0.065 | [−0.548, +0.433] | 0.78 |
| LigUnity-pocket | B | 789 | 0.179 | +1.050 | +0.677 | −0.013 | [−0.461, +0.525] | 0.89 |
| DrugCLIP | A | 682 | **0.086** | +0.887 | +0.536 | −0.053 | [−0.566, +0.507] | 0.83 |
| BindCLIP-randneg | A | 682 | **0.086** | +0.809 | +0.730 | −0.211 | [−0.649, +0.195] | 0.34 |
| BindCLIP-hardneg | A | 682 | **0.086** | +0.724 | +0.782 | −0.272 | [−0.843, +0.271] | 0.34 |
| ConPLex | C | 639 | **0.066** | +1.054 | +1.468 | −0.478 | [−0.915, +0.015] | 0.06 |
| SPRINT | D | 765 | 0.113 | **−2.034** | +0.326 | **+1.146** | [+0.060, +2.006] | **0.046** ⚠️ |
| **Pooled** (+ model fixed effects) | | **6,601** | | **+0.940** | **+0.724** | **−0.038** | **[−0.292, +0.291]** | **0.87** |

> **Both main effects are strong, and the interaction term is zero.** Target novelty and
> ligand novelty are **additive** on the logit scale — each acts independently, neither
> amplifying nor cancelling the other. This confirms §6b's descriptive conclusion with a
> formal statistical test.

### Three limitations that must be read together

**One: agreement in sign across models is not evidence.** Eight of nine estimates are
negative and one positive, which looks like "there is a small negative interaction masked
by insufficient power". **But these nine estimates run on the same 264–323 targets and
are highly correlated with each other — they are not nine independent replicates.**
Running a sign test on them would produce a spurious p (about 0.04). **The pooled model's
CI covering zero is the only citable conclusion.**

**Two: group A and ConPLex have only half of group B's spread in the independent
variable.** The IQR width of active novelty: group B 0.179, SPRINT 0.113, group A 0.086,
ConPLex 0.066. ConPLex's training ligands number only 3,814, and 87.9% of T3's molecules
fall in the lowest tier with only 0.5% in the highest. **Their wide CIs should be read as
"insufficient power", not "no interaction".**

**Three: SPRINT's p=0.046 cannot be cited on its own.** It is one of nine tests (fails BH
correction), and its β_target is **−2.034** — it actually does worse on targets in its
own training set, while it has only **22 seen targets** in the 350 subset. All three
coefficients for this model are unreliable.

### Independent verification

Per project rules, this set of numbers was recomputed by a second agent using a
**completely different estimator**: directly computing the difference in log-odds ratios
after collapsing the four tiers into two, straight from
`T3_novelty_interaction_cells.csv`, with no bootstrap and no target-cluster weighting.
The per-model signs match exactly (including SPRINT being the sole positive one), and the
pooled interaction is −0.271, falling within this document's CI of [−0.292, +0.291]. The
difference in magnitude comes from the different estimator; direction and pattern agree.

---

## 7. Not yet done

- **Per-model layering**: re-cut L1–L4 for DrugCLIP/BindCLIP against A and for ConPLex
  against C respectively, and compute each one's own decay. The data all exists — it's
  just a matter of re-aggregating under different labels
- Replace L1/L2's binary scaffold-seen/unseen boundary with continuous Tanimoto tiers
