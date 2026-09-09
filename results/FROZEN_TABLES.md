# Frozen intermediates: re-cut the layering without re-running anything

## Why this exists

"Has this model seen this target?" is not a fact about the target. It is a fact
about *that model's training set*, and the ten models here do not share one. So
the L1–L4 layering is not a fixed property of the dataset — it is a labelling
that has to be redrawn every time our knowledge of a training set improves.

It has been redrawn **three times already**:

| | what changed | effect |
|---|---|---|
| CD-HIT fall-through bug | 24% of L4 had never been checked | decay −69% → −78% |
| Per-model layering | one shared label replaced by each model's own training set | the two model families separate further |
| Structure/affinity halves | LigUnity reads *two* label files; only one was counted | 27 L4 and 8 L3 targets had in fact been trained on |

Each redraw costs either **minutes** or **a day of GPU**, depending entirely on
what was kept. These three tables make it minutes: every re-cut becomes a
`group by` over data that already exists, and **no model is ever re-run**.

## The three tables

### `T3_molecules.csv.gz` — one row per unique molecule

```
mol_id, inchikey, smiles, novelty_pocketaffdb, novelty_drugclip
```

The two novelty columns are the maximum ECFP4 (Morgan r=2, 2048-bit) Tanimoto to
each of the two training ligand sets — the affinity half's 428,767 ligands and
`train_no_test_af`'s 13,590. **Both are needed**: measuring the DrugCLIP family
against the affinity half's ligands is measuring them against somebody else's
data, which inverted one conclusion before it was caught
([T2](../tasks/T2-affinity-ranking.md)).

### `T3_index_L{1,2,3,4}.csv.gz` — one row per (target, molecule)

```
layer, uniprot, jsonl_pos, lmdb_pos, mol_id, label, paff
```

**`lmdb_pos = -1` means the molecule was never scored.** The eval JSONL is the
*designed* candidate pool; the LMDB is what the models actually saw. Building the
LMDB drops molecules RDKit cannot parse or embed, so 413 of the 546 L1+L4 targets
(75.6%) have a JSONL pool 1–78 molecules larger than the LMDB — **2,495 molecules
in all, 0.03%**. It changes no conclusion, because every analysis validates the
molecule order and therefore falls through to the LMDB branch. But it does mean
the two are not interchangeable:

- Counting pool size, actives or decoys from the JSONL gives the **designed**
  pool, which is what `T3_targets.csv` reports.
- Any analysis over molecules — a decoy background distribution, a novelty
  histogram, anything joined to scores — should **filter `lmdb_pos >= 0`**, or it
  will include molecules that were never scored.

**`jsonl_pos` and `lmdb_pos` are the point of this file.** A model's score array
is ordered either the way the eval JSONL is (actives, then decoys) or the way an
LMDB cursor walks its string keys (`0, 1, 10, 100, …`). The two orders hold the
same molecules and have the **same length**, so comparing lengths — the obvious
check — passes on both and silently mispairs every molecule with somebody else's
score.

That is not hypothetical. It flattened T2's correlations to zero and stood as a
published conclusion for two days; it then recurred twice more in other analyses
([`PATCHES.md`](../PATCHES.md)). Every analysis since has had to reconstruct and
re-validate the order for itself. **Storing both positions ends that.**

Split per layer only to keep every file under 50 MB; 7,945,632 rows in total
(L1 870k · L2 5.18M · L3 222k · L4 1.67M).

### `T3_model_order.csv` — which order each model actually used

```
model, layer, uniprot, n_molecules, order_used   # jsonl | lmdb | FAIL
```

Determined by the only check that works: reconstruct the order, then verify that
every position labelled 1 really holds one of that target's actives. `FAIL` means
neither order reproduces the labels — those targets must be dropped, not guessed
at.

The split turns out to be exactly along the UniMol / non-UniMol line, with no
ambiguous cases at all:

| Models | jsonl | lmdb | FAIL |
|---|---|---|---|
| HypSeek, LigUnity ×2, LiTENCLIP, DrugCLIP, BindCLIP ×2 | **0** | 896–925 | **116–119** |
| ConGLUDe, ConPLex, SPRINT | 903–1062 | **0** | **0** |

The seven UniMol-based models never use the JSONL order, and each has 116–119
targets that neither order reproduces. Those are the same targets every analysis
has been skipping — HypSeek's 116 is exactly the 46 + 30 + 9 + 31 that
`novelty_tiered_ef.py` and `t2_novelty_tiers.py` report per layer. They are now
recorded once instead of being rediscovered each time.

## Re-cutting the layering: worked example

Suppose a training-set manifest improves and some L4 targets should become L1.
Nothing needs re-running:

```python
import gzip, csv, math, collections
import numpy as np

# 1. the new labelling — whatever it is
new_layer = {...}                      # uniprot -> "L1" | ... | "L4"

# 2. which order this model used, per target
order_used = {(r["layer"], r["uniprot"]): r["order_used"]
              for r in csv.DictReader(open("results/frozen/T3_model_order.csv"))
              if r["model"] == "hypseek_rk"}

# 3. walk the index once, keeping the position column that applies
rows = collections.defaultdict(list)
with gzip.open("results/frozen/T3_index_L1.csv.gz", "rt") as f:
    for r in csv.DictReader(f):
        k = (r["layer"], r["uniprot"])
        if order_used.get(k) not in ("jsonl", "lmdb"):
            continue                                   # FAIL: drop the target
        pos = int(r["jsonl_pos"] if order_used[k] == "jsonl" else r["lmdb_pos"])
        rows[k].append((pos, int(r["label"])))

# 4. join to the scores and aggregate under the new labels
z = np.load("T3_hypseek_rk.npz")                       # see RAW_SCORES.md
ef = collections.defaultdict(list)
for (layer, up), mols in rows.items():
    preds = z[f"{layer}/{up}/preds"]
    lab = np.zeros(len(preds), dtype=int)
    for pos, y in mols:
        lab[pos] = y
    k = math.ceil(0.01 * len(preds))                   # ceil, not round
    top = np.argsort(-preds)[:k]
    ef[new_layer[up]].append((lab[top].sum() / k) / (lab.sum() / len(lab)))

print({L: float(np.mean(v)) for L, v in ef.items()})
```

Two conventions that silently change results, both settled in
[`eval/metrics.py`](../eval/metrics.py): enrichment cutoffs use `math.ceil`, and
`r2_score` there means Pearson r², not 1 − SS_res/SS_tot. Use that module rather
than a fresh implementation.

## What is not here

**The scores themselves.** They are 267 MB of `.npz` and are described in
[`RAW_SCORES.md`](RAW_SCORES.md). These three tables are the *keys* — identity,
position, label, affinity, novelty — which is the part that was fragile and
kept being re-derived. The scores are bulky but have never been in doubt.

## Rebuilding

```bash
python timesplit/analysis/freeze_t3.py --out-dir results/frozen
```

Reads the eval JSONL, the pocket LMDBs, both novelty caches and every model's
saved labels. Deterministic; ~20 minutes on CPU.
