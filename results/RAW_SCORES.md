# Raw per-molecule scores

Every number in `results/*.csv` is derived from two arrays per target: the
model's score for each molecule, and that molecule's label. Those arrays are
what you need to recompute a metric we did not report, apply a different
cutoff, or check ours.

They are **not committed** — 267 MB of `.npz` across 15 packages. Ask for the
archive, or rebuild it from a run with
[`../eval/pack_raw.py`](../eval/pack_raw.py).

## What is in a package

One `.npz` per model per task. Keys are paths:

```
T3_hypseek_rk.npz
  T3/L1/C7C422/preds     float32, one score per molecule
  T3/L1/C7C422/labels    int8, 1 = active
  T3/L1/O00141/preds
  ...
```

Scores are stored as float32 (originals are float64; every metric here is
identical to four decimals either way) and labels as int8.
[`raw_scores_manifest.json`](raw_scores_manifest.json) lists each package with
its target count and size.

## Recomputing a metric

```python
import numpy as np
from eval.metrics import enrichment_factor, bedroc

z = np.load("T3_hypseek_rk.npz")
ef = [enrichment_factor(z[f"{t}/labels"], z[f"{t}/preds"], 0.01)
      for t in {k.rsplit("/", 1)[0] for k in z.files}]
print(np.mean(ef))
```

Use [`../eval/metrics.py`](../eval/metrics.py) rather than a fresh
implementation — it is the same code behind every table here, it has 80 tests
against RDKit, and it settles the two conventions that silently change results:
enrichment cutoffs use `math.ceil`, and `r2_score` here means Pearson r²
rather than 1 − SS_res/SS_tot.

## Two things to know before joining anything to these arrays

**Molecule order is the LMDB cursor order, which is lexicographic**
(`0, 1, 10, 100, …`), not numeric. Reading the dataset by numeric index pairs
each score with the wrong molecule. This produced two wrong conclusions before
it was caught; see [`../PATCHES.md`](../PATCHES.md).

Any script that joins these scores to information from outside them — affinity
values, assay types, contamination flags — should call the guard first:

```python
from eval.order_guard import assert_cursor_order
assert_cursor_order()      # raises MoleculeOrderError below 95% agreement
```

It reports ~99.8% agreement under cursor order and ~10% under numeric order, so
the failure is unambiguous. `python eval/order_guard.py` exits non-zero on
failure if you want it in a pipeline.

**Identity should go through the label array, not SMILES strings.** The SMILES
in the LMDB come from the conformer cache and are canonicalised differently from
the eval set, so string matching silently fails. Take actives from `labels`, and
use InChIKey when you need chemical identity.

## What is not here

- Embeddings. Some runs saved pocket/molecule/protein representations; those are
  much larger and are kept on the compute host.
- Models whose runners never persisted per-molecule scores on a given benchmark.
  The manifest is the authoritative list of what exists.
