# Raw per-molecule scores

Every number in `results/*.csv` is derived from two arrays per target: the
model's score for each molecule, and that molecule's label. Those arrays are
what you need to recompute a metric we did not report, apply a different
cutoff, or check ours.

They are **not committed** — 380 MB of `.npz` across 22 packages, too large for
this repository. They are packaged and checksummed, and available on request;
[`../DATA_RELEASE.md`](../DATA_RELEASE.md) is the index for all three tiers and
says what each one does and does not let you do. You can also rebuild them from
a run with [`../eval/pack_raw.py`](../eval/pack_raw.py).

**T3: 12 packages, 256.9 MB** — all ten published models, both HypSeek weights,
and **two for LigUnity-pocket**: `T3_ligunity_pocket_ranking.npz` is the current
checkpoint and reproduces every table on the 350-target convention;
`T3_ligunity_pocket_ranking_ckpt1.npz` is the first checkpoint and reproduces the
full 1,144-target auxiliary tables, which were not re-run
([`../MODELS.md`](../MODELS.md) records why).
⚠️⚠️ **The archive handed out on 2026-09-13 does not match this description.**
Its `T3_ligunity_pocket_ranking.npz` is the **first** checkpoint's scores under the
current checkpoint's name — verified by sha256 (`d0034aaf294eca0b…`, which is this
manifest's `_ckpt1` entry, where the current-checkpoint entry is
`881f65efd7b205ec…`), and `_ckpt1.npz` is not in that archive at all. The archive
is self-consistent — its own `manifest_T3.json` records the `_ckpt1` hash — so an
external checksum check **passes** and the recomputed LigUnity-pocket numbers come
out at the first checkpoint's values (L1 34.78 / L4 7.90 instead of 35.28 / 9.47).
Cause: the archive was packed on 09-13 and the re-run landed on 09-15, so the CSVs
moved and the score packages did not.

✅ **Fixed on 2026-09-15.** The archive now carries the current checkpoint under
that name (`881f65ef…`), keeps the first checkpoint as
`T3_ligunity_pocket_ranking_ckpt1.npz` (`d0034aaf…`), and both manifests were
rewritten to match the files on disk. **A copy fetched before 2026-09-15 still
has the old file under the current name** — re-fetch those two, nothing else
changed.

⚠️ Scripts that take a `--raw` directory load `T3_ligunity_pocket_ranking.npz`
**by name**, so they reproduce the 350-convention numbers as shipped. To re-derive
a *full-set* LigUnity-pocket table (`T3_main.csv`, `T3_actives_gradient.csv`, the
other 1,144-target tables), put the `_ckpt1` file under that name first — otherwise
the run silently mixes the two checkpoints: current weights on the 309 subset
targets, first weights on the other 707.
**T1: 10 packages, 123.5 MB.** LigUnity-pocket,
LigUnity-protein and LiTENCLIP used to be absent because their upstream code
writes only the embedding and never the score, so no file was ever created to
keep. `timesplit/runners/patch_t1_save_preds.py` adds that save and the three
were re-run; `timesplit/analysis/verify_t1_preds.py` checked the result against
`T1_main.csv` before anything was copied into place.

⚠️ **One caveat if you recompute LIT-PCBA from these files.** The re-run
reproduces the published table exactly on DUD-E and DEKOIS — every one of the
six model × benchmark cells agrees to within 5e-5 on EF@1%, EF@5%, BEDROC and
AUROC. On LIT-PCBA it agrees to ~1e-3 instead: the largest gap is
LigUnity-protein's EF@5%, 2.1821 recomputed against 2.1807 published (0.06%).
The difference is confined to the two biggest targets — `VDR` (356k molecules)
and `ALDH1` (145k) — where EF@5% is decided at the top-17,800 boundary and fp16
inference places a handful of molecules differently between runs; 13 of the 15
LIT-PCBA targets reproduce to the last digit. **The published table is the
original run and has not been changed**, so a recomputation from these score
files will differ from it in the third or fourth decimal on LIT-PCBA only.

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
ef = [enrichment_factor(z[f"{t}/preds"], z[f"{t}/labels"], 0.01)
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
