# Patches and bugs

Two lists. The first is what had to be changed in third-party code to run these
models at all; the second is bugs in **our own** analysis code, including two
that produced a wrong conclusion before they were caught.

The second list is here on purpose: it is the honest answer to "why should I
believe these numbers".

---

## Part 1 — changes to third-party code

Every change is either "make it save its raw output" or "make it run". No
modelling logic was altered, and every model uses its own official weights.

### Persist raw per-molecule scores

Several repos print aggregate metrics and discard the scores. Unified evaluation
is impossible without them.

| Repo | Patch | What it adds |
|---|---|---|
| DrugCLIP, BindCLIP | [`timesplit/runners/patch_t3_task.py`](timesplit/runners/patch_t3_task.py) | `saved_preds.npy` + `saved_labels.npy` per target |
| LigUnity, LiTENCLIP, HypSeek | [`timesplit/runners/patch_ligunity_t3.py`](timesplit/runners/patch_ligunity_t3.py) | same, plus registration of the T3 task |
| the FEP branch of those three | [`physics/patch_fep_save.py`](physics/patch_fep_save.py) | one line; the branch stored only embeddings |

Validation that the patch is faithful: DrugCLIP on DEKOIS reproduces the
baseline reported in the LigUnity paper to **0.0%**.

For runs that predate the FEP patch, scores are reconstructed rather than
re-run: [`physics/fep_recover_preds.py`](physics/fep_recover_preds.py) computes
`pocket_emb @ mol_emb.T` then takes the max over pockets — byte-identical to the
official computation.

### `bsz = 64` hardcoded, `--batch-size` inert

[`timesplit/runners/fix_bsz.py`](timesplit/runners/fix_bsz.py)

The T3 task code was adapted from the DEKOIS task and carried its `bsz = 64`
along, so the command-line batch size did nothing. Symptom: constant OOM on
large targets. **Detected because the OOM allocation sizes were byte-identical
after lowering the batch size** — a genuinely inert flag, not a too-small
reduction. Target failure rate went from 70% to **2.4%**.

### `--test-task` is a multi-line `add_argument`

The first version of the task-registration patch inserted a new argument with a
regex that matched the opening of `add_argument(`, landing the insertion *inside*
its argument list → `SyntaxError`. The patch now balances parentheses before
choosing an insertion point.

### ConGLUDe: documentation and code disagree on matrix orientation

The README states rows are proteins; the code computes `ligands @ proteins.T`.
Following the README gives an `IndexError` (or, with square inputs, silently
transposed results). [`timesplit/runners/run_t3_conglude.py`](timesplit/runners/run_t3_conglude.py)
asserts the shape explicitly rather than trusting either source.

### SPRINT: three fixes, one unsolved

[`timesplit/runners/run_t3_sprint.py`](timesplit/runners/run_t3_sprint.py)

1. **Hardcoded HuggingFace URL** — unreachable from this machine; redirected to
   a mirror via `HF_ENDPOINT`.
2. **The TSV reader guesses the delimiter.** `pd.read_table(sep=None)` sees the
   header `Target Sequence`, decides the delimiter is a space, splits the column
   name in two, and raises `KeyError`. Fixed by writing a second tab-separated
   column so the sniffer commits to tabs.
3. **File-descriptor leak** — `transform()` creates a multiprocessing `Pool` per
   call, and PyTorch tensors cross the boundary as `DupFd`. Fixed by batching.
4. **What looked unsolvable was three more bugs, two of them ours.** At ~146,000
   molecules the run died with `Too many open files`, and this was recorded as a
   scale limit. It was not:

   - **Chunking had been silently disabled.** `run_embed`'s default read
     `chunk=20000**9` — about 5×10³⁸ — so `len(rows) <= chunk` was always true
     and the entire molecule set went to a single subprocess. Chunking is what
     releases file descriptors (each chunk is its own process), so the leak ran
     unbounded. Restored to 10,000.
   - **`--num-workers 0` is the worst possible value here.** `featurizers.py`
     reads `n_jobs if n_jobs > 0 else multiprocessing.cpu_count()`, so 0 means
     *104 workers on this machine*, and every featurize call spawned and
     destroyed 104 processes. Load average 8 with no progress for ten hours.
     The earlier DEKOIS run worked because it passed 4. Now fixed at 8.
   - **A cached feature shape mismatch.** A minority of entries come back as
     `[1, 2048]` instead of `[2048]`, and `torch.stack` refuses to mix them.
     They are the same vector; the collate function now squeezes the leading
     axis.

   The FD limit itself was also part of it — the machine's soft limit is 1024
   against a hard limit of 1048576 — so the runner now raises its own limit at
   import instead of depending on the launcher remembering `ulimit`.

   Worth stating plainly: "SPRINT cannot scale past ~150k molecules" was our
   conclusion for two weeks, and it was wrong in a way that cost a 20-hour run
   producing nothing. A job that burns CPU while writing no output is not making
   progress, and the feature cache's mtime would have shown that on day one.

Also worth recording: SPRINT is **not** a sequence-only model. It consumes
SaProt structure-aware sequences (amino acid + foldseek 3Di tokens), so it needs
the same structures as the pocket models —
[`timesplit/structure/gen_saprot_seqs.py`](timesplit/structure/gen_saprot_seqs.py).

### Conformer generation: three failure modes

[`timesplit/build/gen_conformers.py`](timesplit/build/gen_conformers.py),
[`resume_conformers.py`](timesplit/build/resume_conformers.py)

1. 12 molecules produced **empty InChIKeys** → `lmdb.BadValsizeError` on a
   zero-length key. Fallback to an md5 of the SMILES.
2. The whole run was one LMDB transaction, so a crash lost everything. Now
   committed in batches.
3. RDKit ETKDG **hangs** on some molecules — not slow, stuck. The resume script
   runs each molecule in a subprocess with a timeout. `mp.Pool` workers are
   daemonic and cannot spawn children, so it uses `ProcessPoolExecutor`.

### Build environment

- **Uni-Core's `setup.py` imports torch**, so it needs `--no-build-isolation`.
- **`transformers` newer than the pinned torch breaks imports** — hit twice, in
  two environments (ConPLex → 4.36.2 for torch 2.1; HypSeek → 4.44.2 for torch
  2.4). The traceback points nowhere near the cause.

See [`env/`](env/).

---

### HypSeek saves LIT-PCBA embeddings but not the scores it just computed

`test_pcba_target` computes `res_single = alpha_poc * poc_scores + alpha_prot *
prot_scores`, passes it to `cal_metrics`, and then saves the two embedding
matrices and the labels — but not `res_single`. A sibling function,
`test_pcba_target_regression`, does save it, which is why some result
directories have `saved_preds.npy` and others do not. Any scorer that reads
scores from disk reports "no usable results" for the retrieval path.

One line added after the label save:

```python
np.save(f"{out_dir}/saved_preds.npy", res_single)
```

## Part 2 — bugs in our own code

### The one that reached the README

**Molecule order in the LMDB is lexicographic, not numeric — and every analysis
that joined external per-molecule data got it wrong.**

`build_t3_unimol.py` writes ligands into LMDB with `str(i)` as the key. The model
side reads them back **by cursor**, and a cursor over string keys returns
`0, 1, 10, 100, 1000, …`. Our analysis scripts read the same LMDB **by numeric
index**, so from the model's row *k* we recovered a different molecule.

It only affects analyses that join the score array to something outside it —
affinity values, assay types, contamination flags. Metrics computed from
`(scores, labels)` alone — EF, BEDROC, AUROC, everything in T1/T3/T5 — are
untouched, because both arrays come from the model in the same order.

**What it produced:** T2 read as ρ ≈ 0 on T3 for all seven UniMol-family models,
and a paired test appeared to show ranking collapsing from +0.41 to −0.00
(p = 0.0001) between congeneric and cross-database ligands. Corrected: ρ = +0.09
to +0.26 at L1, and the paired difference is +0.41 vs +0.29 at p = 0.27 — no
significant difference at all.

**How it was caught.** Not by review — by a consistency check written for a
different purpose. While preparing the rerank shortlists, the number of actives
in the top 50 was computed two ways: from the model's own label array (269) and
from our molecule mapping (36). A 7× disagreement in a quantity that should be
identical.

**The tell that was ignored for weeks:** ConGLUDe had the highest T3 ranking
correlation of any model (+0.129 when structure models sat at 0.00–0.03), and
ConGLUDe is the one model whose runner iterates the eval JSONL directly instead
of reading the LMDB. An unexplained winner that differs from the pack in exactly
one implementation detail is worth a look.

**The check that now exists:**
[`timesplit/analysis/verify_order.py`](timesplit/analysis/verify_order.py)
compares active positions derived from SMILES against the model's label array
and reports agreement — 99.8% under cursor order, ~10% under numeric order. Any
script that joins per-molecule data should be run behind it.

A second, smaller bug lived in the same code: actives were identified by matching
SMILES *strings* against the eval set. LMDB SMILES come from the conformer cache
and are canonicalised differently, so the match silently failed. Identity now
goes through the model's own label array, and chemistry through InChIKey.

### Two that produced a wrong conclusion

**BH-FDR applied per row instead of as a step-up procedure.** Correcting each
p-value independently instead of running Benjamini–Hochberg over the sorted list
would have discarded the kinase finding as non-significant. It survives correct
correction.

**Reversed bin labels in the Boltz-2 affinity analysis.** `argsort(-pred)`
returns ascending order of the negated array; the quintiles were labelled
backwards, making the binned table appear to contradict the correlation
computed three lines above. The contradiction was the tell.

### Silent path bugs: exit code 0, plausible logs, wrong output

Four found in one sweep. None raised an error; each would have put a wrong
number in front of a reader.

**A runner hardcoded to seed 1 while being called for seed 2.**
`run_t3_hypseek_vs.sh` took the GPU as `$1` but had both the checkpoint path and
`--results-path` written out as `hypseek_vs_seed1`. The scheduler called it for
seed 2, so it loaded seed 1's weights and overwrote seed 1's output directory —
then logged "seed=2 T3 done". Seed 2's T3 would never have existed, and nothing
in the logs said so. The script now takes `$2` as the seed and writes to
`hypseek_vs_s$SEED`. Before running any variant sweep, grep the runner for
hardcoded paths.

**Two published CSVs were stale, one of them carrying retracted numbers.**
`results/T2_on_T3.csv` was regenerated on 08-26 by the *pre-fix* scorer, so the
repository published ρ = +0.003 for HypSeek at L1 — the exact value this file
documents as a bug — five days after the corrected value (+0.260) had been
computed. Separately, `results/T3_main.csv` had been overwritten by an export
run that covered a single model, leaving 1 of 10 models in the main table.
Both now have dedicated export scripts
([`physics/export_t2.py`](physics/export_t2.py),
[`timesplit/analysis/export_t3.py`](timesplit/analysis/export_t3.py)) that read
the authoritative summary JSON, so the table can always be rebuilt. `export_t2.py`
also emits a `spearman_old_retracted` column so the two are visible side by side.

A useful self-check fell out of it: ConGLUDe and ConPLex have identical old and
new values, and they are exactly the two models whose runners iterate the eval
JSONL instead of the LMDB — so they were never exposed to the ordering bug.

**A progress counter whose pattern could never match.** The docking runner
logged `grep -c '^   1 '` against smina's output to report how many ligands had
been scored. smina writes the mode table flush-left (`1       -7.5`), so the
count was 0 for every target from the first one onward. Sixteen targets' worth
of correct results were logged as total failures.

**Partial results silently truncated instead of flagged.** `score_dock.py`
joined scores to the manifest with `n = min(len(aff), len(info))`. Targets killed
by the 90-minute timeout have a *non-random* subset of ligands — the fast ones,
which are the small ones — so enrichment computed over that prefix is biased
upward. It now records coverage per target, warns explicitly, and reports the
summary twice: all targets, and complete targets only.

**The common shape.** All four produced clean exits and plausible logs. The
lesson recorded under the LMDB bug applies here too: the artifacts that look
like progress — an exit code, a directory entry count, a log line — are not
evidence that the right computation ran. Directory entry counts are a
particularly bad signal, because directories are created before their contents
are written.

### Metric bugs

**Enrichment cutoffs must use `ceil`, not `round`.** RDKit's `CalcEnrichment`
uses `math.ceil(numMol * fraction)`. The synthetic tests passed under either
convention because they used sizes of 300/500/1000/2000, for which `n × fraction`
is always an integer. The discrepancy only appeared on real data — a DUD-E target
with 2343 molecules gives `23.43`, so `round` → 23 and `ceil` → 24, affecting 37
of 102 targets. *Synthetic tests with round-number sizes do not exercise rounding
logic.* Test sizes now include 2343, 9448, 1207, 4247.

**Ties resolved by average rank.** Score ties are common; relying on `argsort`
stability makes EF@1% depend on the sorting algorithm, which on tied data can
change it several-fold.

**`r2_score` is Pearson r², not `1 − SS_res/SS_tot`.** Model outputs are cosine
similarities and measured values are log-molar; a regression R² between them is a
large negative number that means nothing. The docstring says so, because the
name invites the wrong assumption.

### Data-construction bugs

**Chain assignment in multi-protein PDB entries.** 47.1% of entries contain more
than one UniProt. Using every chain in the file lets a ligand bound to subunit A
become "the pocket" of subunit B. Protein atoms must be restricted to the
target's own chains, and if the ligand does not contact them, that (PDB, ligand)
pair is invalid for that target — the next candidate has to be tried. Fixing this
removed **153 targets** that previously had bogus pockets.
[`timesplit/structure/extract_pocket_pdb.py`](timesplit/structure/extract_pocket_pdb.py)

**Co-crystal ligand selection.** Ranking candidates by raw Tanimoto lets a 138 Da
fragment win by a 0.02 margin and define a pocket covering a fraction of the real
site. Ranking by size instead picks cardiolipin and other membrane lipids, which
mark the transmembrane face. The working rule buckets similarity into 0.1 bins,
prefers a drug-like MW window within a bucket, and blocks ions, buffers,
lanthanide phasing atoms, lipids, detergents and glycans — while **keeping**
nucleotide cofactors, since for kinases and methyltransferases those are the drug
site. [`timesplit/structure/rank_crystal_ligands.py`](timesplit/structure/rank_crystal_ligands.py)

**Domain truncation followed construct length, not binding sites.** Picking the
widest PDB construct for over-length proteins is unrelated to where ligands bind:
validation against UniProt annotations found **33% of truncations contained no
annotated site**. Ranking candidate regions by site coverage first brings that to
0%. [`timesplit/structure/truncate_domains2.py`](timesplit/structure/truncate_domains2.py),
[`validate_truncation.py`](timesplit/structure/validate_truncation.py)

### Operational

**A duplicate launch put six processes on three GPUs** (a manual start plus the
same script started by a tmux session). Three were killed, and
[`timesplit/analysis/verify_t3_raw.py`](timesplit/analysis/verify_t3_raw.py) was written to
check every raw output for truncation or interleaved writes. Zero corruption
found — but the check exists because the possibility was real.

---

## What this list is for

Several of these were caught because a number looked *slightly* wrong rather than
obviously wrong: identical OOM sizes, a table contradicting the correlation above
it, a fragment-sized pocket, a 0.02 similarity margin. The pattern is that
plausible-looking output is the dangerous kind, which is why the analysis scripts
state how to read a null result in their docstrings before printing anything.
