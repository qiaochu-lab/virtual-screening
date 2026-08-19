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
| DrugCLIP, BindCLIP | [`t3/runners/patch_t3_task.py`](t3/runners/patch_t3_task.py) | `saved_preds.npy` + `saved_labels.npy` per target |
| LigUnity, LiTENCLIP, HypSeek | [`t3/runners/patch_ligunity_t3.py`](t3/runners/patch_ligunity_t3.py) | same, plus registration of the T3 task |
| the FEP branch of those three | [`physics/patch_fep_save.py`](physics/patch_fep_save.py) | one line; the branch stored only embeddings |

Validation that the patch is faithful: DrugCLIP on DEKOIS reproduces the
baseline reported in the LigUnity paper to **0.0%**.

For runs that predate the FEP patch, scores are reconstructed rather than
re-run: [`physics/fep_recover_preds.py`](physics/fep_recover_preds.py) computes
`pocket_emb @ mol_emb.T` then takes the max over pockets — byte-identical to the
official computation.

### `bsz = 64` hardcoded, `--batch-size` inert

[`t3/runners/fix_bsz.py`](t3/runners/fix_bsz.py)

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
transposed results). [`t3/runners/run_t3_conglude.py`](t3/runners/run_t3_conglude.py)
asserts the shape explicitly rather than trusting either source.

### SPRINT: three fixes, one unsolved

[`t3/runners/run_t3_sprint.py`](t3/runners/run_t3_sprint.py)

1. **Hardcoded HuggingFace URL** — unreachable from this machine; redirected to
   a mirror via `HF_ENDPOINT`.
2. **The TSV reader guesses the delimiter.** `pd.read_table(sep=None)` sees the
   header `Target Sequence`, decides the delimiter is a space, splits the column
   name in two, and raises `KeyError`. Fixed by writing a second tab-separated
   column so the sniffer commits to tabs.
3. **File-descriptor leak** — `transform()` creates a multiprocessing `Pool` per
   call, and PyTorch tensors cross the boundary as `DupFd`. Fixed by batching.
4. **Unsolved:** at ~146,000 unique molecules (L1/L2) the loader still exhausts
   shared memory. See [`LIMITATIONS.md`](LIMITATIONS.md) §7.

Also worth recording: SPRINT is **not** a sequence-only model. It consumes
SaProt structure-aware sequences (amino acid + foldseek 3Di tokens), so it needs
the same structures as the pocket models —
[`t3/structure/gen_saprot_seqs.py`](t3/structure/gen_saprot_seqs.py).

### Conformer generation: three failure modes

[`t3/build/gen_conformers.py`](t3/build/gen_conformers.py),
[`resume_conformers.py`](t3/build/resume_conformers.py)

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

## Part 2 — bugs in our own code

### Two that produced a wrong conclusion

**BH-FDR applied per row instead of as a step-up procedure.** Correcting each
p-value independently instead of running Benjamini–Hochberg over the sorted list
would have discarded the kinase finding as non-significant. It survives correct
correction.

**Reversed bin labels in the Boltz-2 affinity analysis.** `argsort(-pred)`
returns ascending order of the negated array; the quintiles were labelled
backwards, making the binned table appear to contradict the correlation
computed three lines above. The contradiction was the tell.

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
[`t3/structure/extract_pocket_pdb.py`](t3/structure/extract_pocket_pdb.py)

**Co-crystal ligand selection.** Ranking candidates by raw Tanimoto lets a 138 Da
fragment win by a 0.02 margin and define a pocket covering a fraction of the real
site. Ranking by size instead picks cardiolipin and other membrane lipids, which
mark the transmembrane face. The working rule buckets similarity into 0.1 bins,
prefers a drug-like MW window within a bucket, and blocks ions, buffers,
lanthanide phasing atoms, lipids, detergents and glycans — while **keeping**
nucleotide cofactors, since for kinases and methyltransferases those are the drug
site. [`t3/structure/rank_crystal_ligands.py`](t3/structure/rank_crystal_ligands.py)

**Domain truncation followed construct length, not binding sites.** Picking the
widest PDB construct for over-length proteins is unrelated to where ligands bind:
validation against UniProt annotations found **33% of truncations contained no
annotated site**. Ranking candidate regions by site coverage first brings that to
0%. [`t3/structure/truncate_domains2.py`](t3/structure/truncate_domains2.py),
[`validate_truncation.py`](t3/structure/validate_truncation.py)

### Operational

**A duplicate launch put six processes on three GPUs** (a manual start plus the
same script started by a tmux session). Three were killed, and
[`t3/analysis/verify_t3_raw.py`](t3/analysis/verify_t3_raw.py) was written to
check every raw output for truncation or interleaved writes. Zero corruption
found — but the check exists because the possibility was real.

---

## What this list is for

Several of these were caught because a number looked *slightly* wrong rather than
obviously wrong: identical OOM sizes, a table contradicting the correlation above
it, a fragment-sized pocket, a 0.02 similarity margin. The pattern is that
plausible-looking output is the dangerous kind, which is why the analysis scripts
state how to read a null result in their docstrings before printing anything.
