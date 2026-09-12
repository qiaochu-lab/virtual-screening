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

### Gradient accumulation does not substitute for batch size in contrastive learning

The official HypSeek recipe is `batch-size 24, update-freq 1` on four GPUs.
Batch 24 does not fit in 24 GB — the first run logged 471,107 OOM events and
finished 50 epochs without applying a single gradient. We dropped to batch 4 and
raised `update-freq` to 6, keeping the optimiser's effective batch at
4 × 6 × 4 = 96, matching the official 24 × 1 × 4 = 96, and recorded in the script
that the two were equivalent.

**They are not.** `three_hybrid_loss` builds its similarity matrix from the
ligands in the current forward pass, and neither the model nor the loss calls
`all_gather`, so negatives come from one GPU's current batch. Accumulation sums
the gradients of six small-batch losses; it never constructs a larger similarity
matrix. The optimiser saw 96; **the contrastive loss saw 4 negatives instead of
24.**

Three independently trained weights, evaluated through one pipeline, are
monotone in the negative pool and unrelated to the optimiser batch:

| | per-GPU batch | update-freq | GPUs | optimiser batch | negatives | DUD-E EF1% |
|---|---|---|---|---|---|---|
| official | 24 | 1 | 4 | 96 | **24** | 51.44 |
| collaborator | 12 | 1 | 1 | **12** | **12** | 49.34 |
| ours | 4 | 6 | 4 | 96 | **4** | 43.29 |

The collaborator's run had one eighth of the official optimiser batch and
reproduced the paper; ours matched the optimiser batch exactly and fell 15.8%
short. The deficit's shape matches the mechanism — AUROC moved 2.3% while EF1%
moved 15.8%, i.e. coarse separation survives and top-of-list discrimination does
not.

**The equivalence is real in ordinary supervised training**, which is why the
substitution looked safe and why the note in the script asserting it went
unchallenged. In any objective whose loss couples the examples within a batch —
contrastive, triplet, listwise ranking — batch size is a property of the loss,
not just of the optimiser.

### Diagnosing a memory ceiling under the wrong parallelism

The same episode has a second lesson worth separating. Batch 24 OOM'd and batch
6 crashed, so we retreated to 4. We never tried 8, 12 or 16, and never tried a
single GPU.

Both failures were measured **under 4-GPU DDP**, where each rank carries extra
memory, and where unicore's OOM recovery is actively unsafe: one rank skipping a
batch while the others keep communicating desynchronises NCCL and aborts the
process with SIGABRT. That crash is a property of the parallel setup, not of the
memory. The collaborator fit batch 12 on a single 4090 — the same 24 GB we had.

**We measured our parallelism and called it our hardware.** Before shrinking a
batch to fit, establish what a different parallel configuration allows; and treat
a distributed-training crash as evidence about distribution, not about capacity.

### A shared mutable intermediate silently narrowed a control

`t5_structure_source.py` reported whichever models happened to be in
`results/t3/summary.json` — `for m in sorted(s)`. That file is rewritten by every
`score_t3.py --models ...` run, so the table's contents depended on whatever
command ran last. The published version listed BindCLIP-hardneg and
BindCLIP-randneg, four comparisons, all non-significant, and the conclusion was
"no significant difference anywhere; predicted structures are usable
substitutes."

Re-running with all ten models: **four differ at L4 with p < 0.05, all favouring
experimental structures**, LigUnity-pocket losing 46% of its EF1% (11.70 → 6.29),
and 8 of 10 pointing that way. Only ConGLUDe survives BH-FDR across the 20
comparisons, so the corrected claim is "consistent direction, one firm case" —
but it is not "no effect", and finding 4 in the README said the wrong thing for
as long as the table stood.

The script now requires an explicit `--models` list, prints which requested
models are missing from the summary instead of skipping them silently, applies
BH-FDR and a sign test itself, marks the sequence-only negative controls, and
writes [`results/T5_structure_source.csv`](results/T5_structure_source.csv).

**Same shape as the stale-export bugs above**: a shared mutable file that
downstream code reads without asserting what it expected to find. Any script
that consumes an intermediate should state its inputs and fail when they are
absent, rather than reporting on whatever subset is present.

### A metric floor that made one model look like an exception

T3's decay was reported as the raw ratio `(L1−L4)/L1`. EF@1% has a floor at 1.0 —
a random ranking scores 1, not 0 — so that ratio understates the loss for a model
whose L1 already sits near the floor. SPRINT's L1 is 2.52, only 1.52 above
random; the raw ratio calls that a 46% decay and makes it the lone outlier
against everyone else's 64–80%, which is why the published range quietly excluded
it and the README claimed "all ten models lose 64–77%" while listing nine.

On the excess over random, `((L1−1)−(L4−1))/(L1−1)`, SPRINT is 76% and the full
ten-model range is **68–84%**. That is also the convention the AUROC decay in the
same tables already used, with 0.5 as its floor. Both columns are now published
side by side.

**The lesson is about baselines, not about SPRINT.** A ratio is only a decay if
the quantity bottoms out at zero. EF, AUROC, BEDROC and PR-AUC all have non-zero
random baselines, and mixing conventions across columns of one table is how an
artifact turns into an exception that gets explained away.

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

---

## "Not found" silently became "not present" — a layer label that was never checked

`timesplit/build/build_t3.py` decides whether a target's *family* was seen in
training:

```python
f = fam.get(up)
layer = "L3" if (f is not None and f in train_fams) else "L4"
```

`fam` is parsed from a precomputed CD-HIT 40% clustering file shipped with
LigUnity. A target **absent from that file** returns `None` and falls straight
through to L4 — the hardest layer, the one that carries the headline claim.
Absence of evidence was recorded as evidence of absence.

**61 of 254 L4 targets (24%) were not in the clustering file at all.** A
sensitive mmseqs2 search against the training set found that 10 of them have
≥70% identity to a training target — including `I6WXK4` × `P96830` at **100%
identity over full length**, i.e. the same protein. Relabelling at the 40%
threshold moves 30 targets out of L4 and changes the headline decay from −69%
to −78%.

The failure mode is not the clustering file; it is that a lookup miss and a
genuine negative produced the same branch. Any `dict.get()` feeding a
classification needs the miss handled explicitly.

---

## Alignment-free similarity search returns garbage without a coverage filter

The first mmseqs run used `-e 10000` and no coverage constraint, and reported
several "new" targets at **100% identity** to a training protein. They were
5–10 residue fragments: `fident=1.00` with `qcov=1%`. `fident` is identity
*over the aligned region*, so a perfect match on a handful of residues scores 1.0.

Requiring the alignment to cover ≥50% of **both** sequences and `E ≤ 1e-3`
turned that into a usable signal. The subsequent relabelling analysis
(`--relabel-above 0.40`) initially emptied L4 entirely, because it was still
reading the unfiltered table — the filter has to be applied where the hits are
parsed, not downstream.

---

## The lmdb cursor order, for the third time

`timesplit/analysis/ligand_novelty.py` asked: of the actives a model puts in its
top 1%, how similar are they to training ligands? The first answer looked
suspiciously like the background distribution of the whole pool.

That is the signature of a molecule-order mismatch. The models read `.lmdb`,
whose cursor order is lexicographic (`0, 1, 10, 100, …`), while the eval JSONL
is actives-then-decoys. **Equal length does not imply equal order**, and the
existing helper returned the JSONL order whenever the lengths matched.

The fix is a hard check rather than a length check: reconstruct the order, then
verify that every position labelled 1 really holds one of that target's actives;
if not, try the lmdb order; if that fails too, skip the target. 31–47 targets
per model fail both and are skipped — reported rather than silently absorbed.
With the check in place the answer inverted: models retrieve *far more*
familiar chemistry than the pool contains (HypSeek's L1 top-1% actives have
median similarity 0.969 to training ligands, against a pool median of 0.727).

This is the same trap as the T2 correction of 2026-08-21. Both times the tell
was a result that looked like a random draw from the pool.

---

## A patch that reported success without applying

Adding `per_target` output to `score_t2_v2.py` was done with three
`str.replace()` calls followed by an unconditional `print("done")`. None of the
three matched, the file was rewritten unchanged, the message printed, and the
hour-long rerun produced exactly the same aggregates as before. The waste was
only caught because the downstream subset script refused to find `per_target`.

Every in-place patch now asserts before writing:

```python
assert old in s, f"no match: {old[:60]!r}"
assert s.count(old) == 1, f"{s.count(old)} matches"
```

---

### A default that hid a contradiction with the paper

`three_hybrid_loss.py` reads `alpha_prot` — the weight on the protein-sequence
pathway — as `float(getattr(args, "alpha_prot", 1.0))`. `test_task.py` reads the
same name as `getattr(self.args, "alpha_prot", 0)`. Argparse never defined it,
so **training always ran with the pathway on and evaluation always ran with it
off**, and nothing in the config surfaced the difference. The paper states that
removing the sequence pathway costs performance.

Exposed as a real flag so the setting is measurable rather than implicit:

```python
parser.add_argument("--alpha-prot", type=float, default=0.0,
                    help="protein-sequence pathway weight (eval default 0, training default 1)")
```

Both settings are now reported. Turning it on raises DUD-E EF@1% from 51.41 to
53.02 and *lowers* LIT-PCBA from 6.82 to 5.21.

Note this flag has no effect on T3: `test_t3_target` — written for this project —
scores with `res = pocket_reps @ mol_reps.T` and never computes `prot_scores`.
Every HypSeek T3 number here is pocket-pathway-only. That is a stated convention,
not a defect, but it has to be stated.

---

### The swap directory is named after the substitute, and the first analysis assumed otherwise

The target-swap experiment holds the ligand pool fixed and replaces only the
target. The obvious implementation — swap `{target}_pocket.lmdb` in place —
does not work: these models read the protein sequence **from the directory
name**, not from the lmdb. So the swap tree has to name the directory after the
substitute `T'` and link `T`'s ligand pool into it under `T'`'s filenames.

Which means the pairing is `swap/{T'}` against `correct/{T}`, and the first
analysis paired `swap/{T}` against `correct/{T}` instead. That directory holds
`T`'s own pocket with *someone else's* ligands — a comparison between two
different molecule sets, reported as a −98% collapse before the check caught it.

The check that caught it compares the label arrays element-wise:

```python
if len(yc) != len(ys) or not np.array_equal(yc, ys):
    skipped += 1
    continue
```

Under the wrong pairing every pair failed (4791 vs 3364 molecules, and so on).
Under the right pairing all 19 passed. The scorer now runs this on every pair
and reports the skip count rather than silently dropping.

The number that came out of the corrected analysis happened to be −98% as well,
which is worth stating plainly: **the invalid comparison produced a plausible
answer.** Nothing about the magnitude signalled the error.

---

### A queue that checked idempotence before waiting for a slot

`swap_queue.sh` skips a job whose results already exist, then waits for a free
GPU slot. In that order, a job that finishes *during the wait* still gets
launched — the check ran while it was still incomplete. Two processes then write
the same result directory.

Two lines, in the wrong order. The fix re-checks after acquiring the slot:

```bash
while [ "$(running)" -ge "$MAXJOBS" ]; do sleep 60; done
if done_already "$M" "$RND"; then
  say "skipping $M round$RND (already completed while waiting)"; continue
fi
```

The same script also counted its own `grep` as a running job:

```bash
running(){ ps -eo args --no-headers | grep -cE 'run_swap_(full|seq)\.sh' ; }
```

`grep`'s own argv contains the pattern, so the count was always one too high and
four slots ran three jobs. Bracketing one character — `[r]un_swap` — stops the
pattern from matching itself.

---

### EF exceeded its own ceiling, and the docstring named the wrong ceiling

`enrichment_factor` resolved ties through average ranks:

```python
ranks = _ranks(scores)
n_active_top = int(labels[ranks <= n_top].sum())
```

A tie group straddling the cutoff gets one shared average rank, so **the whole
group** passes `ranks <= n_top` — `n_active_top` can exceed `n_top`. The
ligand-only baseline, whose Tanimoto scores tie constantly, reported EF@1% of
**51.13 against a structural ceiling of 51.00**. The same mechanism drops whole
groups whose average rank lands past the cutoff, which is why L1 read 94.2% of
ceiling while L2–L4 read 99–100%; that asymmetry was an artefact, not chemistry.

Ties are now resolved by expected value — a group straddling the cutoff
contributes its actives in proportion to the slots left:

```python
if size <= left:
    got += act; left -= size
else:
    got += act * left / size; left = 0
```

Measured before changing anything: on real model scores the difference is at
most **0.04 EF units (0.1%)**, and zero for HypSeek — continuous scores barely
tie. So no published T1/T2/T3/T5/T6 number moves. Only the ligand-only baseline
was affected, and it now reads 99.6–99.9% of ceiling across all four layers.

The docstring also claimed the ceiling is `1/fraction`. It is
`min(1/fraction, n_total/n_active)` — at 1:50 actives are 1/51 of the pool, so
EF@1% tops out at **51, not 100**. Under the wrong figure a model at 39 looks
like it uses 39% of the available range when it actually uses 77%.

38 metric tests pass after the change.

## family_queue.sh's idle-GPU detection doesn't match its own process name

When `family_queue.sh` picks a card, it first collects "cards already
occupied by this queue"; the pattern it wrote was:

```bash
grep -oE '[r]un_swap_fam\.sh [a-z_]+ ([0-9])'
```

but the process that actually runs is called `run_swap_fam_pocket.sh` /
`run_swap_fam_seq.sh` (`run_swap_fam.sh` is only a dispatcher that exits once
it has launched them). The pattern never matches, so `USED` stays empty
forever, and card selection is left with only the "utilization < 15%"
criterion — and a job that has just started takes tens of seconds to push
utilization up, so during that window it looks idle.

The log from the 2026-09-09 round left a trace of this:

```
[14:41] started drugclip round1 GPU0
[14:42] started bindclip_randneg round1 GPU0   ← same card
[14:43] started bindclip_hardneg round1 GPU1
[14:47] started conglude round1 GPU1           ← same card
```

**No OOM occurred** (the two jobs are 8.3 GB + 5.6 GB, both fit on a 24 GB
card), and the number of cards in use still stayed under the 4-GPU cap, so
that round's results are valid. But this was luck — swap in two models with
larger memory footprints and it would crash.

The fix is to change the pattern to match the real process name:

```bash
grep -oE 'run_swap_fam(_pocket|_seq)?\.sh [a-z_0-9]+ ([0-9])'
```

**This is the second time the same kind of pitfall has shown up in this
project.** Last time it was `swap_queue.sh`'s `running()`, which used
`grep -cE 'run_swap_...'` and counted its own command line too, so four slots
could actually only run three jobs; the fix was to use a character class like
`[r]un_swap` so grep would not match itself. Both times the failure was
**a process-name pattern that didn't match the actual process**, in opposite
directions: once overcounting, once undercounting.

When writing this kind of scheduling script, after changing the pattern run
`ps -eo args --no-headers | grep -oE '<your-pattern>'` once and see what it
actually catches — don't confirm it just by reading the code.

## Four index-misalignment bugs share the same underlying data-structure cause: parallel lists

As of 2026-09-09, this project has had four bugs where "the index was wrong
but the program raised no error":

| # | Where | Symptom | Check in place at the time |
|---|---|---|---|
| 1 | model reads lmdb vs. eval-set jsonl | lexicographic molecule order vs. insertion order | length comparison only |
| 2 | `ligand_novelty.py` part B | same as above | length comparison only |
| 3 | `prep_dock.py` | same as above | length comparison only |
| 4 | `per_target` in `score_t2_v2.py` | `ups.append` was called twice; `zip` silently truncated | none |

What the four have in common is not "carelessness" but the **data
structure**: molecule order, scores, labels and target names were stored as
several parallel arrays or lists, aligned by **index** — and index alignment
is checked by nothing.

### Two layers of defense, and neither is optional

**First layer: `zip(..., strict=True)`.** Unequal lengths immediately raise a
`ValueError` instead of truncating to the shorter one. Bug #4 would have
blown up on the spot with just this one line — `ups` had 2N entries, `new_r`
had N. Python 3.8 doesn't have this parameter; 3.10 does, and the server runs
3.10.20. Every `zip` in the repository that joins parallel lists now has it.

**Second layer: `strict=True` does not stop bugs #1–3.** In those three
cases the two lists had **exactly the same length**, just a different order,
and `strict=True` lets that straight through. The only way to catch it is to
**assert semantics at the boundary where external data is read**:

```python
# not "is the length right" but "at the positions where the label is 1, are
# these really that target's actives"
got = {seq[i] for i in range(n) if labels[i] == 1}
return seq if got == act else None
```

### The fix that addresses the root cause

**Wherever a record can be composed, don't use parallel lists.** If bug #4
had been written from the start as

```python
rows.append({"uniprot": up, "spearman": r, "kendall": t, "n_actives": len(pairs)})
```

a duplicate append would immediately produce one extra whole row and the
count would not match, and **every row is internally self-consistent by
construction** — a misaligned state simply cannot be represented. Parallel
lists hand the invariant "the same index is the same thing" over to the
programmer's memory; a record hands it to the data structure instead.

A related note in the same vein: **percentages are for human reading; ratios
must be computed from the raw counts.** The relative enrichment in §3a was at
one point reported as 10.7×, because it divided the already-rounded 3.2% by
0.3%; computed from the counts it is 12.6×. This is the same category of
mistake as "enrichment cutoffs must use `ceil`, not `round`" — display-time
rounding must never feed back into computation.

## A completion marker in an append-only log cannot be trusted as a completion signal

The aggregation chain hanging off the Boltz-2 rerank ran empty once, at
2026-09-10 09:19: it aggregated **0 scores**, wrote out an empty result, and
exited.

The criterion it checked was "does the log contain 'all four shards
finished'". But `results/logs/boltz_rerank_sub.log` is in **append** mode:

```
[09-09_21:51] all four shards finished        ← written when wait returned after the first run (N=5) was killed
[09-09_21:53] shard_0 started on GPU0 (N=1)   ← the second launch
```

When the chain grepped at 09-10 09:19, it matched that stale marker from 12
hours earlier.

**A marker in an append-only log only tells you that "some run finished",
not that "this run finished".** Using it as a signal requires either
clearing the log every round, writing the marker to a file tagged with a
round ID, or only starting to look for the marker after the launch line. All
three are more work than "grep a fixed string".

Changed to two criteria that don't depend on external state:

- **score count reaches the target** (3,400 / 3,747, leaving margin for
  parse failures)
- **or stall detection**: score count > 0 and no increase for 60 consecutive
  minutes (whether it finished or hung, it should be aggregated either way)

### The fourth occurrence of the same family

The first three were all variants of "using process state to judge
completion" (see the "parallel lists" item above and the family_queue item):
grep matching its own command line, a pattern that didn't match the actual
process name, a script name inside a heredoc carried over from the parent
process's command line. This time it switched to a log marker, and it was
still wrong.

**The common thread: treating "an external signal that may be stale or
self-referential" as a completion signal.** There is only one reliable
criterion — **directly count what this specific run is supposed to
produce.**

## When n is small, three summary statistics give three different answers — only the per-unit table is trustworthy

Not a code bug — a **reading-method bug**. But like the rest of this section
it made me write two successive wrong conclusions into public documents, so
it belongs here too.

**Setting**: the property control for the Boltz rerank. Do the two groups of
actives (retrieval-found vs. retrieval-missed) differ systematically across
seven molecular properties? The unit is the target, **n=5**.

The same data, three summary criteria, three different answers:

| Criterion | Conclusion it gives | Why it's wrong |
|---|---|---|
| p grows larger after pairing | "the effect disappears — it was a composition effect" | n collapses from 1,813 molecules to 5 targets, so p necessarily blows up. **At n=5, the lower bound of a two-sided Wilcoxon p-value is 2/2⁵ = 0.0625** — this test could never reach 0.05 to begin with |
| pooled/paired effect-size shrinkage ratio | "MW shrinking 76% is a composition effect; novelty shrinking only 8% is a real within-group effect" | the shrinkage is a **ratio**, the same trap as the EF ratio this project retired. It doesn't fail because the denominator is small (checked pooled/SD — all seven dimensions are ≥0.25 SD, which rules that out) but because the **paired median happens to land near the pooled value**, masking heterogeneity between units |
| van Elteren stratified test | heavy atoms Z=+3.59, **p=0.0003** | the idea was right in principle — the paired median collapses 1,813 molecules down to 5 numbers, throwing away exactly where the power comes from. But the unweighted mean of the per-stratum effects is **−0.001**, and **the sign flips depending on the weighting**. A stratified test assumes the per-stratum effects point the same way, and that assumption fails here |

**Listing the five targets' numbers directly makes it visible at a glance:**

```
per-target AUC = P(missed value > found value), 0.5 = indistinguishable within that target
Target        MW    heavy atoms   novelty (affinity half)
O14578     0.619   0.608      0.094
O42275     0.217   0.162      0.116
P20648     0.500   0.478      0.648
Q8N1C3     0.848   0.845      0.508
Q96DB2     0.448   0.400      0.470
```

**Not one of the seven dimensions points the same way across the five
targets.** The "8% shrinkage" effect is carried by two targets, O14578 and
O42275; P20648 goes the other way, and the other two sit right at 0.5.
**There is no common effect to pool** — which is exactly why the three
summary statistics fight each other: they are all answering the question
"how large is the common effect", and that question's premise does not hold.

**There is an even more basic pitfall here**: pooled and paired must use
**the same set of units**. The first version compared a 12-target pooled
figure against a 5-target paired one, and that comparison was invalid to
begin with (the pooled set mixed in three targets that were not in the
paired set at all).

### Rules

- **For a comparison with n ≤ 10, list the per-unit table first, and only
  then decide whether a summary statistic is worth computing.** When the
  table and the statistic disagree, trust the table.
- Before reporting "the effect exists / disappears", ask first: **do the
  units point the same way?** If not, don't use any single number.
- Pairing cuts n down to the number of units — **don't read the resulting
  increase in p as the effect disappearing.**
- When n is small, print the Wilcoxon lower bound `2/2**n` alongside the
  p-value, as a reminder of whether it could ever reach 0.05.
- Template: [`physics/check_missed_vs_found_props.py`](physics/check_missed_vs_found_props.py) —
  its main output is the per-unit table; the pooled/paired figure is
  demoted to a secondary output and carries a warning.

### Why this deserves its own entry: this project has more small-n analyses than large-n ones

L3 has only 19 targets, the homologous-family swap's L4 has only 7 pairs,
the T6-RE subset has n=11, the fully-docked targets number n=5–9, and the
FEP benchmark has 16 systems. **All of these apply here.**

A pass has already been made back through them: the conclusions in the
places above are all **negative or weak statements** ("no significant
decline", "cannot show it actively gets worse"), and the homologous-family
swap already states that "the L4 rows cannot be cited on their own — the
conclusion mainly rests on L1's 32–34 pairs." The two places carrying
**positive** causal conclusions (the ConPLex/ConGLUDe class-based reversal)
are at n=24 and n=76, both with per-target pairing plus BH-FDR — **they are
not in this failure mode.**

## The same proportion has two different values on two target sets — default to writing both

The single most frequent category of problem today, showing up three times:
README finding 9, T3 v2's 242/222, and this round's recall@200 — I wrote
"top-200 covers 25%" in the README, which is the figure for the **12 targets
actually run**; **the full L4 350-quota subset's 68 targets give 22.6%**.
Both are correct; mixing them up is the error.

The root cause is that this project has target sets nested across multiple
layers: the full set → the 350-quota subset (328 entries / 293 targets) →
the VSDS-matched subset (242 entries / 222 targets) → whichever targets a
given round actually ran. **The same proportion has a different value at
every layer**, and the denominator is usually not written next to the
number.

### Rule

When reporting a coverage / recall / hit-rate proportion, **default to
writing it as "X% (12 targets) / Y% (all 68)"** rather than picking one and
waiting for someone else to notice the definition doesn't match. When only
one value is given, the denominator must immediately follow it.

## A nested heredoc dropped the quotes from a patch, and the job launched anyway

Not a bug in the analysis code — a bug in how a patch was applied to it. It
belongs in this list because the failure shape is the one this project keeps
hitting: **the command reported success and the wrong thing ran.**

The intent was to add one model to a hardcoded list on the compute host, then
start the job that reads it:

```bash
ssh host '
  python - <<EOF
p = "$B/export_t3_clean.py"
s = open(p).read()
old = \'"litenclip", "hypseek_rk", "conglude", "conplex", "sprint"]\'
...
EOF
  setsid nohup python "$B/export_t3_clean.py" > log 2>&1 &
'
```

The heredoc delimiter was written as `<<EOF`, not `<<'EOF'`, inside a
single-quoted `ssh` argument. The shell therefore expanded and quote-stripped
the body before handing it to Python, so the string literal arrived as bare
tokens:

```
old=litenclip, hypseek_rk, conglude, conplex, sprint]
                                                    ^ SyntaxError: unmatched ']'
```

**The patch did not apply. The launch on the next line ran regardless**, because
it was a separate statement rather than chained with `&&`. The job spent its
first minutes computing the old model list — the exact gap the patch existed to
close — and would have written a plausible-looking CSV with the new weight
missing.

### What caught it

Only that the same command printed both the SyntaxError and, further down,
`grep` finding nothing for the new model name. Had the script been quieter, the
output would have been indistinguishable from a clean run.

### Rules

- **Never nest an unquoted heredoc inside a quoted remote command.** Use
  `<<'EOF'` so the body is passed through verbatim.
- **Better: do not send code through a heredoc at all.** Edit the file locally,
  where it is under version control and syntax-checkable, then ship the whole
  file — `sed 's#local#remote#g' file | ssh host 'cat > dest'`. That is how the
  fix was finally applied, and it has the side benefit of keeping the two copies
  provably identical.
- **Chain the patch and the launch with `&&`,** so a failed patch cannot be
  followed by a run. A patch that fails open is worse than one that fails shut.
- **Verify the patch landed before trusting the run** — `grep` for the new value
  on the remote copy, as a separate assertion, not as a line of output nobody
  reads.

### Same family

`recompute_all.sh` would have silently reverted the checkpoint switch because its
model list was stale; a completion marker in an append-only log reported a run
that had been killed; three process-name greps matched their own command line.
All four share the signature: **exit status 0, and the wrong inputs.**

### The self-matching grep, found alive four more times

Cleaning up afterwards turned up four background waiters that had been looping
for up to three days:

```bash
until ! pgrep -f score_t2_v2.py >/dev/null; do sleep 10; done; <payload>
```

The waiter's own command line contains the string `score_t2_v2.py`, so `pgrep -f`
always matches the waiter itself, the condition is never false, and the payload
never runs. Two of the four were waiting on work that had finished days earlier;
their payloads — a summary invariant check and a subset re-export — simply never
executed, and nothing reported that.

The diagnostic written to investigate them fell into the same trap: it passed the
process names as literals, so `pgrep -f` matched the diagnostic's own shell and
reported the very processes it was checking for. The bracket idiom
(`pgrep -f "[s]core_t2_v2.py"`) defeats this, because the pattern text no longer
matches itself — but the durable fix is not to identify work by its command line
at all. Wait on a **file** the job produces, or on a PID captured at launch.

