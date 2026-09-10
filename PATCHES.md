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
  say "跳过 $M round$RND（等待期间已完成）"; continue
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

## family_queue.sh 的空闲 GPU 检测匹配不上自己的进程名

`family_queue.sh` 挑卡时先收集「已被本队列占用的卡」，模式写的是：

```bash
grep -oE '[r]un_swap_fam\.sh [a-z_]+ ([0-9])'
```

但实际起来的进程叫 `run_swap_fam_pocket.sh` / `run_swap_fam_seq.sh`
（`run_swap_fam.sh` 只是个分发器，起完就退出）。模式匹配不上，`USED` 恒为空，
选卡就只剩「利用率 <15%」这一条判据——而一个刚起来的任务要几十秒才把利用率
拉上去，这段窗口里它看起来是空闲的。

2026-09-09 那轮的日志留了痕：

```
[14:41] 起 drugclip round1 GPU0
[14:42] 起 bindclip_randneg round1 GPU0   ← 同一张
[14:43] 起 bindclip_hardneg round1 GPU1
[14:47] 起 conglude round1 GPU1           ← 同一张
```

**没有 OOM**（两个任务 8.3 GB + 5.6 GB，24 GB 的卡放得下），而且用掉的卡比
4 张上限还少，所以那一轮结果是有效的。但这是运气：换两个显存大的模型就会崩。

修法是把模式改成匹配真实进程名：

```bash
grep -oE 'run_swap_fam(_pocket|_seq)?\.sh [a-z_0-9]+ ([0-9])'
```

**同一类坑在本项目里出现过第二次。**上一次是 `swap_queue.sh` 的 `running()`
用 `grep -cE 'run_swap_...'` 把自己的命令行也数了进去，四个槽实际只能跑三个，
修法是用 `[r]un_swap` 这种字符类让 grep 不匹配自己。两次都是**进程名模式和
实际进程对不上**，方向相反：一次多算，一次少算。

写这类调度脚本时，模式改完先跑一次
`ps -eo args --no-headers | grep -oE '<你的模式>'`
看它到底抓到了什么，别靠读代码确认。

## 四次索引错位有同一个数据结构根因：并行列表

到 2026-09-09 为止，本项目出现过四次「下标对错了、但程序不报错」的 bug：

| # | 在哪 | 表现 | 当时的检查 |
|---|---|---|---|
| 1 | 模型读 lmdb vs 评测集 jsonl | 分子顺序字典序 vs 插入序 | 只比长度 |
| 2 | `ligand_novelty.py` B 段 | 同上 | 只比长度 |
| 3 | `prep_dock.py` | 同上 | 只比长度 |
| 4 | `score_t2_v2.py` 的 `per_target` | `ups.append` 打了两次，`zip` 静默截断 | 无 |

四次的共同点不是「粗心」，是**数据结构**：分子顺序、打分、标签、靶点名被存成
几个平行的数组或列表，靠**下标**对齐，而下标对齐是没有任何东西检查的。

### 两个层次的防御，缺一不可

**第一层：`zip(..., strict=True)`。** 长度不等立刻抛 `ValueError`，而不是截断到
最短。第 4 次那个 bug 只要有这一行就会当场炸——`ups` 有 2N 条、`new_r` 有 N 条。
Python 3.8 没有这个参数，3.10 有；服务器上是 3.10.20。仓库里所有拼并行列表的
`zip` 都加了。

**第二层：`strict=True` 挡不住第 1–3 次。** 那三次两个列表**长度完全相同**，
只是顺序不同，`strict=True` 一样放行。要挡住只能**在读外部数据的边界上断言语义**：

```python
# 不是「长度对不对」，是「标签为 1 的位置上，是不是真的是这个靶点的 active」
got = {seq[i] for i in range(n) if labels[i] == 1}
return seq if got == act else None
```

### 治本的那条

**能合成一个 record 就别用并行列表。** 第 4 次如果一开始写成

```python
rows.append({"uniprot": up, "spearman": r, "kendall": t, "n_actives": len(pairs)})
```

重复 append 会立刻多出一整行、条数对不上，而且**每行内部永远自洽**——错位这种
状态根本不可表示。并行列表把「同一个下标是同一个东西」这个不变量交给了程序员
记忆，record 把它交给了数据结构。

顺带一条同源的：**百分比是给人读的，倍数要从原始计数算。** §3a 那个相对富集
一度报成 10.7×，因为拿四舍五入后的 3.2% / 0.3% 相除；从计数算是 12.6×。
和「EF 取整必须用 ceil 不是 round」是同一类——展示用的取整不能进计算。

## 追加日志里的结束标记不能当完成信号

挂在 Boltz-2 重排后面的汇总链在 2026-09-10 09:19 空跑了一次：它拿 **0 个分数**
汇总，写出一个空结果，然后退出。

判据写的是「日志里有没有『四个 shard 全部结束』」。而
`results/logs/boltz_rerank_sub.log` 是 **append** 模式：

```
[09-09_21:51] 四个 shard 全部结束        ← 第一次（N=5）被杀掉后 wait 返回写的
[09-09_21:53] shard_0 起在 GPU0 (N=1)   ← 第二次启动
```

链在 09-10 09:19 一 grep 就命中了 12 小时前那条陈旧标记。

**追加日志里的标记只说明「某一次跑完了」，不说明「这一次跑完了」。** 想用它
当信号，得要么每轮清空日志、要么把标记写成带轮次 ID 的文件、要么在启动行之后
才开始找标记。三种都比「grep 一个固定串」麻烦。

改成两个不依赖外部状态的判据：

- **出分数达标**（3,400 / 3,747，留出解析失败的余量）
- **或停滞检测**：出分数 > 0 且连续 60 分钟没增长（跑完或卡死，都该汇总）

### 这是同一个家族的第四次

前三次都是「用进程状态判完成」的变体（见上一条「并行列表」和
family_queue 那条）：grep 匹配到自己的命令行、模式和实际进程名对不上、
heredoc 里的脚本名被父进程的命令行带上。这次换成了日志标记，还是错的。

**共同点：把「一个可能陈旧或自指的外部信号」当成完成信号。**
可靠的判据只有一类——**直接数这次任务该产出的东西**。
