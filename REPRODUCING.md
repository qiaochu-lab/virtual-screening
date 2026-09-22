# Reproducing these numbers

The route from raw data to every published figure, and the traps on it. Each
stage links to the page that owns it rather than restating it; what this page
adds is the **cross-cutting parts** — the conventions that apply everywhere, and
the ways a careful person still gets a different number than we did.

**The reporting convention is the 350-quota subset: 328 evaluation records over
293 unique targets** (L1 56 · L2 178 · L3 19 · L4 75). The full 1,144-target set
is an auxiliary check. Any table labelled "full set" is the auxiliary one.

---

## 1. The dataset

Built rather than borrowed, because DUD-E / DEKOIS / LIT-PCBA have synthetic
decoys, participated in checkpoint selection, and have no temporal holdout.

| | |
|---|---|
| Cutoff | **2024-12**, read from LigUnity's own training labels (max `version` 34, 565 ChEMBL v34 assays) combined with BindingDB 2024m5 |
| Sources | ChEMBL 37 + BindingDB 202608, post-cutoff records, set-differenced against the training sets, InChIKey-deduplicated |
| Layers | L1 target seen / ligand new · L2 target seen / scaffold new · L3 target unseen / family seen · L4 both unseen. Family = CD-HIT 40% |
| Actives | measured pAffinity ≥ 6, InChIKey-deduplicated, ≥ 10 per target |
| Decoys | **cross-target, not property-matched**, 1:50; excludes the target's own actives, anything active on a target in the same mmseqs 40% cluster, and anything sharing a Bemis–Murcko scaffold with its actives |
| Structures | experimental PDB where available, Boltz-2 otherwise; usable pocket coverage 95.4% |
| Pockets | residue-level 6 Å, replicating DrugCLIP's `get_different_raid()`; validated at 100% coordinate overlap against the authors' own pockets on five DUD-E targets |

Construction pipeline and the four things that are easy to get wrong:
[`timesplit/README.md`](timesplit/README.md). Subset selection (iterative
proportional fitting, the two ways it went wrong first, class deviations):
[`tasks/T3-dataset-v2.md`](tasks/T3-dataset-v2.md).

⚠️ **Absolute numbers here are not comparable to published values** — the decoys
are drawn differently on purpose. Only the L1→L4 decay within one fixed setup
means anything.

---

## 2. How each model was run

**Official code, official weights; only the metric computation is unified.** One
checkpoint per model across all tasks. Full checkpoint table, the reasons behind
each variant, and the interface quirks: [`MODELS.md`](MODELS.md).

The UniMol-family models (DrugCLIP, BindCLIP ×2, LigUnity ×2, LiTENCLIP,
HypSeek) share one command skeleton, and **one deliberate departure from the
official defaults**:

```
--batch-size 8          # official is 32 (DrugCLIP) / 256 (LigUnity, LiTENCLIP)
--max-pocket-atoms 511  # HypSeek's T3 runs used 256 — see below
--fp16 --seed 1 --num-workers 4
```

Batch size 8 because T3's molecules come from ChEMBL/BindingDB and reach **336
atoms** where DEKOIS tops out at 50; UniMol's attention is O(n²), so the official
setting OOMs on ~70% of targets. Everything else matches each model's own
official DEKOIS branch, so inference paths differ only where the models do.

Several upstream repositories print aggregate metrics and discard the
per-molecule scores, which makes unified evaluation impossible. Every patch is
either "make it save its raw output" or "make it run" — no modelling logic was
touched, and DrugCLIP on DEKOIS reproduces the published baseline to 0.0%. The
list, including the four bugs that cost the most time:
[`PATCHES.md`](PATCHES.md).

**Model-side conventions that have to travel with the numbers:**

1. **Every HypSeek T3 number is pocket-pathway only.** `test_t3_target` scores
   with `pocket_reps @ mol_reps.T` and never computes the sequence pathway, so
   `--alpha-prot` has no effect there. Stated convention, not a defect.
2. **HypSeek's T3 runs used `--max-pocket-atoms 256`** while everything else used
   511. At 6 Å, 19.7% of pockets exceed 256 atoms and get center-weighted
   cropping. Re-run at 511 it changes nothing measurable (L1 AUROC 0.923→0.924),
   but the asymmetry exists.
3. **Screening tables use HypSeek `_vs`, ranking tables use `_rk`.** Analyses
   built before that switch were computed with `_rk` and are labelled at each
   appearance.
4. **LiTENCLIP 2.0 is a different architecture, not LiTENCLIP with new weights.**
   Easy to misread, so stated plainly: the main model file was replaced
   (`LiTENCLIP.py`, 447 lines → `three_hybrid_model.py` 392 + `lorentz.py` 269,
   Lorentz hyperbolic geometry adapted from Meta's MERU), the checkpoint went
   654 MB → 331 MB, and a **third tower** was added — a protein-sequence pathway
   on `facebook/esm2_t12_35M_UR50D`. It is built inside HypSeek's codebase
   (the package calls itself HypLiTEN), which is where its T1 behaviour comes
   from: DUD-E 50.57 sits beside HypSeek's 51.41, while v1 scored 43.95.
   Scoring is `alpha_poc * max(pocket @ mol) + alpha_prot * max(protein @ mol)`,
   both alphas 1.0 here. ⚠️ Porting the T3 path from v1 verbatim **silently
   disables the sequence tower** — v1 calls
   `pocket_forward(protein_sequences=seq, ...)` and v2's `pocket_forward` takes
   no such argument, so it vanishes into `**kwargs`. Pocket radius is 6 Å, as
   for every other model. **Its training data is unknown**: the package ships no
   training config or log, and while the run is named
   `hypliten_..._litpcba_dude`, every `DUDE`/`PCBA` reference in it sits in
   *evaluation* scripts — that name is not evidence either way about whether
   DUD-E or LIT-PCBA were trained on. A pocket-tower-only variant
   (`litenclip_v2poc`) is kept as a like-for-like control against v1, which has
   no sequence tower; it is a diagnostic, not a published model.

5. **DrugJEPA trains on the same corpus as four models already in these tables.**
   Verified by hash, not by reading the paper: its
   `pocket_name2idx_train_blend.json` (sha256 `1d3d707d…`, 3,393,961 B) and
   `mol_smi2idx_train_blend.json` (sha256 `431f8068…`, 31,913,119 B) are
   **byte-identical** to this project's figshare copies of the affinity half —
   the corpus behind LigUnity ×2, HypSeek and LiTENCLIP. Only its PDBbind label
   file differs. Two consequences. First, DrugJEPA is a **controlled
   architecture comparison**: same training data, different architecture (JEPA +
   MoE against contrastive). Second, its benchmark-exposure figures are **not
   specific to it** — 52.4% of DUD-E actives, 39.4% of DEKOIS, 34.0% of
   LIT-PCBA and 60.0% of CASF-2016 complexes sit in that corpus, and that
   applies to all five models equally. Scoring is
   `(pocket @ mol.T).max(0)` with the sequence fused inside `pocket_forward`,
   so it is single-tower like LiTENCLIP v1, not three-tower like v2, and has no
   alpha to set. Pocket radius 6 Å, batch 8 (T3) / 128 (T1) as for every other
   model; the upstream defaults are `--max-pocket-atoms 2048` and a batch size
   hardcoded to 64 inside `test_dude_target`, both overridden here. Its scores
   come out as float16.

⚠️ **LigUnity-pocket was re-run on a newer screening checkpoint on 2026-09-14**
(md5 `f8ffada8…`). Everything on the 350-target convention uses it; the full
1,144-target tables keep the first checkpoint's rows. Both score packages are
published. What changed in the conclusions: [`PATCHES.md`](PATCHES.md) part 3.

---

## 3. The evaluation layer

Published numbers for the same model on the same benchmark disagree across
papers — DrugCLIP's DUD-E EF1% is 31.99 in its own paper and 30.52 re-run in
BindCLIP's — because each ships its own metric code. Here the models keep their
own inference and share one metric implementation
([`eval/README.md`](eval/README.md)):

| Convention | Value | Why it matters |
|---|---|---|
| EF cutoff rounding | **`ceil`, not `round`** | matches RDKit `CalcEnrichment`; synthetic test sizes pass under either convention, so this only shows up on real data |
| EF ties | expected value — a tie group straddling the cutoff counts proportionally | counting such a group in full once put a baseline **above** the theoretical ceiling |
| AUROC / BEDROC ties | average rank | |
| BEDROC α | 80.5 | Truchon & Bayly (2007) |
| R² | Pearson *r*², not `1 − SS_res/SS_tot` | scores and measured affinities are on different scales |
| EF ceiling | `min(1/fraction, n_total/n_active)` | not `1/fraction` |

Validated three ways: definitions from the literature, agreement with RDKit
within 1e-6, and reproduction of each model's published values within 2%.
Comparisons are paired per target, and bootstraps resample **targets, not
molecules**.

---

## 4. What you need to recompute a number

Three tiers, indexed in [`DATA_RELEASE.md`](DATA_RELEASE.md):

- **Tier 1 (in this repository)** — `results/frozen/`: the molecule table, the
  per-target index with **both** position columns, and `T3_model_order.csv`.
- **Tier 2 (on request)** — one `.npz` per model per task, keys
  `"<task>/<layer>/<uniprot>/{preds,labels}"`, with sha256 manifests.
  Format and a worked example: [`results/RAW_SCORES.md`](results/RAW_SCORES.md).
- **Tier 3 (on request)** — 6 Å pockets and ligand lmdbs for the subset.

**The two position columns are the important part.** A model reading an lmdb sees
lexicographic cursor order (0, 1, 10, 100, …); the eval-set jsonl is in a
different order. Joining scores to labels by the wrong one silently mislabels
every molecule — that bug cost this project two retracted conclusions, so the
mapping ships rather than being left to be re-derived. Any script that joins
scores to information **outside** the score array must first call
`eval/order_guard.py::assert_cursor_order()`.

`order_used` in `T3_model_order.csv` is decided per (model, target)
(`freeze_t3.py`): try the jsonl order — array length must match **and** the
actives implied by the labels must match — then the lmdb cursor order under the
same two tests, else `FAIL`. Every UniMol-family model has 116–119 FAIL targets;
45 of the 309 scored subset targets are FAIL. **`FAIL` limits re-layering from
the frozen index; it does not affect any published metric**, which is computed
from each model's own self-consistent `saved_preds.npy` / `saved_labels.npy`.

---

## 5. Traps

1. **EF rounding uses `ceil`** (§3). Invisible on synthetic sizes.
2. **fp16 makes LIT-PCBA wobble ~1e-3** at the top-5% boundary on its two
   largest targets (`VDR` 356k molecules, `ALDH1` 145k); 13 of 15 targets
   reproduce to the last digit. DUD-E and DEKOIS agree to 5e-5. **The published
   table is the original run and was not changed.**
3. **Molecule order**: lmdb cursor order vs jsonl order (§4).
4. **Don't recompute from embeddings** — unless the saved embeddings are the
   very arrays the model multiplied. Recomputing dot products from float16
   embeddings generally lands different molecules at the EF@1% boundary, so the
   published score arrays are the safe input. The exception is worth knowing,
   because it was measured on DrugJEPA's DUD-E run (2026-09-22): there the
   model saves `saved_mols_embed.npy` and `saved_target_embed.npy` immediately
   after computing `res = pocket_reps @ mol_reps.T` from those same fp16 arrays,
   and reconstructing the product reproduces the native scores **bit for bit on
   all 102 targets** (max absolute difference 0.0, and the top-1% set identical
   on every target). The trap is not fp16 as such — it is reconstructing from
   embeddings that were cast or recomputed somewhere between the score and the
   save. Check which case you are in before trusting either.
5. **`summary.json` is rewritten whole.** `score_t3.py` and `score_t2_v2.py`
   start from `summary = {}`, so passing a subset of models silently drops the
   rest. Pass all of them, or write to a temp file and diff.
6. **T3 has two result trees**: `results/t3_raw/<model>/T3/<layer>/<uniprot>/`
   and `results/t3/<model>/<layer>/<uniprot>/` (no `T3/` level). A packaging
   script that walked only the first silently omitted two models.
7. **Subset filters must key on `(layer, uniprot)`**, never uniprot alone — 35
   targets appear in more than one layer, because L1/L2 are split by ligand
   scaffold.
8. **`T3_vsds_matched.csv` is the current 350 quota (328 records);
   `T3_vsds_matched_q250.csv` is the superseded 250 quota.** The unsuffixed file
   is the newer one, which reads backwards and has caused mistakes — check the
   row count.
9. **Scripts carry a hardcoded `B = "/data/work/..."`**. They are published as a
   record of what was executed, not as a turnkey package. ⚠️ **Copying one onto a
   real tree without fixing `B` fails silently**: `model_order()` falls back to
   `{B}/data/T3_6A/.../*_lig.lmdb`, every lookup misses, all targets fail the
   order check, and the script writes a header-only table and **exits 0**. Seen
   on 2026-09-15: 315 of 315 targets skipped, empty tiered table, no error.
10. **LIT-PCBA was subsampled for ConPLex and SPRINT** (`--max-decoys`; actives
    never subsampled). EF must use the sampling ratio recorded in the output.
11. **HypSeek T3 = pocket pathway only, 256-atom cap** (§2).
12. **Absolute numbers are not comparable to published values** (§1).
13. ⚠️ **The archive handed out on 2026-09-13 carried the first checkpoint's
    LigUnity-pocket scores under the current checkpoint's filename**, with its own
    manifest agreeing with the file — so an external checksum check passed on the
    wrong data and LigUnity-pocket recomputed to L1 34.78 / L4 7.90 instead of
    35.28 / 9.47. **Corrected on 2026-09-15**; a copy fetched before that date
    still has it. See [`results/RAW_SCORES.md`](results/RAW_SCORES.md).
14. **`T3_model_order.csv` covered 10 of the 11 T3 packages** until 2026-09-15
    (`hypseek_official_vs` was missing from `freeze_t3.py`'s default list). Fixed;
    if you hold an older copy, that model has no ordering verdict in it.
15. **"Structure half" has two non-equivalent derivations** (§6).
16. ⚠️ **Subset and full-set tables are on different tie conventions since
    2026-09-15.** The 350-subset tables were regenerated with the shared
    `eval/metrics.py::top_weights` (a tie group straddling the top-1% cutoff is
    counted proportionally). The auxiliary full-set tables — including
    `T3_seq_vs_pocket_per_target.csv` — were **not** regenerated, per the
    standing convention that the 1,144-target set is auxiliary. They remain
    internally consistent under the older argsort-slicing convention. Do not
    compare a subset cell against its full-set counterpart digit-for-digit, and
    do not "fix" a full-set table by re-running the current script against a
    newer score package: that silently folds a **checkpoint change** into what
    looks like a tie-convention change (measured: 659 cells moving by up to
    46.67, versus at most 0.71 for the convention alone).
17. ⚠️ **`T3_novelty_tiered_ef.csv` means different things in this repo and on
    the working machine.** Here it is the **full 1,144-target** table from
    2026-09-07 — the one README and `tasks/T3-leakage.md` cite for "27.1 on
    chemistry it has seen, 4.5 on chemistry it has not" — and the 350-subset
    version lives beside it as `T3_novelty_tiered_ef_subset.csv`. On the working
    machine the unsuffixed name holds the **subset** numbers instead, and its
    own `_subset.csv` is an older pre-tie-fix copy. The two are easy to tell
    apart by `n_targets`: the novel tier at L1 has 18 targets on the full set
    and 7 on the subset. Check that column before copying either file across.

---

## 6. Two calibration choices that are not settled

**Which mirroring table decides `corrected` layering — settled 2026-09-15 in
favour of the union table.** The main table reports `original` and `corrected`
layers, where `corrected` relabels L4 targets whose homology to training exceeds
0.40. `score_subset.py --mirroring` now defaults to
`T3_target_mirroring_union.csv`, which compares against 4,847 targets where the
old default (`T3_target_mirroring.csv`) compared against only 2,196.
**Fourteen subset L4 targets** that are in fact homologous to training were
being scored as fully novel, and the switch relabels them to L3. At the subset
level that is L3 19→33 and L4 75→61 (recomputed 2026-09-22 directly from
`T3_vsds_matched.csv` and the union table; the counts hold for the subset as a
whole, independent of any model). Per-model `n_targets` move by less, because
no model covers all 75 original L4 targets: typically by 7 — e.g. L3 26→33 /
L4 61→54 for the five strongest, L3 26→33 / L4 62→55 for the DrugCLIP and
BindCLIP family. **Neither "7" nor "26→33 / 61→54" is a subset-level fact**;
both are per-model rows. An earlier revision of this section called the 7 a
subset figure, which it is not.
`original` layering is untouched. **This was not a free win** — it changes the
`corrected` half of the main table for every model, raising L3 for most and
lowering L4 for most, but not uniformly.

The concrete case for the union table: **12 subset L3/L4 targets carry a 100%
self-match under it and none under the default** — six of them have no identity
recorded there at all, so the default files them under "no hit". Those same 12
are exactly the targets `LIMITATIONS.md` §25 counts as seen through the structure
half (4 L3 + 8 L4): one set of targets, two symptoms. Per-target membership for
all 328 subset records is in
[`results/T3_subset_train_membership.csv`](results/T3_subset_train_membership.csv)
(producing script:
[`timesplit/analysis/subset_train_membership.py`](timesplit/analysis/subset_train_membership.py)).

**Which derivation of "the structure half".** Two exist and they do not agree:

| Derivation | Source | UniProts | Subset L4 hits |
|---|---|---|---|
| the label file's `uniprot` field | `train_label/train_label_pdbbind_seq.json` (`train_set_crossover.py`) | 3,468 | **8** |
| lmdb pockets mapped through PDB→UniProt | `train_no_test_af/train.lmdb` + `drugclip_pdb2uniprot.json` (`build_train_union.py`) | 3,551 | 15 |

Published counts use the first and reproduce exactly under it. State which one
any statement about structure-half membership used, or the same sentence yields
two answers.
