"""The protein x ligand capability map, and the four analyses built on it.

Why this file exists
--------------------
Finding 18 taught this project what a headline number without a producing
script costs: it cannot be re-derived when the inputs change. Four results
destined for section 2.4 were computed interactively --- the 2D map, the
continuous chemical curve, sequence-vs-pocket explanatory power, and paralog
selectivity --- so this is their producing script.

It refuses to report anything until it has reproduced two published tables
(``--selfcheck``, on by default). If either fails, it exits non-zero rather
than printing numbers.

What it computes
----------------
1. ``map``      EF@1% over (protein band) x (ligand-similarity tier)
2. ``tests``    in-training vs not (the "step"), and across the three
                sequence-novel bands (the "gradient"), both BH-corrected
3. ``chem``     per-target Spearman(similarity, rank-percentile) --- the
                continuous chemical resolution curve
4. ``seqpkt``   sequence identity vs PocketVec distance as predictors of
                per-target EF, run twice: all targets, then sequence-novel only
5. ``paralog``  within-family selectivity on same-family pairs sharing ligands
6. ``frontier`` the degenerate capability frontier at several EF thresholds

⚠️ The column named "novelty" holds a SIMILARITY
------------------------------------------------
``frozen/T3_molecules.csv.gz``'s ``novelty_pocketaffdb`` is the **maximum**
ECFP4 Tanimoto to the training ligands, not a distance. Low value = novel
chemistry, which is why the tier labels read ``全新 <0.35``. Treating it as a
"novelty score" flips the sign of every correlation. The direction is pinned
by ``--selfcheck``: grouping actives by the published tier names must give a
mean rank-percentile that rises with the published EF.

⚠️ Two cohort rules that are easy to get wrong
----------------------------------------------
* **EF needs no molecule-order alignment** (it uses only preds+labels), but
  every analysis that joins per-molecule information --- similarity, pAff ---
  does. Alignment drops the ``FAIL`` targets, so the EF cohort and the
  per-molecule cohort are **not the same set**. Mixing them silently changes n.
* **"Label vector matches" does not prove molecules match.** Any permutation
  inside the active block and inside the decoy block reproduces the labels
  while scrambling every molecule. Order verdicts are therefore read from
  ``frozen/T3_model_order.csv``, never re-derived here.

Subset keying
-------------
The 350-quota subset is 328 records / 293 uniprots; 35 uniprots appear in more
than one layer, so every filter keys on ``(layer, uniprot)``.
"""
import argparse
import collections
import csv
import gzip
import math
import os
import sys

import numpy as np
from scipy import stats

B = "/data/work/vs-benchmark"

TIERS = [(0.0, 0.35, "全新 <0.35"), (0.35, 0.50, "远 0.35–0.5"),
         (0.50, 0.70, "近 0.5–0.7"), (0.70, 1.01, "极近 ≥0.7")]
BANDS = ["in training", ">=60%", "20-60%", "no hit"]
NOVEL_BANDS = [">=60%", "20-60%", "no hit"]
LAYERS = ("L1", "L2", "L3", "L4")
MIN_T = 3          # a tier needs this many actives on a target to count
FRAC = 0.01
THIN = 10          # below this many targets a cell is flagged, never quoted alone

MODELS = ["ligunity_protein_ranking", "ligunity_pocket_ranking", "litenclip",
          "hypseek_official_vs", "hypseek_rk", "drugclip", "bindclip_randneg",
          "bindclip_hardneg", "conglude", "conplex", "sprint"]


def tier_of(v):
    for lo, hi, name in TIERS:
        if lo <= v < hi:
            return name
    return None


def _ef_fallback(scores, labels, fraction):
    """Byte-for-byte the convention of eval/metrics.py: ceil rounding, ties
    counted by expected value. Used ONLY when that module cannot be imported;
    which one is active is printed at startup, because a silently forked metric
    is the thing the unified evaluation layer exists to prevent."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    n_total, n_active = len(labels), int(labels.sum())
    if not n_active or not n_total:
        return float("nan")
    n_top = max(1, int(math.ceil(n_total * fraction)))
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], labels[order]
    got, left, i = 0.0, n_top, 0
    while i < n_total and left > 0:
        j = i
        while j < n_total and s[j] == s[i]:
            j += 1
        size, act = j - i, int(y[i:j].sum())
        if size <= left:
            got += act
            left -= size
        else:
            got += act * left / size
            left = 0
        i = j
    return (got / n_top) / (n_active / n_total)


def resolve_ef(eval_dir, quiet=False):
    """Prefer the repository's shared implementation; never fork it quietly."""
    for cand in (eval_dir, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "..", "eval")):
        if not cand:
            continue
        try:
            sys.path.insert(0, cand)
            from metrics import enrichment_factor as ef
            # Report the module's REAL file, never the candidate directory:
            # once `metrics` is in sys.modules (this file lives two levels under
            # the repo root, so import time already resolved it), a later
            # `from metrics import ...` succeeds from cache whatever `cand` is,
            # and printing `cand` would claim an origin that was never read.
            import metrics as _m
            if not quiet:
                print("EF implementation: %s (shared)"
                      % getattr(_m, "__file__", f"{cand}/metrics.py"))
            return ef
        except Exception:
            sys.path.pop(0)
    if not quiet:
        print("EF implementation: built-in fallback -- eval/metrics.py was NOT "
              "importable; pass --eval-dir to use the shared one",
              file=sys.stderr)
    return _ef_fallback


# Resolved at IMPORT time, not only in main(): reviewers import this module and
# call the functions directly, and in that path a main()-only assignment would
# leave the fallback silently active -- the exact failure this file warns about.
#
# Run as a script, stay quiet here and let main() report once after it has had
# its --eval-dir; imported as a module, there is no second chance, so a fallback
# must announce itself immediately.
enrichment_factor = resolve_ef(None, quiet=(__name__ == "__main__"))


class Data:
    def __init__(self, a):
        self.a = a
        self.sub = {(r["layer"], r["uniprot"])
                    for r in csv.DictReader(open(a.subset))}
        self.order = {(r["model"], r["layer"], r["uniprot"]): r["order_used"]
                      for r in csv.DictReader(open(a.model_order))}
        self.sim = {r["mol_id"]: float(r[a.sim_column])
                    for r in csv.DictReader(gzip.open(a.molecules, "rt"))}
        self.mir = {r["uniprot"]: r for r in csv.DictReader(open(a.mirroring))}
        self.idx = {}
        for L in LAYERS:
            d = collections.defaultdict(list)
            with gzip.open(a.index_glob.replace("{layer}", L), "rt") as fh:
                for r in csv.DictReader(fh):
                    if r["uniprot"] in {u for _, u in self.sub}:
                        d[r["uniprot"]].append((int(r["jsonl_pos"]),
                                                int(r["lmdb_pos"]),
                                                int(r["label"]), r["mol_id"]))
            self.idx[L] = d

    def band(self, up):
        r = self.mir.get(up)
        if not r:
            return "no row"
        if float(r["self_hit"]) > 0:
            return "in training"        # the target itself is in the training set
        v = r["identity"]
        if not v:
            return "no hit"
        i = float(v)
        return ">=60%" if i >= 0.60 else "20-60%" if i >= 0.20 else "<20%"

    def npz(self, model):
        path = self.a.ckpt2 if (model == "ligunity_pocket_ranking" and
                                self.a.ckpt2) else f"{self.a.npz_dir}/T3_{model}.npz"
        return np.load(path, allow_pickle=True)

    def scored(self, model):
        """(layer, uniprot) -> (preds, labels). No alignment: EF needs none."""
        d, out = self.npz(model), {}
        for k in d.keys():
            if not k.endswith("/preds"):
                continue
            _, L, up, _ = k.split("/")
            if (L, up) not in self.sub:
                continue
            p = d[k].reshape(-1).astype(float)
            y = d[f"T3/{L}/{up}/labels"].reshape(-1).astype(int)
            if len(p) == len(y):
                out[(L, up)] = (p, y)
        return out

    def aligned(self, model):
        """(layer, uniprot) -> (mol_ids, labels, preds), FAIL targets dropped.
        Required by anything that joins per-molecule data."""
        out = {}
        for (L, up), (p, y) in self.scored(model).items():
            verdict = self.order.get((model, L, up))
            if verdict is None or verdict == "FAIL":
                continue
            pool = self.idx[L].get(up) or []
            rows = [r for r in pool if r[1] >= 0] if verdict == "lmdb" else pool
            if len(rows) != len(y):
                continue
            col = 1 if verdict == "lmdb" else 0
            lab = np.full(len(rows), -1, dtype=int)
            mol = [None] * len(rows)
            ok = True
            for r in rows:
                i = r[col]
                if not 0 <= i < len(rows):
                    ok = False
                    break
                lab[i], mol[i] = r[2], r[3]
            if ok and (lab == y).all():
                out[(L, up)] = (mol, lab, p)
        return out

    def tier_cells(self, model):
        """(layer, uniprot, tier) -> EF_tier, plus per-target hit/total."""
        out = {}
        for (L, up), (mol, lab, p) in self.aligned(model).items():
            k = int(math.ceil(FRAC * len(lab)))
            top = set(np.argsort(-p)[:k].tolist())
            tot, hit = collections.Counter(), collections.Counter()
            for i in range(len(lab)):
                if lab[i] != 1 or mol[i] is None or mol[i] not in self.sim:
                    continue
                t = tier_of(self.sim[mol[i]])
                if t is None:
                    continue
                tot[t] += 1
                if i in top:
                    hit[t] += 1
            for t in tot:
                if tot[t] >= MIN_T:
                    out[(L, up, t)] = (hit[t] / tot[t]) / FRAC
        return out


def bh(ps):
    ps = np.asarray(ps, dtype=float)
    n, o = len(ps), np.argsort(ps)
    q, run = np.empty(n), 1.0
    for i in range(n - 1, -1, -1):
        run = min(run, ps[o[i]] * n / (i + 1))
        q[o[i]] = run
    return q


def selfcheck(D):
    """Reproduce two published tables, and pin the similarity direction.
    Returns a list of failures; empty means the pipeline is trustworthy."""
    bad = []
    pub = collections.defaultdict(dict)
    for r in csv.DictReader(open(D.a.main_table)):
        pub[(r["layering"], r["model"])][r["layer"]] = (int(r["n_targets"]),
                                                        float(r["ef1"]))
    relabel = {r["uniprot"] for r in csv.DictReader(open(D.a.mirroring_plain))
               if r["identity"] and float(r["identity"]) >= 0.40}
    n_ok = 0
    for m in MODELS:
        ef = {t: enrichment_factor(p, y, FRAC)
              for t, (p, y) in D.scored(m).items()}
        for mode, rel in (("original", set()), ("corrected", relabel)):
            agg = collections.defaultdict(list)
            for (L, up), v in ef.items():
                agg["L3" if (L == "L4" and up in rel) else L].append(v)
            for L in LAYERS:
                if L not in agg:
                    continue
                want = pub.get((mode, m), {}).get(L)
                if not want:
                    continue
                if abs(float(np.mean(agg[L])) - want[1]) > 5e-4 or len(agg[L]) != want[0]:
                    bad.append(f"main table {m} {mode} {L}: "
                               f"{np.mean(agg[L]):.4f}/{len(agg[L])} != "
                               f"{want[1]:.4f}/{want[0]}")
                else:
                    n_ok += 1
    pubt = {(r["model"], r["layer"], r["tier"]): (int(r["n_targets"]),
                                                  float(r["ef_tier"]))
            for r in csv.DictReader(open(D.a.tiered_table))}
    n_tier = 0
    for m in sorted({k[0] for k in pubt}):
        cells = D.tier_cells(m)
        byl = collections.defaultdict(lambda: collections.defaultdict(list))
        for (L, _up, t), v in cells.items():
            byl[L][t].append(v)
        for L in LAYERS:
            for _, _, t in TIERS:
                v = byl[L].get(t)
                want = pubt.get((m, L, t))
                if not v or not want:
                    continue
                if abs(float(np.mean(v)) - want[1]) > 5e-3 or len(v) != want[0]:
                    bad.append(f"tiered {m} {L} {t}: {np.mean(v):.2f}/{len(v)}"
                               f" != {want[1]:.2f}/{want[0]}")
                else:
                    n_tier += 1
    # direction: mean rank-percentile must rise with the published EF.
    # Checked on a strong AND a weak model: a strong model's monotonicity is
    # robust enough to survive a mispairing, whereas a weak model's
    # percentiles sit near 0.5 and are far more sensitive to it.
    for m in ("ligunity_protein_ranking", "sprint"):
        acc = collections.defaultdict(list)
        for (L, up), (mol, lab, p) in D.aligned(m).items():
            pct = stats.rankdata(p) / len(p)
            for i in range(len(lab)):
                if lab[i] == 1 and mol[i] is not None and mol[i] in D.sim:
                    t = tier_of(D.sim[mol[i]])
                    if t:
                        acc[t].append(pct[i])
        means = [np.mean(acc[t]) for _, _, t in TIERS if acc.get(t)]
        if len(means) < len(TIERS):
            bad.append(f"similarity direction ({m}): only {len(means)} of "
                       f"{len(TIERS)} tiers populated")
        elif means != sorted(means):
            bad.append(f"similarity direction ({m}): rank-percentile not "
                       f"monotone across published tiers "
                       f"({[round(x,3) for x in means]}) -- is --sim-column a "
                       f"distance rather than a similarity?")
    print(f"selfcheck: {n_ok} main-table cells and {n_tier} tiered cells "
          f"reproduced; direction {'OK' if not any('direction' in b for b in bad) else 'FAILED'}")
    return bad


def cmd_map(D, args):
    rows = [["model", "protein_band", "ligand_tier", "n_targets",
             "ef_mean", "ef_se", "thin_n_lt_%d" % THIN]]
    for m in MODELS:
        acc = collections.defaultdict(lambda: collections.defaultdict(list))
        for (L, up, t), v in D.tier_cells(m).items():
            acc[D.band(up)][t].append(v)
        print(f"\n{m}")
        print("%-14s" % "band" + "".join("%20s" % t[2] for t in TIERS))
        for b in BANDS:
            line = "%-14s" % b
            for _, _, t in TIERS:
                v = acc[b].get(t)
                if not v:
                    line += "%20s" % "--"
                    continue
                se = np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
                line += "%20s" % ("%.1f±%.1f (%d)%s" % (np.mean(v), se, len(v),
                                                        "*" if len(v) < THIN else ""))
                rows.append([m, b, t, len(v), "%.4f" % np.mean(v),
                             "%.4f" % se, int(len(v) < THIN)])
            print(line)
    if args.out:
        with open(args.out, "w", newline="") as f:
            csv.writer(f).writerows(rows)
        print(f"\nwrote {args.out} ({len(rows)-1} cells)")


def cmd_tests(D, args):
    recs = []
    for m in MODELS:
        acc = collections.defaultdict(lambda: collections.defaultdict(list))
        for (L, up, t), v in D.tier_cells(m).items():
            acc[D.band(up)][t].append(v)
        for _, _, t in TIERS:
            a = acc["in training"].get(t) or []
            groups = [acc[b].get(t) or [] for b in NOVEL_BANDS]
            pooled = [x for g in groups for x in g]
            if len(a) < 3 or len(pooled) < 3:
                continue
            p_step = stats.mannwhitneyu(a, pooled, alternative="two-sided").pvalue
            ok = [g for g in groups if len(g) >= 3]
            p_grad = stats.kruskal(*ok).pvalue if len(ok) >= 2 else float("nan")
            recs.append(dict(model=m, tier=t, ef_in=float(np.mean(a)),
                             ef_not=float(np.mean(pooled)), p_step=p_step,
                             p_grad=p_grad,
                             min_n=min(len(g) for g in groups)))
    qs, qg = bh([r["p_step"] for r in recs]), bh([r["p_grad"] for r in recs])
    for i, r in enumerate(recs):
        r["q_step"], r["q_grad"] = qs[i], qg[i]
    print("%-24s %-12s %7s %7s %9s %9s %5s" %
          ("model", "tier", "ef_in", "ef_not", "q_step", "q_grad", "minN"))
    for r in recs:
        print("%-24s %-12s %7.1f %7.1f %9.2g %9.2g %5d %s" %
              (r["model"], r["tier"], r["ef_in"], r["ef_not"], r["q_step"],
               r["q_grad"], r["min_n"], "STEP" if r["q_step"] < 0.05 else ""))
    n = len(recs)
    print(f"\ncells {n}   step BH q<0.05: {sum(r['q_step']<0.05 for r in recs)}"
          f"   gradient BH q<0.05: {sum(r['q_grad']<0.05 for r in recs)}"
          f"   (expected false positives at .05: {0.05*n:.1f})")
    print(f"cells where every novel band has n>={THIN}: "
          f"{sum(r['min_n']>=THIN for r in recs)}/{n}   "
          f"median min-n {np.median([r['min_n'] for r in recs]):.0f}")
    print("\nthe step is 'was this target in the training set'; the gradient is\n"
          "identity to the nearest training homologue among sequence-novel targets")
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(recs[0]))
            w.writeheader()
            w.writerows(recs)
        print(f"wrote {args.out}")


def cmd_chem(D, args):
    """Continuous chemical resolution curve. Per target, so the pooled mixture
    cannot masquerade as a within-target trend."""
    print("per-target Spearman(similarity, rank-percentile inside that pool)")
    print("positive = an active closer to the training ligands ranks higher\n")
    print("%-24s %6s %9s %7s %7s %10s" %
          ("model", "n_tgt", "median", "IQR", "neg%", "wilcoxon"))
    for m in MODELS:
        rhos = []
        for (L, up), (mol, lab, p) in D.aligned(m).items():
            pct = stats.rankdata(p) / len(p)
            v = [(D.sim[mol[i]], pct[i]) for i in range(len(lab))
                 if lab[i] == 1 and mol[i] is not None and mol[i] in D.sim]
            if len(v) < args.min_actives:
                continue
            x = np.array([a for a, _ in v])
            y = np.array([b for _, b in v])
            if len(set(x)) < 3:
                continue
            r, _ = stats.spearmanr(x, y)
            if np.isfinite(r):
                rhos.append(r)
        if len(rhos) < 10:
            print("%-24s %6d  too few" % (m, len(rhos)))
            continue
        rh = np.array(rhos)
        q1, q3 = np.percentile(rh, [25, 75])
        print("%-24s %6d %+9.3f %7.3f %6.0f%% %10.2g" %
              (m, len(rh), np.median(rh), q3 - q1, 100 * (rh < 0).mean(),
               stats.wilcoxon(rh).pvalue))
    print("\n⚠️ a nonzero share of targets runs the other way -- this is a\n"
          "   trend over targets, not a law that holds on each one")


def cmd_seqpkt(D, args):
    """Sequence identity vs pocket distance as predictors of per-target EF.
    Run twice: the in-training step would otherwise pose as sequence power."""
    pv = {}
    for r in csv.DictReader(open(args.pocket_neighbors)):
        if r["d_train"] not in ("", "nan"):
            pv[(r["layer"], r["uniprot"])] = (float(r["d_train"]),
                                              float(r["d_control"]))
    for label, novel_only in (("ALL targets with PocketVec", False),
                              ("sequence-novel only", True)):
        print(f"\n=== {label} ===")
        print("%-24s %5s %16s %16s" % ("model", "n", "rho(EF,seq_id)",
                                       "rho(EF,pocket)"))
        seqs, pkts = [], []
        for m in MODELS:
            rows = []
            for (L, up), (p, y) in D.scored(m).items():
                if (L, up) not in pv or up not in D.mir:
                    continue
                if novel_only and float(D.mir[up]["self_hit"]) > 0:
                    continue
                ident = D.mir[up]["identity"]
                d_tr, d_ct = pv[(L, up)]
                rows.append((enrichment_factor(p, y, FRAC),
                             float(ident) if ident else 0.0, d_ct - d_tr))
            if len(rows) < 12:
                print("%-24s %5d  too few" % (m, len(rows)))
                continue
            ef = np.array([r[0] for r in rows])
            sid = np.array([r[1] for r in rows])
            pk = np.array([r[2] for r in rows])
            r1, p1 = stats.spearmanr(ef, sid)
            r2, p2 = stats.spearmanr(ef, pk)
            seqs.append(r1)
            pkts.append(r2)
            print("%-24s %5d %+8.3f p=%-5.3g %+8.3f p=%-5.3g" %
                  (m, len(rows), r1, p1, r2, p2))
        if seqs:
            print("median across models:  seq %+.3f   pocket %+.3f   "
                  "|seq|>|pocket| in %d/%d" %
                  (np.median(seqs), np.median(pkts),
                   sum(abs(a) > abs(b) for a, b in zip(seqs, pkts)), len(seqs)))


def cmd_paralog(D, args):
    """Within-family selectivity. Uses rank-percentile inside each target:
    raw score differences are not safe, because scores are not on a common
    scale across targets (measured, not assumed -- see --precheck)."""
    act = collections.defaultdict(dict)
    sub_up = {u for _, u in D.sub}
    with gzip.open(args.actives, "rt") as fh:
        for r in csv.DictReader(fh):
            if r["uniprot"] in sub_up:
                act[r["uniprot"]][r["mol_id"]] = float(r["paff"])
    pairs = []
    for r in csv.DictReader(open(args.redundancy)):
        a, b = r["uniprot_a"], r["uniprot_b"]
        if a not in act or b not in act or float(r["identity"]) < args.min_identity:
            continue
        shared = set(act[a]) & set(act[b])
        if len(shared) < args.min_shared:
            continue
        d = [abs(act[a][s] - act[b][s]) for s in shared]
        if sum(d) / len(d) >= args.selective_delta:
            pairs.append((float(r["identity"]), a, b, shared))
    print(f"same-family pairs with a differential-activity signal: {len(pairs)}")
    print("\nrho(delta rank-percentile, delta pAff) per pair")
    print("positive = the model prefers the target the ligand is stronger on\n")
    total_pos = total = 0
    for m in MODELS:
        pct = {}
        for (L, up), (mol, lab, p) in D.aligned(m).items():
            pr = stats.rankdata(p) / len(p)
            pct.setdefault(up, {}).update(
                {mid: pr[i] for i, mid in enumerate(mol) if mid is not None})
        cells, pos = [], 0
        for ident, a, b, shared in pairs:
            if a not in pct or b not in pct:
                continue
            com = [s for s in shared if s in pct[a] and s in pct[b]]
            if len(com) < args.min_shared:
                continue
            dr = np.array([pct[a][s] - pct[b][s] for s in com])
            dp = np.array([act[a][s] - act[b][s] for s in com])
            if len(set(dp)) < 3:
                continue
            r, _ = stats.spearmanr(dr, dp)
            if np.isfinite(r):
                cells.append((a, b, r, len(com)))
                pos += r > 0
        if not cells:
            print("%-24s  no usable pairs" % m)
            continue
        total_pos += pos
        total += len(cells)
        print("%-24s %d/%d positive   median rho %+.3f" %
              (m, pos, len(cells), np.median([c[2] for c in cells])))
    if total:
        print(f"\nall {total} model-pair cells: {total_pos} positive "
              f"({100*total_pos/total:.0f}%)")
    print(f"\n⚠️ usable pairs per model are few; report the per-pair table and a\n"
          f"   direction count, not a significance claim. A sign test over k\n"
          f"   pairs cannot go below 2/2^k.")


def cmd_frontier(D, args):
    """The degenerate capability frontier: a contour needs two resolved axes,
    and the protein axis carries one bit, so this gives two regimes instead."""
    acc = {}
    for m in MODELS:
        a = collections.defaultdict(lambda: collections.defaultdict(list))
        for (L, up, t), v in D.tier_cells(m).items():
            a[D.band(up)][t].append(v)
        for _, _, t in TIERS:
            ins = a["in training"].get(t) or []
            out = [x for b in NOVEL_BANDS for x in (a[b].get(t) or [])]
            if ins and out:
                acc[(m, t)] = (float(np.mean(ins)), float(np.mean(out)))
    for th in args.thresholds:
        print(f"\n=== EF@1% >= {th} ===")
        print("%-24s %-34s %-34s" % ("model", "in training: tiers still met",
                                     "not in training: tiers still met"))
        for m in MODELS:
            got = []
            for which in (0, 1):
                met = [t for _, _, t in TIERS
                       if (m, t) in acc and acc[(m, t)][which] >= th]
                got.append(", ".join(met) if met else "(none)")
            print("%-24s %-34s %-34s" % (m, got[0], got[1]))
    print("\n'(none)' = in that protein regime the model never reaches this EF,\n"
          "however familiar the chemistry. Thresholds are exogenous: several are\n"
          "shown rather than one being chosen here.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["map", "tests", "chem", "seqpkt",
                                     "paralog", "frontier", "all"])
    ap.add_argument("--npz-dir", required=True,
                    help="packaged per-molecule scores, one .npz per model. "
                         "REQUIRED deliberately, with no default: the archive "
                         "handed out on 2026-09-13 carries LigUnity-pocket's "
                         "FIRST checkpoint under the current name, so a "
                         "convenient default would quietly mix checkpoints. "
                         "Point --ckpt2 at the current-checkpoint .npz if the "
                         "directory you pass is that archive.")
    ap.add_argument("--ckpt2", default=None,
                    help="LigUnity-pocket's current-checkpoint .npz, if the "
                         "packaged one under --npz-dir is the first checkpoint")
    ap.add_argument("--subset", default=f"{B}/results/export/T3_vsds_matched.csv")
    ap.add_argument("--index-glob",
                    default=f"{B}/results/export/frozen/T3_index_{{layer}}.csv.gz")
    ap.add_argument("--molecules",
                    default=f"{B}/results/export/frozen/T3_molecules.csv.gz")
    ap.add_argument("--model-order",
                    default=f"{B}/results/export/frozen/T3_model_order.csv")
    ap.add_argument("--sim-column", default="novelty_pocketaffdb",
                    help="max Tanimoto to a training ligand set -- a SIMILARITY")
    ap.add_argument("--mirroring",
                    default=f"{B}/results/export/T3_target_mirroring_union.csv",
                    help="the union table; the plain one understates homology")
    ap.add_argument("--mirroring-plain",
                    default=f"{B}/results/export/T3_target_mirroring.csv",
                    help="used only to reproduce the published corrected layering")
    ap.add_argument("--main-table",
                    default=f"{B}/results/export/T3_main_vsds_subset.csv")
    ap.add_argument("--tiered-table",
                    default=f"{B}/results/export/T3_novelty_tiered_ef_subset.csv")
    ap.add_argument("--pocket-neighbors",
                    default=f"{B}/results/export/T3_pocket_neighbors.csv")
    ap.add_argument("--redundancy",
                    default=f"{B}/results/export/T3_target_redundancy.csv")
    ap.add_argument("--actives", default=f"{B}/data/release/actives.csv.gz")
    ap.add_argument("--min-actives", type=int, default=10)
    ap.add_argument("--min-identity", type=float, default=0.30)
    ap.add_argument("--min-shared", type=int, default=5)
    ap.add_argument("--selective-delta", type=float, default=1.0)
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[20, 10, 5, 2])
    ap.add_argument("--out", default=None)
    ap.add_argument("--eval-dir", default=f"{B}/eval",
                    help="directory holding the repository's metrics.py")
    ap.add_argument("--no-selfcheck", action="store_true",
                    help="skip reproducing the published tables. Only for "
                         "debugging -- never for producing numbers to report.")
    args = ap.parse_args()

    global enrichment_factor
    enrichment_factor = resolve_ef(args.eval_dir)

    D = Data(args)
    if not args.no_selfcheck:
        bad = selfcheck(D)
        if bad:
            print("\n⛔ selfcheck failed; refusing to report numbers:",
                  file=sys.stderr)
            for b in bad[:20]:
                print("   " + b, file=sys.stderr)
            # Point at the cause, not just the symptom: if every failure sits
            # on one model and that model is LigUnity-pocket, the likely reason
            # is the checkpoint the score package carries.
            hit = {b.split()[2] for b in bad if b.startswith("main table")}
            if hit == {"ligunity_pocket_ranking"} and not args.ckpt2:
                print("\n   ⚠️ every failure is on LigUnity-pocket and --ckpt2 "
                      "was not given.\n   The 2026-09-13 archive ships that "
                      "model's FIRST checkpoint under the current\n   name "
                      "(sha d0034aaf…, which the repo manifest calls "
                      "'_ckpt1'), while the\n   published tables are the "
                      "current one (sha 881f65ef…). Pass --ckpt2.",
                      file=sys.stderr)
            sys.exit(1)

    jobs = ["map", "tests", "chem", "seqpkt", "paralog", "frontier"] \
        if args.what == "all" else [args.what]
    for j in jobs:
        print("\n" + "=" * 78)
        print(j)
        print("=" * 78)
        globals()[f"cmd_{j}"](D, args)


if __name__ == "__main__":
    main()
