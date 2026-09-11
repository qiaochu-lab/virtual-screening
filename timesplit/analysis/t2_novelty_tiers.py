"""T2 ranking ability × ligand novelty — separating "the target got novel"
from "the molecule got novel".

Why this is needed
------------
T2 currently only reports L1-L4, i.e. stratified solely by **target**
novelty. But 53.9% of L1's actives are near-duplicates of a training
ligand (Tanimoto >= 0.7), while L4 has only 6.4% — so as ρ drops from L1's
+0.26 to L4's +0.10, two things are changing at once, and we've been
attributing all of it to the target.

The T1/T3 side already separated this same confound once: the main table's
L4 enrichment of 9.38 splits into "27.1 on chemistry it has seen / 4.5 on
completely novel chemistry" once tiered by ligand novelty. T2 has never done
this split. This script fills that gap.

Conventions
------------
* Novelty = this molecule's maximum Tanimoto (ECFP4) to a training-set
  ligand, in the same four tiers as `ligand_novelty.py` /
  `novelty_tiered_ef.py`, reusing their cache directly.
* **Within each target**, Spearman(model score, measured pAff) is computed
  using only the actives that fall in the same tier; if a tier has fewer
  than --min-tier actives for that target, that target's tier is skipped.
* Then average over targets. A pooled column is also given (same set of
  targets, not tiered), so "after tiering" and "before tiering" are compared
  on the same set of targets.

⚠️ Two checks that must be enforced
------------
1. **Molecule order.** The model reads from lmdb, whose cursor order is
   lexicographic (0,1,10,100,…), different from the eval-set jsonl order but
   the same length; comparing only lengths silently mismatches them — this
   pitfall has bitten this project three times already (see PATCHES.md).
   This reuses `novelty_tiered_ef.py`'s hard check: after reconstructing the
   order, it must verify "the positions labelled 1 really are this target's
   actives"; if that fails, skip and report it.
2. **Every tier's n must be printed.** Spearman is far more sensitive to
   small samples than EF — on T3, L1's "novel <0.35" tier has only 18
   targets, the thinnest cell in the whole table — reporting ρ without n
   would let noise pass for a conclusion.

Known limitation
------------
Novelty is computed relative to **PocketAffDB's ligands**
(`train_label_blend_seq_full.json`), not relative to each model's own
training ligands. So for models not trained on PocketAffDB (the DrugCLIP
family, ConPLex, ConGLUDe, SPRINT), this tiering is only an approximation.
A per-model version would need each model's own training-ligand list, which
is currently only available for SPRINT and ConPLex.
"""
import argparse
import collections
import csv
import json
import os
import pickle

import numpy as np
from scipy import stats

B = "/data/work/vs-benchmark"
TIERS = [(0.0, 0.35, "全新 <0.35"), (0.35, 0.50, "远 0.35–0.5"),
         (0.50, 0.70, "近 0.5–0.7"), (0.70, 1.01, "极近 ≥0.7")]
TIER_NAMES = [t[2] for t in TIERS]


def tier_of(v):
    for lo, hi, name in TIERS:
        if lo <= v < hi:
            return name
    return None


def model_order(up, L, n, rec, labels):
    """Reconstruct the molecule order the model saw and hard-verify it; returns None if it doesn't match."""
    act = {x["smiles"] for x in rec["actives"]}

    def ok(seq):
        if seq is None or len(seq) != n:
            return None
        got = {seq[i] for i in range(n) if labels[i] == 1}
        return seq if got == act else None

    r = ok([x["smiles"] for x in rec["actives"]] +
           [x["smiles"] for x in rec["decoys"]])
    if r is not None:
        return r
    path = f"{B}/data/T3_6A/{L}/{up}/{up}_lig.lmdb"
    if not os.path.exists(path):
        return None
    try:
        import lmdb
        e = lmdb.open(path, subdir=False, readonly=True, lock=False)
        out = []
        with e.begin() as t:
            for _k, v in t.cursor():
                out.append(pickle.loads(v)["smi"])
        e.close()
    except Exception:
        return None
    return ok(out)


def rho(pairs):
    """pairs = [(score, paff)]; returns None if variance is 0 or there are too few points."""
    s = np.array([p[0] for p in pairs], dtype=float)
    a = np.array([p[1] for p in pairs], dtype=float)
    if np.std(s) == 0 or np.std(a) == 0:
        return None
    r = stats.spearmanr(s, a).statistic
    return float(r) if np.isfinite(r) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--novelty", default=f"{B}/data/t3/ligand_novelty.json")
    ap.add_argument("--eval-dir", default=f"{B}/data/t3/eval")
    ap.add_argument("--raw", default=f"{B}/results/t3_raw")
    ap.add_argument("--models", nargs="+",
                    default=["hypseek_rk", "ligunity_protein_ranking",
                             "ligunity_pocket_ranking", "litenclip",
                             "conglude", "drugclip"])
    ap.add_argument("--layers", nargs="+", default=["L1", "L2", "L3", "L4"])
    ap.add_argument("--min-tier", type=int, default=5,
                    help="该靶点该档至少这么多活性才算；Spearman 比 EF 对小 n 敏感，"
                         "所以比 novelty_tiered_ef.py 的 3 高")
    ap.add_argument("--subset", default=None)
    ap.add_argument("--out", default=f"{B}/results/export/T2_novelty_tiers.csv")
    ap.add_argument("--out-paired", default=f"{B}/results/export/T2_novelty_paired.csv")
    args = ap.parse_args()

    keep = None
    if args.subset:
        keep = {(r["layer"], r["uniprot"]) for r in csv.DictReader(open(args.subset))}
        print(f"子集过滤：{len(keep)} 条")

    nov = json.load(open(args.novelty))
    recs = {L: [json.loads(x) for x in open(f"{args.eval_dir}/{L}.jsonl")]
            for L in args.layers}

    rows = [["model", "layer", "tier", "n_targets", "n_actives_median",
             "spearman_mean", "spearman_sem", "frac_positive"]]
    prows = [["model", "layer", "contrast", "n_paired_targets",
              "rho_familiar", "rho_novel", "delta", "wilcoxon_p",
              "n_familiar_wins"]]

    print("\nT2 排序能力 × 配体新颖度")
    print(f"（每靶点每档 ≥{args.min_tier} 个活性才计入；pooled = 同一批靶点不分档）")
    print("=" * 108)

    for m in args.models:
        # bucket[layer][tier] = [per-target ρ]; paired[layer] = [(familiar ρ, novel ρ)]
        bucket = collections.defaultdict(lambda: collections.defaultdict(list))
        nact = collections.defaultdict(lambda: collections.defaultdict(list))
        pooled = collections.defaultdict(list)
        paired = collections.defaultdict(list)
        paired_half = collections.defaultdict(list)
        n_bad = collections.Counter()

        for L in args.layers:
            d = f"{args.raw}/{m}/T3/{L}"
            if not os.path.isdir(d):
                d = f"{args.raw}/{m}/{L}"
            if not os.path.isdir(d):
                continue
            for r in recs[L]:
                up = r["uniprot"]
                if keep is not None and (L, up) not in keep:
                    continue
                try:
                    p = np.load(f"{d}/{up}/saved_preds.npy").reshape(-1)
                    y = np.load(f"{d}/{up}/saved_labels.npy")
                except Exception:
                    continue
                if len(p) != len(y):
                    continue
                order = model_order(up, L, len(y), r, y)
                if order is None:
                    n_bad[L] += 1
                    continue
                aff = {a["smiles"]: float(a["paff"]) for a in r["actives"]}

                by_tier = collections.defaultdict(list)  # tier name -> [(score, paff)]
                allp = []
                for i in range(len(y)):
                    if y[i] != 1:
                        continue
                    smi = order[i]
                    if smi not in aff:
                        continue
                    t = tier_of(nov.get(smi, -1))
                    if t is None:
                        continue
                    by_tier[t].append((float(p[i]), aff[smi]))
                    allp.append((float(p[i]), aff[smi]))

                # Two halves: novel <0.5 / familiar >=0.5. Very few targets have both
                # extreme tiers reach 5 actives at once, so per-target pairing on
                # them alone would collapse to n=5-9; splitting into halves brings
                # the paired sample back up to the tens.
                half = {"生半 <0.5": by_tier[TIER_NAMES[0]] + by_tier[TIER_NAMES[1]],
                        "熟半 ≥0.5": by_tier[TIER_NAMES[2]] + by_tier[TIER_NAMES[3]]}
                hgot = {}
                for h, pr in half.items():
                    if len(pr) >= args.min_tier:
                        v = rho(pr)
                        if v is not None:
                            hgot[h] = v
                if len(hgot) == 2:
                    paired_half[L].append((hgot["熟半 ≥0.5"], hgot["生半 <0.5"]))

                got = {}
                for t, pr in by_tier.items():
                    if len(pr) < args.min_tier:
                        continue
                    v = rho(pr)
                    if v is None:
                        continue
                    bucket[L][t].append(v)
                    nact[L][t].append(len(pr))
                    got[t] = v
                if got:
                    v = rho(allp)
                    if v is not None:
                        pooled[L].append(v)
                # Per-target pairing: within the same target, familiar chemistry vs completely novel chemistry
                if TIER_NAMES[3] in got and TIER_NAMES[0] in got:
                    paired[L].append((got[TIER_NAMES[3]], got[TIER_NAMES[0]]))

        print(f"\n{m}")
        print("-" * 108)
        print("%-5s %14s" % ("层", "pooled") +
              "".join("%20s" % t for t in TIER_NAMES))
        for L in args.layers:
            if L not in bucket:
                continue
            pv = pooled[L]
            cells = []
            for t in TIER_NAMES:
                v = bucket[L].get(t)
                if not v:
                    cells.append("—")
                    continue
                a = np.array(v)
                sem = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else float("nan")
                cells.append(f"{a.mean():+.3f} (n={len(a)})")
                rows.append([m, L, t, len(a), int(np.median(nact[L][t])),
                             f"{a.mean():.4f}", f"{sem:.4f}",
                             f"{(a > 0).mean():.3f}"])
            print("%-5s %14s" % (L, f"{np.mean(pv):+.3f} (n={len(pv)})" if pv else "—")
                  + "".join("%20s" % c for c in cells))
            if pv:
                rows.append([m, L, "pooled", len(pv), "",
                             f"{np.mean(pv):.4f}",
                             f"{np.std(pv, ddof=1)/np.sqrt(len(pv)):.4f}"
                             if len(pv) > 1 else "",
                             f"{(np.array(pv) > 0).mean():.3f}"])

        # Paired test: means can be misleading, must pair per target
        for L in args.layers:
            pr = paired[L]
            if len(pr) < 5:
                continue
            fam = np.array([x[0] for x in pr])
            new = np.array([x[1] for x in pr])
            try:
                w = stats.wilcoxon(fam, new).pvalue
            except Exception:
                w = float("nan")
            prows.append([m, L, "极近≥0.7 vs 全新<0.35", len(pr),
                          f"{fam.mean():.4f}", f"{new.mean():.4f}",
                          f"{(fam - new).mean():.4f}", f"{w:.4g}",
                          int((fam > new).sum())])
            print(f"  配对 极近≥0.7 vs 全新<0.35  {L}: "
                  f"n={len(pr)}  {fam.mean():+.3f} vs {new.mean():+.3f}  "
                  f"Δ={fam.mean()-new.mean():+.3f}  p={w:.3g}  "
                  f"熟胜 {int((fam>new).sum())}/{len(pr)}")

        for L in args.layers:
            pr = paired_half[L]
            if len(pr) < 5:
                continue
            fam = np.array([x[0] for x in pr])
            new = np.array([x[1] for x in pr])
            try:
                w = stats.wilcoxon(fam, new).pvalue
            except Exception:
                w = float("nan")
            prows.append([m, L, "熟半≥0.5 vs 生半<0.5", len(pr),
                          f"{fam.mean():.4f}", f"{new.mean():.4f}",
                          f"{(fam - new).mean():.4f}", f"{w:.4g}",
                          int((fam > new).sum())])
            print(f"  配对 熟半≥0.5 vs 生半<0.5   {L}: "
                  f"n={len(pr)}  {fam.mean():+.3f} vs {new.mean():+.3f}  "
                  f"Δ={fam.mean()-new.mean():+.3f}  p={w:.3g}  "
                  f"熟胜 {int((fam>new).sum())}/{len(pr)}")

        if n_bad:
            print("  ⚠️ 分子顺序校验未通过而跳过：" +
                  "  ".join(f"{L} {n}" for L, n in sorted(n_bad.items())))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    with open(args.out_paired, "w", newline="") as f:
        csv.writer(f).writerows(prows)
    print(f"\n写入 {args.out}")
    print(f"写入 {args.out_paired}")
    print("\n读法：如果 pooled 的 L1→L4 衰减在分档之后大幅变小，"
          "说明原来那条曲线主要是配体新颖度在动，不是靶点新颖度。")


if __name__ == "__main__":
    main()
