"""Select from T3 a subset with "actives >=50 per target and class composition matched to VSDS-vd".

Background
----------
Two decisions the advisor made on 2026-09-04: (1) each target must have
>=50 active molecules; (2) target class composition should reference
VSDS-vd (Gu et al., Nat Mach Intell 7:509-520, 2025,
DOI 10.1038/s42256-025-00993-0).

VSDS-vd's class composition isn't a ready-made table from the paper --
it was produced by downloading the dataset from Zenodo
(https://zenodo.org/records/13684010), taking the 147 UniProt accessions
of the DTEBV-D subset, and re-annotating them with **the exact same**
ChEMBL classification-tree convention as this repo's
annotate_target_class3.py. Using the same convention on both sides is the
only thing that makes this comparison meaningful.

Two gaps that can't be fixed
-------------------------------
Nuclear receptors and P450 only have 6 and 2 members respectively in the
>=50 pool, while VSDS-vd's proportions would call for 17 and 9. **Taking
every single one still isn't enough** -- and this isn't a filtering
artifact: these two families were studied early, have few members, and
essentially produce no new targets after the time split (both are 0 in
L3+L4). This is reported honestly as a limitation rather than shrinking
the whole subset down to 49 (P450's ceiling) just to balance these two
classes.

Why both "class" and "layer" need to be balanced at once
------------------------------------------------------------
The first version quota'd by class alone, packing L3/L4 first within each
class, and as a result the "other enzymes" class's 46 slots were entirely
consumed by the L4 layer, leaving that largest class at 0 in L1/L2 --
composition across layers became completely unbalanced, making a
layer-wise comparison meaningless. Now it uses capacity-bounded iterative
proportional fitting (IPF): row margins = VSDS-vd's class proportions,
column margins = the candidate pool's own layer proportions, each cell
capped by actual inventory, with final rounding by the largest-remainder
method.

Why random sampling within a class rather than "take the ones with the most actives"
------------------------------------------------------------------------------------
Both approaches can fill the quota, but selecting by active count would
systematically favor well-studied, popular targets, and would further
widen the spread of per-target active counts (up to 3262 in the pool).
Fixed-seed random sampling keeps the active-count distribution
undistorted.
"""
import argparse
import collections
import csv
import json
import random

# Class composition of VSDS-vd's DTEBV-D 147 targets (re-annotated under the same convention; see the module docstring)
VSDS = {"激酶": 35, "其他酶": 27, "GPCR": 23, "蛋白酶": 23, "核受体": 11,
        "表观": 10, "其他/未分类": 7, "P450": 6, "离子通道": 3, "转运体": 2}
ORDER = ["激酶", "其他酶", "GPCR", "蛋白酶", "核受体", "表观",
         "离子通道", "P450", "转运体", "其他/未分类"]
LAYERS = ["L1", "L2", "L3", "L4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="results/T3_targets.csv")
    ap.add_argument("--min-actives", type=int, default=50)
    ap.add_argument("--quota", type=int, default=250,
                    help="名义配额；供给不足的类别会少于配额，所以实得数更小")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/T3_vsds_matched.csv")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.targets))
            if int(r["n_actives"]) >= args.min_actives]
    nv = sum(VSDS.values())
    p = {c: VSDS[c] / nv for c in VSDS}

    pool = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        pool[r["protein_class"]][r["layer"]].append(r)

    cap = {(c, L): len(pool[c].get(L, [])) for c in ORDER for L in LAYERS}
    # L3 is the thinnest layer (new target / family already seen), with only
    # about twenty in the pool to begin with; proportional allocation would
    # leave it with single digits and effectively void that layer, so the
    # whole layer is kept as-is and excluded from proportional allocation.
    n_layer = collections.Counter(r["layer"] for r in rows)
    rest = [L for L in LAYERS if L != "L3"]
    n_rest = sum(n_layer[L] for L in rest)
    q_layer = {"L3": float(n_layer["L3"])}
    for L in rest:
        q_layer[L] = (args.quota - n_layer["L3"]) * n_layer[L] / n_rest
    q_class = {c: args.quota * p.get(c, 0) for c in ORDER}

    # Capacity-bounded iterative proportional fitting: rows = class, columns = layer, each cell capped by inventory
    x = {k: min(cap[k], 1.0) for k in cap}
    for _ in range(200):
        for c in ORDER:                                   # normalize rows
            s_ = sum(x[(c, L)] for L in LAYERS)
            if s_ > 0:
                for L in LAYERS:
                    x[(c, L)] = min(cap[(c, L)], x[(c, L)] * q_class[c] / s_)
        for L in LAYERS:                                  # normalize columns
            s_ = sum(x[(c, L)] for c in ORDER)
            if s_ > 0:
                for c in ORDER:
                    x[(c, L)] = min(cap[(c, L)], x[(c, L)] * q_layer[L] / s_)

    # Round by the largest-remainder method, still bounded by inventory
    alloc = {k: min(cap[k], int(v)) for k, v in x.items()}
    frac = sorted(((x[k] - int(x[k]), k) for k in x if alloc[k] < cap[k]), reverse=True)
    need = round(sum(min(cap[k], x[k]) for k in x)) - sum(alloc.values())
    for _, k in frac[:max(0, need)]:
        alloc[k] += 1

    rng = random.Random(args.seed)
    keep, short = [], []
    for c in ORDER:
        got_c = 0
        for L in LAYERS:
            avail = list(pool[c].get(L, []))
            rng.shuffle(avail)
            keep += avail[:alloc[(c, L)]]
            got_c += alloc[(c, L)]
        if got_c < round(q_class[c]):
            short.append((c, round(q_class[c]), got_c))

    n = len(keep)
    got = collections.Counter(r["protein_class"] for r in keep)
    lay = collections.Counter(r["layer"] for r in keep)

    print(f"活性门槛 ≥{args.min_actives}：候选池 {len(rows)} 个靶点")
    print(f"名义配额 {args.quota} → 实得 {n} 个\n")
    print("%-11s %9s %14s %8s" % ("类别", "VSDS-vd", "本子集", "偏差"))
    print("-" * 46)
    for c in ORDER:
        pv, pb = 100 * p.get(c, 0), 100 * got[c] / n
        print("%-11s %8.1f%% %14s %+7.1f" %
              (c, pv, f"{got[c]} ({pb:.0f}%)", pb - pv))
    print("-" * 46)
    dev = max(abs(100 * got[c] / n - 100 * p.get(c, 0)) for c in ORDER)
    print("%-11s %9s %14s %+7.1fpp" % ("最大偏差", "", "", dev))

    print("\n分层：" + " · ".join(f"{L} {lay[L]}" for L in LAYERS))
    na = sorted(int(r["n_actives"]) for r in keep)
    print(f"每靶点活性数：最小 {na[0]} · 中位 {na[len(na)//2]} · 最大 {na[-1]}"
          f"   （VSDS-vd 中位 28）")

    if short:
        print("\n⚠️ 供给不足、已全部取用的类别（写进 limitation）：")
        for c, q, g in short:
            print(f"   {c:<8} 需 {q:>3}，池子里只有 {g:>3}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(keep, key=lambda r: (r["layer"], r["protein_class"],
                                                r["uniprot"])))
    print(f"\n靶点清单写入 {args.out}")


if __name__ == "__main__":
    main()
