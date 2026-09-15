"""Same-family target pairs that share measured ligands — the raw material for a
paralog-selectivity test.

Why this file exists
--------------------
The count "112 pairs, 31 with mean |ΔpAff| >= 1.0" was quoted in planning from an
ad-hoc script in /tmp. Finding 18 already taught this project what that costs: a
headline number with no producing script cannot be re-derived when the inputs
change. This is the producing script.

What it finds
-------------
For every pair of 350-quota targets whose mmseqs identity clears --min-identity,
count the ligands measured against **both** members and summarise |ΔpAff|. A pair
is usable for a selectivity test only if the same ligand has a different measured
affinity on the two targets — that difference is the label the test predicts.

⚠️ Score comparability is NOT established here
-----------------------------------------------
A selectivity test would compare `s(P_A, L) - s(P_B, L)` against
`pAff_A - pAff_B`. That assumes model scores are comparable *across targets*.
For dot-product contrastive models the two scores live in one space, so this is
plausible — but it is an assumption, not a measurement, and this project has been
wrong before by assuming rather than checking. Before using the Δ of raw scores,
compare the two targets' score distributions over the shared ligands; if one is
systematically shifted, use ranks or within-target centring instead.

Note the shared ligands sit at *different positions* in the two targets' candidate
pools (e.g. HDAC1/HDAC2 share 408 actives, one at pool index 640 and 229), so the
join must go through results/frozen/T3_index_*.csv.gz rather than by position.
"""
import argparse
import collections
import csv
import json

B = "/data/work/vs-benchmark"


def actives_by_target(eval_dir, keep):
    """uniprot -> {smiles: pAff} for actives of the kept targets."""
    act = collections.defaultdict(dict)
    for layer in ("L1", "L2", "L3", "L4"):
        for line in open(f"{eval_dir}/{layer}.jsonl"):
            r = json.loads(line)
            if r["uniprot"] not in keep:
                continue
            for a in r["actives"]:
                act[r["uniprot"]][a["smiles"]] = float(a["paff"])
    return act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=f"{B}/results/export/T3_vsds_matched.csv")
    ap.add_argument("--redundancy", default=f"{B}/results/export/T3_target_redundancy.csv")
    ap.add_argument("--eval-dir", default=f"{B}/data/t3/eval")
    ap.add_argument("--min-identity", type=float, default=0.30)
    ap.add_argument("--min-shared", type=int, default=5)
    ap.add_argument("--selective-delta", type=float, default=1.0,
                    help="mean |dpAff| at or above this counts as usable signal")
    ap.add_argument("--expect-pairs", type=int, default=None)
    ap.add_argument("--expect-selective", type=int, default=None)
    args = ap.parse_args()

    keep = {r["uniprot"] for r in csv.DictReader(open(args.subset))}
    act = actives_by_target(args.eval_dir, keep)
    print(f"subset targets with actives: {len(act)}")

    rows = []
    for r in csv.DictReader(open(args.redundancy)):
        a, b = r["uniprot_a"], r["uniprot_b"]
        if a not in act or b not in act:
            continue
        ident = float(r["identity"])
        if ident < args.min_identity:
            continue
        shared = set(act[a]) & set(act[b])
        if len(shared) < args.min_shared:
            continue
        d = [abs(act[a][s] - act[b][s]) for s in shared]
        rows.append((ident, a, b, len(shared), sum(d) / len(d), max(d)))
    rows.sort(reverse=True)

    sel = [x for x in rows if x[4] >= args.selective_delta]
    print(f"pairs with identity >= {args.min_identity:.0%} and >= {args.min_shared} "
          f"shared ligands: {len(rows)}")
    print(f"of those, mean |dpAff| >= {args.selective_delta}: {len(sel)}")
    print("\n%-8s %-9s %-9s %7s %9s %9s"
          % ("ident", "A", "B", "shared", "mean|dp|", "max|dp|"))
    for ident, a, b, n, mean, mx in rows[:15]:
        print("%-8.3f %-9s %-9s %7d %9.2f %9.2f" % (ident, a, b, n, mean, mx))

    if args.expect_pairs is not None:
        assert len(rows) == args.expect_pairs, \
            f"pairs {len(rows)} != published {args.expect_pairs}"
    if args.expect_selective is not None:
        assert len(sel) == args.expect_selective, \
            f"selective {len(sel)} != published {args.expect_selective}"
    if args.expect_pairs or args.expect_selective:
        print("\nreproduced the published counts")


if __name__ == "__main__":
    main()
