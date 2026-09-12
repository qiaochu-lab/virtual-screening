"""Exact permutation test behind finding 18's between-group difference.

Why this file exists
--------------------
The headline statistic of finding 18 — "group B ranks 1.73 places better on
targets only the affinity half covers, exact permutation p = 0.0095 (2/210)" —
was computed ad hoc and only ever written into prose. No script produced it and
no CSV held it, so it could not be re-derived when the inputs changed. This
closes that gap.

The test
--------
Each model has one number: its mean within-target rank on "L-only" targets
minus the same on "P-only" targets (the `Lonly_minus_Ponly` column of
train_set_crossover.py's output). Negative = relatively stronger on the targets
only the affinity half covers.

Statistic: mean(group B) - mean(the rest). Group B is the four models trained
on the affinity half. Under the null, group membership carries no information,
so every way of choosing 4 of the 10 models is equally likely: C(10,4) = 210
splits, enumerated exhaustively rather than sampled. p is the fraction of
splits whose difference is at least as extreme (as negative) as observed.

Validation: run with --expect to assert the published values are reproduced
from the committed CSVs before trusting it on new ones.
"""
import argparse
import csv
import itertools

B = "/data/work/vs-benchmark"
GROUP_B = {"ligunity_pocket_ranking", "ligunity_protein_ranking",
           "litenclip", "hypseek_official_vs", "hypseek_rk"}


def run(path, col="Lonly_minus_Ponly"):
    d = {r["model"]: float(r[col]) for r in csv.DictReader(open(path))}
    ms = sorted(d)
    grp = [m for m in ms if m in GROUP_B]
    if len(grp) != 4:
        raise SystemExit(
            f"{path}: expected 4 group-B models among {len(ms)}, found {len(grp)}: {grp}. "
            "Ranking must cover ten models — check that exactly one HypSeek weight is present.")
    rest = [m for m in ms if m not in GROUP_B]

    def diff(sel):
        others = [m for m in ms if m not in sel]
        return sum(d[m] for m in sel) / len(sel) - sum(d[m] for m in others) / len(others)

    obs = diff(grp)
    splits = list(itertools.combinations(ms, len(grp)))
    hits = sum(1 for s in splits if diff(s) <= obs + 1e-12)
    return {"n_models": len(ms), "group": grp, "rest": rest,
            "mean_B": sum(d[m] for m in grp) / len(grp),
            "mean_rest": sum(d[m] for m in rest) / len(rest),
            "diff": obs, "hits": hits, "n_splits": len(splits),
            "p": hits / len(splits), "per_model": d}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=f"{B}/results/export/T3_train_set_crossover.csv")
    ap.add_argument("--expect-diff", type=float,
                    help="assert |diff| rounds to this (validation against published values)")
    ap.add_argument("--expect-p", type=float, help="assert p equals this")
    a = ap.parse_args()

    r = run(a.csv)
    print(f"{a.csv}")
    print(f"  models ranked   : {r['n_models']}")
    print(f"  group B         : {', '.join(r['group'])}")
    print(f"  mean rank diff  : B {r['mean_B']:+.3f}   rest {r['mean_rest']:+.3f}")
    print(f"  between-group   : {r['diff']:+.3f}  ({abs(r['diff']):.2f} places)")
    print(f"  exact permutation p = {r['hits']}/{r['n_splits']} = {r['p']:.4f}")

    if a.expect_diff is not None:
        # Compare numerically, not on the rounded string. The full-set value is
        # exactly -1.845 — dead on the two-decimal boundary, which the docs
        # render half-up as 1.85 while Python's round() gives 1.84. The
        # tolerance has to include that boundary, hence <= half a display digit.
        assert abs(abs(r["diff"]) - a.expect_diff) <= 0.0051, \
            f"diff {abs(r['diff']):.4f} not consistent with published {a.expect_diff}"
        print(f"  ✓ reproduces published {a.expect_diff} places")
    if a.expect_p is not None:
        assert abs(r["p"] - a.expect_p) < 5e-5, f"p {r['p']:.4f} != expected {a.expect_p}"
        print(f"  ✓ reproduces published p = {a.expect_p}")


if __name__ == "__main__":
    main()
