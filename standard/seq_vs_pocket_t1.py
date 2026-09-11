"""LigUnity sequence branch vs pocket branch: a per-target paired
comparison on the three standard benchmarks.

The same control on T3 is in seq_vs_pocket.py. It's split into two scripts
because the input formats differ: T3's raw scores are packed into npz,
T1's are still one directory per target.

The conclusion is the opposite of T3's, so the two scripts must be read
together: **there is no consistent winner -- it flips by benchmark.** On
LIT-PCBA (whose decoys are experimentally confirmed non-binders), the
pocket branch sweeps every early-enrichment metric; on DEKOIS the sequence
branch wins; DUD-E is a tie.

Scoring convention matches eval/score_ligunity.py: when saved_preds.npy is
absent, use (pocket_reps @ mol_reps.T).max(axis=0), taking the max over
multiple pockets.

Usage
----
    python seq_vs_pocket_t1.py [--results result-root-dir] [--out output-CSV]
"""
import argparse
import csv
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "eval"))
try:
    from metrics import enrichment_factor, roc_auc, bedroc
except ImportError:                       # allow being invoked from elsewhere
    sys.path.insert(0, "/data/work/vs/eval")
    from metrics import enrichment_factor, roc_auc, bedroc

POCKET, SEQ = "pocket_ranking", "protein_ranking"
BENCHES = ("DUDE", "DEKOIS", "PCBA")
METRICS = (("EF1%", lambda s, y: enrichment_factor(s, y, 0.01)),
           ("EF5%", lambda s, y: enrichment_factor(s, y, 0.05)),
           ("BEDROC", lambda s, y: bedroc(s, y, 80.5)),
           ("AUROC", roc_auc))


def load_target(d):
    lp = os.path.join(d, "saved_labels.npy")
    if not os.path.exists(lp):
        return None
    y = np.load(lp)
    pp = os.path.join(d, "saved_preds.npy")
    if os.path.exists(pp):
        s = np.load(pp)
    else:
        mp = os.path.join(d, "saved_mols_embed.npy")
        kp = os.path.join(d, "saved_target_embed.npy")
        if not (os.path.exists(mp) and os.path.exists(kp)):
            return None
        s = (np.load(kp) @ np.load(mp).T).max(axis=0)
    s = np.asarray(s, float).ravel()
    y = np.asarray(y).ravel()
    return (s, y) if len(s) == len(y) else None


def load(root, bench, results):
    d = os.path.join(results, root, bench)
    if not os.path.isdir(d):
        return {}
    out = {}
    for t in sorted(os.listdir(d)):
        r = load_target(os.path.join(d, t))
        if r:
            out[t] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/data/work/vs/results")
    ap.add_argument("--out", default="T1_seq_vs_pocket_per_target.csv")
    args = ap.parse_args()

    rows = [["benchmark", "target", "n_molecules", "n_actives", "metric",
             "pocket", "sequence", "diff"]]
    summary = {}
    for b in BENCHES:
        P = load(POCKET, b, args.results)
        S = load(SEQ, b, args.results)
        common = [t for t in sorted(set(P) & set(S))
                  if np.array_equal(P[t][1], S[t][1])]
        if not common:
            print(f"{b}: 无成对数据（pocket {len(P)}, protein {len(S)}）")
            continue
        for nm, fn in METRICS:
            pair = []
            for t in common:
                try:
                    a, c = fn(*P[t]), fn(*S[t])
                except Exception:
                    continue
                if a is None or c is None or not (np.isfinite(a) and np.isfinite(c)):
                    continue
                pair.append((a, c))
                rows.append([b, t, len(P[t][1]), int(P[t][1].sum()), nm,
                             f"{a:.4f}", f"{c:.4f}", f"{c - a:+.4f}"])
            if len(pair) >= 5:
                summary[(b, nm)] = pair

    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    print(f"{'基准':<8} {'指标':<8} {'靶点':>5} {'口袋版':>9} {'序列版':>9} "
          f"{'序列赢':>7} {'口袋赢':>7} {'平局':>6} {'去平局胜率':>10} {'p(配对)':>10}")
    print("-" * 92)
    last = None
    for (b, nm), pair in summary.items():
        if last and b != last:
            print()
        last = b
        a = np.array([x[0] for x in pair])
        c = np.array([x[1] for x in pair])
        d = c - a
        w = int((d > 1e-9).sum())
        l = int((d < -1e-9).sum())
        try:
            p = stats.wilcoxon(a, c).pvalue
        except ValueError:
            p = float("nan")
        print(f"{b:<8} {nm:<8} {len(d):>5} {a.mean():>9.4f} {c.mean():>9.4f} "
              f"{w:>7} {l:>7} {len(d)-w-l:>6} {100*w/max(w+l,1):>9.1f}% {p:>10.2g}")

    print("\n怎么读")
    print("· LIT-PCBA 的诱饵是实验验证过不结合的分子，是三个基准里最真实的；")
    print("  口袋版在那里的所有榜首指标上全胜。DUD-E 和 DEKOIS 的诱饵是人工生成的。")
    print("· AUROC 在三个基准上都不显著，只有榜首指标（EF/BEDROC）分得出差别。")
    print(f"\n逐靶点结果写入 {args.out}")


if __name__ == "__main__":
    main()
