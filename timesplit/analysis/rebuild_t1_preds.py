"""Reconstruct the missing T1 per-target score files from saved embeddings.

The gap
-------
Seven of the ten models in T1_main.csv ship `saved_preds.npy` per target, so
anyone can recompute their DUD-E / DEKOIS / LIT-PCBA numbers from the release.
Three do not: LigUnity-pocket, LigUnity-protein and LiTENCLIP. Their upstream
code stores only the two embedding matrices and the labels -- see the note in
`timesplit/runners/patch_ligunity_t3.py`: *"The official code only stores the
embedding, not the score"*. Our T3 patch added the score save on the T3 path;
the T1 path (test_dude_target / test_dekois_target / test_pcba_target) never
got the same treatment.

Why no GPU re-run is needed
---------------------------
The score is a deterministic function of what is already on disk. Upstream
computes, verbatim:

    res = pocket_reps @ mol_reps.T
    res_single = res.max(axis=0)

and both matrices are saved (`saved_target_embed.npy`, `saved_mols_embed.npy`).
So this is a CPU matmul over files we already have, not 2.5 GPU-hours of
re-inference. It also cannot drift from what the model actually produced,
which a re-run could.

Validation is not optional
--------------------------
A reconstruction that is never checked is worth nothing, so this script
recomputes EF@1%, EF@5%, BEDROC and AUROC from the reconstructed scores and
diffs them against the committed T1_main.csv row. It refuses to write anything
unless every metric matches to the tolerance below. The published table was
produced from `res_single` by the same metric layer, so agreement should be
near-exact; the tolerance only absorbs float32/float16 rounding.
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from eval.metrics import enrichment_factor, roc_auc, bedroc  # noqa: E402

B = "/data/work/vs-benchmark"

# model name in T1_main.csv -> directory under results/
DIRS = {
    "litenclip": "litenclip",
    "ligunity_pocket": "pocket_ranking",
    "ligunity_protein": "protein_ranking",
}
BENCHMARKS = ("DUDE", "DEKOIS", "PCBA")


def scores_for(tdir):
    """Reproduce upstream's `(pocket_reps @ mol_reps.T).max(axis=0)`."""
    mol = np.load(f"{tdir}/saved_mols_embed.npy")
    pocket = np.load(f"{tdir}/saved_target_embed.npy")
    labels = np.load(f"{tdir}/saved_labels.npy")
    # float16 embeddings would lose precision in the matmul; upstream does this
    # in float32 on the GPU, so match that rather than the storage dtype.
    res = pocket.astype(np.float32) @ mol.astype(np.float32).T
    return res.max(axis=0), labels


def metrics(scores, labels):
    return dict(
        EF1=enrichment_factor(scores, labels, 0.01),
        EF5=enrichment_factor(scores, labels, 0.05),
        BEDROC=bedroc(scores, labels, alpha=80.5),
        AUROC=roc_auc(scores, labels),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=f"{B}/results")
    ap.add_argument("--table", default=f"{B}/results/export/T1_main.csv",
                    help="committed T1 table to validate against")
    ap.add_argument("--tol", type=float, default=0.01,
                    help="max allowed absolute difference on any metric")
    ap.add_argument("--write", action="store_true",
                    help="write saved_preds.npy; without it, validate only")
    args = ap.parse_args()

    published = {}
    with open(args.table) as f:
        for r in csv.DictReader(f):
            published[(r["model"], r["benchmark"])] = r

    ok_all = True
    pending = []          # (path, scores) to write once everything validates
    for model, sub in DIRS.items():
        for bm in BENCHMARKS:
            root = f"{args.results}/{sub}/{bm}"
            if not os.path.isdir(root):
                print(f"  {model:18} {bm:7} 目录不存在，跳过")
                continue
            targets = sorted(t for t in os.listdir(root)
                             if os.path.isdir(f"{root}/{t}"))
            per = []
            for t in targets:
                d = f"{root}/{t}"
                try:
                    s, y = scores_for(d)
                except FileNotFoundError as e:
                    print(f"  ⚠️ {model}/{bm}/{t}: 缺文件 {e.filename}")
                    ok_all = False
                    continue
                if len(s) != len(y):
                    print(f"  ⚠️ {model}/{bm}/{t}: 长度不一致 {len(s)} vs {len(y)}")
                    ok_all = False
                    continue
                per.append(metrics(s, y))
                pending.append((f"{d}/saved_preds.npy", s))

            if not per:
                continue
            got = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
            ref = published.get((model, bm))
            if ref is None:
                print(f"  {model:18} {bm:7} 表里没有这一行，无法校验")
                ok_all = False
                continue

            n_ok = int(ref["n_targets"]) == len(per)
            diffs = {k: abs(got[k] - float(ref[k])) for k in got}
            passed = n_ok and all(v <= args.tol for v in diffs.values())
            ok_all &= passed
            flag = "✅" if passed else "❌"
            print(f"  {flag} {model:18} {bm:7} n={len(per):3}/{ref['n_targets']:3}  "
                  + "  ".join(f"{k} {got[k]:8.4f} vs {float(ref[k]):8.4f} "
                              f"(Δ{diffs[k]:.4f})" for k in ("EF1", "AUROC")))
            if not passed:
                print("       全部指标差值：" +
                      "  ".join(f"{k}Δ{diffs[k]:.4f}" for k in diffs))

    print()
    if not ok_all:
        print("❌ 校验未全部通过 —— 不写任何文件。")
        return 1
    print(f"✅ 全部通过（容差 {args.tol}）。")
    if not args.write:
        print("   这是校验模式；加 --write 才会写入 saved_preds.npy。")
        return 0
    for path, s in pending:
        np.save(path, s)
    print(f"   已写入 {len(pending)} 个 saved_preds.npy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
