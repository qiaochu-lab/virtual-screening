"""Check a freshly-produced set of T1 score files against the published table.

Used after `patch_t1_save_preds.py` + a re-run, to establish that the re-run
reproduces T1_main.csv rather than quietly redefining it. Nothing is copied
into the canonical results directories until this passes.

Reads `saved_preds.npy` / `saved_labels.npy` per target, recomputes EF@1%,
EF@5%, BEDROC and AUROC through the project's own metric layer, averages over
targets, and diffs against the committed row. A re-run of the same weights on
the same data through the same forward path should agree to the last printed
digit; the tolerance exists only to keep a 1e-9 float wobble from failing the
run, not to paper over a real difference.
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from eval.metrics import enrichment_factor, roc_auc, bedroc  # noqa: E402

B = "/data/work/vs-benchmark"
BENCHMARKS = ("DUDE", "DEKOIS", "PCBA")


def metrics(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    return dict(
        EF1=enrichment_factor(scores, labels, 0.01),
        EF5=enrichment_factor(scores, labels, 0.05),
        BEDROC=bedroc(scores, labels, alpha=80.5),
        AUROC=roc_auc(scores, labels),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True,
                    help="目录，下面是 <benchmark>/<target>/saved_preds.npy")
    ap.add_argument("--model", required=True, help="T1_main.csv 里的模型名")
    ap.add_argument("--table", default=f"{B}/results/export/T1_main.csv")
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    published = {}
    with open(args.table) as f:
        for r in csv.DictReader(f):
            published[(r["model"], r["benchmark"])] = r

    ok_all = True
    for bm in BENCHMARKS:
        root = f"{args.dir}/{bm}"
        if not os.path.isdir(root):
            print(f"  {bm:7} 目录不存在，跳过")
            continue
        per = []
        missing = 0
        for t in sorted(os.listdir(root)):
            d = f"{root}/{t}"
            if not os.path.isfile(f"{d}/saved_preds.npy"):
                missing += 1
                continue
            s = np.load(f"{d}/saved_preds.npy")
            y = np.load(f"{d}/saved_labels.npy")
            if len(s) != len(y):
                print(f"  ⚠️ {bm}/{t}: 长度 {len(s)} vs {len(y)}")
                ok_all = False
                continue
            per.append(metrics(s, y))
        if not per:
            print(f"  {bm:7} 没有可用的 preds")
            ok_all = False
            continue

        got = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
        ref = published.get((args.model, bm))
        if ref is None:
            print(f"  {bm:7} 表里没有 {args.model} 这一行")
            ok_all = False
            continue
        diffs = {k: abs(got[k] - float(ref[k])) for k in got}
        n_ok = int(ref["n_targets"]) == len(per)
        passed = n_ok and all(v <= args.tol for v in diffs.values())
        ok_all &= passed
        print(f"  {'✅' if passed else '❌'} {bm:7} n={len(per):3}/{ref['n_targets']:3}"
              + (f" 缺{missing}" if missing else "") + "  "
              + "  ".join(f"{k} {got[k]:9.4f} vs {float(ref[k]):9.4f} (Δ{diffs[k]:.5f})"
                          for k in ("EF1", "EF5", "BEDROC", "AUROC")))

    print()
    print("✅ 与已发表表格一致" if ok_all else "❌ 不一致 —— 不要拷进正式目录")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
