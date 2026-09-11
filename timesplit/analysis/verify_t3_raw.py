"""Check whether the raw score files under t3_raw are complete and usable, and remove the bad ones.

Why this is needed: duplicate-started processes have been killed mid-run
before, and a .npy file may have been left half-written. A bad file
surfacing only when metrics are computed later would be too late to catch —
this scans ahead of time instead.

Criteria: must load successfully, lengths must match, labels must have both
positives and negatives, and scores must have no NaN/Inf. A target whose
directory fails any check is removed entirely — better to lose one target
than let dirty data into the results table.
"""
import argparse, os, shutil
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="/data/work/vs-benchmark/results/t3_raw")
a = ap.parse_args()

if not os.path.isdir(a.raw):
    print(f"{a.raw} 不存在"); raise SystemExit(0)

for m in sorted(os.listdir(a.raw)):
    root = f"{a.raw}/{m}/T3"
    if not os.path.isdir(root):
        continue
    ok = bad = 0
    reasons = {}
    for layer in sorted(os.listdir(root)):
        for t in sorted(os.listdir(f"{root}/{layer}")):
            d = f"{root}/{layer}/{t}"
            why = None
            try:
                s = np.load(f"{d}/saved_preds.npy")
                l = np.load(f"{d}/saved_labels.npy")
                if len(s) != len(l):
                    why = "长度不一致"
                elif l.sum() == 0 or l.sum() == len(l):
                    why = "标签全同"
                elif not np.isfinite(s).all():
                    why = "分数含 NaN/Inf"
            except Exception as e:
                why = type(e).__name__
            if why:
                shutil.rmtree(d, ignore_errors=True)
                reasons[why] = reasons.get(why, 0) + 1
                bad += 1
            else:
                ok += 1
    print(f"  {m}: 可用 {ok}  剔除 {bad}" + (f"  {reasons}" if reasons else ""))
