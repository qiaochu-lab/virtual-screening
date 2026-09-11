"""Export T2_on_T3.csv from summary_t2_v2.json (the corrected convention, aligned by molecule identity).

History: the 08-26 export used the old score_t2.py, which read the lmdb by
numeric index rather than cursor order, so molecules and scores were
misaligned throughout. The old column is kept alongside so the retraction
can be checked against it.
"""
import csv, json, os
B = "/data/work/vs"
d = json.load(open(f"{B}/results/t3/summary_t2_v2.json"))
out = f"{B}/results/export/T2_on_T3.csv"
os.makedirs(os.path.dirname(out), exist_ok=True)
cols = ["model","layer","n_targets","spearman","spearman_sem","kendall",
        "frac_positive","median_n_actives","spearman_old_retracted"]
with open(out,"w",newline="") as f:
    w = csv.writer(f); w.writerow(cols)
    for m in sorted(d):
        for L in ("L1","L2","L3","L4"):
            r = d[m].get(L)
            if not r: continue
            w.writerow([m, L, r["n_targets"], f"{r['spearman']:+.4f}",
                        f"{r['spearman_sem']:.4f}", f"{r['kendall']:+.4f}",
                        f"{r['frac_positive']:.3f}", r["median_n_actives"],
                        f"{r.get('spearman_old', float('nan')):+.4f}"])
print("写入", out)
