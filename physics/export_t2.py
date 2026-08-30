"""从 summary_t2_v2.json（按分子身份对齐的修正口径）导出 T2_on_T3.csv。

历史：08-26 的导出用的是旧的 score_t2.py，读 lmdb 时按数值下标而非游标序，
分子与打分整体错位。旧列一并保留，方便对照撤回了什么。
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
