"""Export the complete T3_main.csv (all models) from results/t3/summary.json.

History: an export back in August overwrote T3_main.csv with only the sprint
model; the complete data had been sitting in T3_main_clean.csv all along.
This script lets the main table be rebuilt at any time.
"""
import csv, json, os
B = "/data/work/vs"
d = json.load(open(f"{B}/results/t3/summary.json"))
out = f"{B}/results/export/T3_main.csv"
os.makedirs(os.path.dirname(out), exist_ok=True)
keys = [("AUROC","auroc"),("BEDROC","bedroc"),("EF0.1%","ef01"),("EF1%","ef1"),("EF5%","ef5")]
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "layer", "n_targets"] + [h for h, _ in keys])
    for m in sorted(d):
        for L in ("L1", "L2", "L3", "L4"):
            r = d[m].get(L)
            if not r:
                continue
            row = [m, L, r.get("n_targets") or r.get("n")]
            for _, k in keys:
                v = r.get(k)
                row.append(f"{v:.4f}" if isinstance(v, float) else v)
            w.writerow(row)
print("写入", out)
