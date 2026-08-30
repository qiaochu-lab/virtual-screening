"""从 results/t3/summary.json 导出完整的 T3_main.csv（全部模型）。

历史：08 月某次导出把 T3_main.csv 覆盖成只含 sprint 一个模型；完整数据一直
在 T3_main_clean.csv 里。这个脚本让主表可以随时重建。
"""
import csv, json, os
B = "/data/work/vs"
d = json.load(open(f"{B}/results/t3/summary.json"))
out = f"{B}/results/export/T3_main.csv"
os.makedirs(os.path.dirname(out), exist_ok=True)
keys = [("AUROC","auroc"),("BEDROC","bedroc"),("EF0.1%","ef01"),("EF1%","ef1"),("EF5%","ef5")]
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "layer", "n_targets"] + keys)
    for m in sorted(d):
        for L in ("L1", "L2", "L3", "L4"):
            r = d[m].get(L)
            if not r:
                continue
            row = [m, L, r.get("n_targets") or r.get("n")]
            for k in keys:
                v = r.get(k, r.get(k.replace("%", "")))
                row.append(f"{v:.4f}" if isinstance(v, float) else v)
            w.writerow(row)
print("写入", out)
