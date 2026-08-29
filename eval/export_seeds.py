"""HypSeek _vs 两个种子 vs 官方 _rk，T1 与 T3 并排。

种子结论只能靠 DUD-E / DEKOIS：LIT-PCBA 只有 15 个靶点，两个种子在那档
差 17.6%，本身就说明该基准的噪声量级。
"""
import csv, json, os
B = "/data/yicheng/xqc/vs-benchmark"
out = f"{B}/results/export/T1_T3_hypseek_seeds.csv"
os.makedirs(os.path.dirname(out), exist_ok=True)

T1 = {  # 手工录入自 score_ligunity.py 的输出（作者基准列不在本表内）
    "hypseek_rk":    {"DUDE": 56.39, "DEKOIS": 28.83, "PCBA": 8.34},
    "hypseek_vs_s1": {"DUDE": 43.29, "DEKOIS": 23.32, "PCBA": 4.44},
    "hypseek_vs_s2": {"DUDE": 42.03, "DEKOIS": 22.63, "PCBA": 5.22},
}
d = json.load(open(f"{B}/results/t3/summary.json"))

rows = [["task", "benchmark_or_layer", "metric",
         "hypseek_rk", "hypseek_vs_seed1", "hypseek_vs_seed2",
         "seed_spread_pct", "vs_rk_pct"]]

def pct(a, b):
    return 100 * (b - a) / a if a else float("nan")

for b in ("DUDE", "DEKOIS", "PCBA"):
    rk, s1, s2 = (T1[m][b] for m in ("hypseek_rk", "hypseek_vs_s1", "hypseek_vs_s2"))
    rows.append(["T1", b, "EF1%", f"{rk:.2f}", f"{s1:.2f}", f"{s2:.2f}",
                 f"{abs(pct(s1, s2)):.1f}", f"{pct(rk, (s1 + s2) / 2):+.1f}"])

for L in ("L1", "L2", "L3", "L4"):
    for key, name in (("auroc", "AUROC"), ("ef1", "EF1%"), ("bedroc", "BEDROC")):
        try:
            rk = d["hypseek_rk"][L][key]
            s1 = d["hypseek_vs_s1"][L][key]
            s2 = d["hypseek_vs_s2"][L][key]
        except KeyError:
            continue
        rows.append(["T3", L, name, f"{rk:.4f}", f"{s1:.4f}", f"{s2:.4f}",
                     f"{abs(pct(s1, s2)):.1f}", f"{pct(rk, (s1 + s2) / 2):+.1f}"])

with open(out, "w", newline="") as f:
    csv.writer(f).writerows(rows)
print("写入", out)
print(open(out).read())
