"""把各任务的结果导出成 CSV，供交接与外部分析。

只导出聚合层（每模型每层一行），不导原始打分——后者几十 GB 且含数据集内容。
"""
import csv, json, os
import numpy as np
from scipy import stats

B = "/data/work/vs-benchmark"
OUT = f"{B}/results/export"
os.makedirs(OUT, exist_ok=True)

# ---------- T3 主表 ----------
s = json.load(open(f"{B}/results/t3/summary.json"))
with open(f"{OUT}/T3_main.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "layer", "n_targets", "AUROC", "BEDROC", "EF0.1%", "EF1%", "EF5%"])
    for m in sorted(s):
        for L in ["L1", "L2", "L3", "L4"]:
            if L not in s[m]: continue
            d = s[m][L]
            w.writerow([m, L, len(d["per_target"]),
                        f"{d['auroc']:.4f}", f"{d['bedroc']:.4f}",
                        f"{d.get('ef01', float('nan')):.3f}",
                        f"{d['ef1']:.3f}", f"{d['ef5']:.3f}"])
print("  T3_main.csv")

# ---------- T5 阈值 ----------
rows = []
for tag, fn in [("4A", "summary_4a.json"), ("6A", "summary.json"), ("8A", "summary_8a.json")]:
    p = f"{B}/results/t3/{fn}"
    if not os.path.exists(p): continue
    d = json.load(open(p))
    for m in d:
        base = m.replace("_4a", "").replace("_8a", "")
        for L in ["L1", "L2", "L3", "L4"]:
            if L in d[m]:
                rows.append([base, tag, L, f"{d[m][L]['ef1']:.3f}", f"{d[m][L]['auroc']:.4f}"])
with open(f"{OUT}/T5_pocket_threshold.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model", "threshold", "layer", "EF1%", "AUROC"]); w.writerows(rows)
print("  T5_pocket_threshold.csv")

# ---------- T2 (T3 数据) ----------
t2 = json.load(open(f"{B}/results/t3/summary_t2.json"))
with open(f"{OUT}/T2_on_T3.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "layer", "n_targets", "spearman", "spearman_sem",
                "kendall", "frac_positive", "median_n_actives"])
    for m in sorted(t2):
        for L in ["L1", "L2", "L3", "L4"]:
            if L not in t2[m]: continue
            d = t2[m][L]
            w.writerow([m, L, d["n_targets"], f"{d['spearman']:+.4f}",
                        f"{d['spearman_sem']:.4f}", f"{d['kendall']:+.4f}",
                        f"{d['frac_positive']:.3f}", d["median_n_actives"]])
print("  T2_on_T3.csv")

# ---------- T2 (FEP 数据) + 物理参考 ----------
ref = json.load(open(f"{B}/data/t3/fep_reference.json"))
phys = {**ref["jacs"], **ref["merck"]}
JACS = set(ref["jacs"])
rows = []
for m in sorted(os.listdir(f"{B}/results/fep")):
    root = f"{B}/results/fep/{m}/FEP"
    if not os.path.isdir(root): continue
    for t in sorted(os.listdir(root)):
        d = f"{root}/{t}"
        try:
            p = np.load(f"{d}/saved_preds.npy"); y = np.load(f"{d}/saved_labels.npy")
        except Exception: continue
        if len(p) != len(y) or np.std(p) == 0: continue
        rows.append([m, t, "JACS" if t in JACS else "Merck", len(y),
                     f"{stats.spearmanr(p, y).statistic:+.4f}",
                     f"{stats.kendalltau(p, y).statistic:+.4f}",
                     f"{stats.pearsonr(p, y).statistic:+.4f}"])
for t, v in sorted(phys.items()):
    rows.append(["_reference_UniFEP", t, "JACS" if t in JACS else "Merck",
                 v["n"], "", f"{v['kendall']:+.4f}", ""])
with open(f"{OUT}/T2_on_FEP.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "system", "set", "n_ligands", "spearman", "kendall", "pearson"])
    w.writerows(rows)
print("  T2_on_FEP.csv")

# ---------- 靶点元信息 ----------
cls = json.load(open(f"{B}/data/t3/target_class.json"))["class"]
qual = json.load(open(f"{B}/data/t3/target_quality.json"))
grade, conf = qual["grade"], qual.get("confidence", {})
layers = {}
for L in ["L1", "L2", "L3", "L4"]:
    p = f"{B}/data/t3/eval/{L}.jsonl"
    if not os.path.exists(p): continue
    for line in open(p):
        d = json.loads(line)
        layers.setdefault(d["uniprot"], []).append(
            (L, d["n_actives"], d["n_decoys"]))
with open(f"{OUT}/T3_targets.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["uniprot", "layer", "n_actives", "n_decoys",
                "protein_class", "structure_grade", "boltz_plddt", "boltz_iptm"])
    for up, ls in sorted(layers.items()):
        for L, na, nd in ls:
            c = conf.get(up, {})
            w.writerow([up, L, na, nd, cls.get(up, ""), grade.get(up, ""),
                        f"{c.get('complex_plddt', ''):.3f}" if c else "",
                        f"{c.get('iptm', ''):.3f}" if c else ""])
print("  T3_targets.csv")
print(f"\n已导出到 {OUT}")
