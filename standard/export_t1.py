"""T1 汇总：九个模型 × 三个标准基准，一张表。

各模型的原始输出散在两处布局：
  · UniMol 系与 LigUnity 系  results/<模型>/<基准>/<靶点>/
  · 后接的三个模型          results/t1_raw/<模型>/<基准>/<靶点>/
指标一律由统一评测层从逐分子打分重算，不抄任何论文数字。
靶点数各模型可能不同（序列超长、结构缺失、SMILES 解析失败），一并记下来，
不然 EF 的分母不一样会被误读成模型差异。
"""
import os, sys, json
import numpy as np
B = "/data/work/vs-benchmark"
sys.path.insert(0, f"{B}/eval")
from metrics import bedroc, enrichment_factor, roc_auc

MODELS = [
    ("hypseek_rk",       f"{B}/results/hypseek_rk"),
    ("ligunity_pocket",  f"{B}/results/pocket_ranking"),
    ("ligunity_protein", f"{B}/results/protein_ranking"),
    ("litenclip",        f"{B}/results/litenclip"),
    ("drugclip",         f"{B}/results/drugclip"),
    ("bindclip_randneg", f"{B}/results/bindclip_randneg"),
    ("bindclip_hardneg", f"{B}/results/bindclip_hardneg"),
    ("conglude",         f"{B}/results/t1_raw/conglude"),
    ("conplex",          f"{B}/results/t1_raw/conplex"),
    ("sprint",           f"{B}/results/t1_raw/sprint"),
]
BENCH = ["DUDE", "DEKOIS", "PCBA"]

def score_target(d):
    p = f"{d}/saved_preds.npy"
    if os.path.exists(p):
        s = np.load(p).reshape(-1)
    else:                      # 只落了 embedding 的，按官方规则复原：口袋×分子取 max
        mp, pp = f"{d}/saved_mols_embed.npy", f"{d}/saved_target_embed.npy"
        if not os.path.exists(pp):
            pp = f"{d}/saved_pocket_embed.npy"
        if not (os.path.exists(mp) and os.path.exists(pp)):
            return None
        s = (np.load(pp) @ np.load(mp).T).max(axis=0)
    y = np.load(f"{d}/saved_labels.npy")
    if len(s) != len(y) or y.sum() == 0 or y.sum() == len(y):
        return None
    return dict(ef1=enrichment_factor(s, y, 0.01), ef5=enrichment_factor(s, y, 0.05),
                bedroc=bedroc(s, y, 80.5), auroc=roc_auc(s, y))

rows = ["model,benchmark,n_targets,EF1,EF5,BEDROC,AUROC"]
print("%-18s %-8s %7s %8s %8s %8s %8s" % ("模型","基准","靶点","EF1%","EF5%","BEDROC","AUROC"))
print("-"*70)
for name, root in MODELS:
    for b in BENCH:
        d = f"{root}/{b}"
        if not os.path.isdir(d):
            continue
        vals = [score_target(f"{d}/{t}") for t in sorted(os.listdir(d))]
        vals = [v for v in vals if v]
        if not vals:
            continue
        m = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
        print("%-18s %-8s %7d %8.2f %8.2f %8.4f %8.4f" %
              (name, b, len(vals), m["ef1"], m["ef5"], m["bedroc"], m["auroc"]))
        rows.append(f"{name},{b},{len(vals)},{m['ef1']:.4f},{m['ef5']:.4f},{m['bedroc']:.4f},{m['auroc']:.4f}")
os.makedirs(f"{B}/results/export", exist_ok=True)
open(f"{B}/results/export/T1_main.csv","w").write("\n".join(rows)+"\n")
print(f"\n写入 {B}/results/export/T1_main.csv")
