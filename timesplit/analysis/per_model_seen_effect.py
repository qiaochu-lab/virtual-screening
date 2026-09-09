"""逐模型 seen/unseen 的难度对照——不用除法的版本。

第一版用 r_t = EF(模型,t) / median(EF(其余模型,t)) 归一化难度，但分母接近 0
时比值会爆掉：drugclip 的倍数中位是 1.85、均值却是 0.57，两个方向相反，
说明这个比值的分布被极端值主导，不能报。

换两个不需要除法的统计量：

1. **P(seen > unseen)** —— 随机取一个 seen 靶点和一个 unseen 靶点，
   前者 r 更大的概率。就是 Mann-Whitney U / (n1·n2)，有界在 [0,1]，
   0.5 = 没差别。它和 p 值是同一个检验的两面，但能读出效应大小。

2. **组内名次** —— 每个靶点上把十个模型按 EF1 排名（1 = 最好），
   比较该模型在自己 seen 靶点和 unseen 靶点上的平均名次。
   完全绕开除法，也自动把靶点难度消掉（名次是靶点内部的相对量）。
   名次变好（数值变小）才说明有训练集优势。

两个统计量方向一致，结论才算立住。
"""
import csv
import json
import os

import numpy as np
from scipy.stats import mannwhitneyu, rankdata

B = "/data/work/vs-benchmark"
MODEL_SET = {
    "drugclip": "A", "bindclip_randneg": "A", "bindclip_hardneg": "A",
    "ligunity_pocket_ranking": "B", "ligunity_protein_ranking": "B",
    "litenclip": "B", "hypseek_rk": "B",
    "conplex": "C", "sprint": "D",
}
LAYERS = ["L1", "L2", "L3", "L4"]


def load_sets():
    """四套训练集。B 组是**两个**标签文件的并集，见 per_model_layers.load_B 的注释。

    train_task.py:523-524 同时读 pdbbind 半（3,468 UniProt / 16,744 PDB）和
    blend 半（2,196 UniProt）。pdbbind 半覆盖的 16,744 个 PDB 与 DrugCLIP 的
    train_no_test_af 完全相同，所以 **A ⊂ B**。
    """
    import pickle

    import lmdb
    Bs = set()
    for f in ("train_label_blend_seq_full.json",
              "train_label/train_label_pdbbind_seq.json"):
        p = f"{B}/data/raw/figshare/{f}"
        if os.path.exists(p):
            Bs |= {a["uniprot"] for a in json.load(open(p)) if a.get("uniprot")}
    pdb2up = json.load(open(f"{B}/data/t3/drugclip_pdb2uniprot.json"))
    e = lmdb.open(f"{B}/data/train_no_test_af/train.lmdb",
                  subdir=False, readonly=True, lock=False)
    A = set()
    with e.begin() as t:
        for _, v in t.cursor():
            pk = pickle.loads(v).get("pocket")
            if pk:
                A |= set(pdb2up.get(str(pk).split("_")[0].upper()[:4], []))
    e.close()
    C = {r["uniprot"] for r in csv.DictReader(
        open(f"{B}/results/export/T3_conplex_train_coverage.csv"))
        if float(r["best_identity_conplex_bindingdb"]) >= 0.95
        or r["in_dude_57"] == "1"}
    D = {l.strip() for l in open(f"{B}/data/sprint_train_uniprots.txt") if l.strip()}
    return {"A": A, "B": Bs, "C": C, "D": D}


def main():
    keep = {r["uniprot"] for r in csv.DictReader(
        open(f"{B}/results/export/T3_vsds_matched.csv"))}
    SETS = load_sets()
    S = json.load(open(f"{B}/results/t3/summary.json"))

    EF = {}
    for m in S:
        for L in LAYERS:
            for r in S[m][L]["per_target"]:
                if r["uniprot"] in keep:
                    EF.setdefault(r["uniprot"], {})[m] = r["ef1"]

    # 只用十个模型都有结果的靶点，名次才是可比的
    full = sorted(t for t in EF if len(EF[t]) == len(S))
    print(f"十个模型都有结果的子集靶点：{len(full)} / {len(keep)}")

    RANK = {}
    for t in full:
        ms = sorted(EF[t])
        rk = rankdata([-EF[t][m] for m in ms], method="average")   # 1 = 最好
        RANK[t] = dict(zip(ms, rk))

    rows = [["model", "train_set", "n_seen", "n_unseen", "P_seen_gt_unseen",
             "p_mwu", "rank_seen", "rank_unseen", "rank_gain"]]
    print("\n每个模型在「自己训练集见过的靶点」上是否更强（难度已由靶点内名次消掉）")
    print("%-24s %-3s %6s %7s %14s %9s %9s %9s %8s"
          % ("模型", "集", "seen", "unseen", "P(seen>unseen)", "p",
             "名次 seen", "名次 un", "名次差"))
    print("-" * 100)
    for m, tag in MODEL_SET.items():
        T = SETS[tag]
        es, eu, ks, ku = [], [], [], []
        for t in full:
            (es if t in T else eu).append(EF[t][m])
            (ks if t in T else ku).append(RANK[t][m])
        if len(es) < 5 or len(eu) < 5:
            print(f"{m}: 有一边样本太少（{len(es)}/{len(eu)}），跳过"); continue
        u = mannwhitneyu(es, eu, alternative="greater")
        P = u.statistic / (len(es) * len(eu))
        rs, ru = float(np.mean(ks)), float(np.mean(ku))
        print("%-24s %-3s %6d %7d %14.3f %9.5f %9.2f %9.2f %+8.2f"
              % (m, tag, len(es), len(eu), P, u.pvalue, rs, ru, ru - rs))
        rows.append([m, tag, len(es), len(eu), f"{P:.4f}", f"{u.pvalue:.6f}",
                     f"{rs:.3f}", f"{ru:.3f}", f"{ru - rs:.3f}"])
    print("-" * 100)
    print("\nP > 0.5 且 名次差 > 0 = 该模型在自己见过的靶点上确实更强（两个口径一致）。")
    print("名次是靶点内部十个模型的相对排位，靶点难不难对它没有影响。")
    print("注意这是**保守**估计：其余九个模型里有的也训练过同一批靶点。")

    out = f"{B}/results/export/T3_per_model_seen_effect.csv"
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\n写入 {out}")


if __name__ == "__main__":
    main()
