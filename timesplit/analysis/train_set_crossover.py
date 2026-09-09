"""训练集归属的交叉对照：把靶点切成四格，看每个模型在哪一格相对更强。

上一步出现了两个口径打架：
  · 绝对 EF：A 组（DrugCLIP/BindCLIP）在自己见过的靶点上明显更高（P=0.67~0.69）
  · 靶点内名次：A 组的名次几乎没变（−0.10~−0.13），B 组却好了 1.7 名
名次是靶点内十个模型的相对排位，靶点难不难会被自动消掉。两个口径分歧，
说明 A 组的「见过更高」是**靶点本身好做**，B 组的才是**训练集专属优势**。

但还有一个说不清的：B（PocketAffDB）是精选的带亲和力标签的小集合，
它的模型也许只是「在被研究透的靶点上普遍更强」，与训不训练过无关。
四格交叉能分开：

    A∩B      两套都有
    A only   只有 train_no_test_af 有
    B only   只有 PocketAffDB 有
    neither  两套都没有

如果 B 组模型的优势真来自「训练过」，它在 **B only** 格应当明显好于 **A only** 格；
如果只是「好靶点效应」，两格应该差不多（因为两格的靶点都被某个大集合收录过）。
A 组模型的方向应当相反。这是一个交叉设计，两组互为对照。
"""
import csv
import json
import os

import numpy as np
from scipy.stats import mannwhitneyu, rankdata

B = "/data/work/vs-benchmark"
GROUP = {
    "drugclip": "A", "bindclip_randneg": "A", "bindclip_hardneg": "A",
    "ligunity_pocket_ranking": "B", "ligunity_protein_ranking": "B",
    "litenclip": "B", "hypseek_rk": "B",
    "conplex": "C", "sprint": "D", "conglude": "?",
}
LAYERS = ["L1", "L2", "L3", "L4"]


def load_sets():
    import pickle

    import lmdb
    lab = json.load(open(f"{B}/data/raw/figshare/train_label_blend_seq_full.json"))
    Bs = {a["uniprot"] for a in lab if a.get("uniprot")}
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
    return A, Bs


def main():
    keep = {r["uniprot"] for r in csv.DictReader(
        open(f"{B}/results/export/T3_vsds_matched.csv"))}
    A, Bs = load_sets()
    S = json.load(open(f"{B}/results/t3/summary.json"))

    EF = {}
    for m in S:
        for L in LAYERS:
            for r in S[m][L]["per_target"]:
                if r["uniprot"] in keep:
                    EF.setdefault(r["uniprot"], {})[m] = r["ef1"]
    full = sorted(t for t in EF if len(EF[t]) == len(S))

    RANK = {}
    for t in full:
        ms = sorted(EF[t])
        RANK[t] = dict(zip(ms, rankdata([-EF[t][m] for m in ms], method="average")))

    CELL = {}
    for t in full:
        a, b = t in A, t in Bs
        CELL[t] = ("A∩B" if a and b else "A only" if a else
                   "B only" if b else "neither")
    cells = ["A∩B", "A only", "B only", "neither"]
    print("四格靶点数（十个模型都有结果的 %d 个）：" % len(full))
    for c in cells:
        print("  %-8s %3d" % (c, sum(1 for t in full if CELL[t] == c)))

    print("\n各模型在四格里的平均靶点内名次（1 = 十个模型里最好，越小越强）")
    print("%-24s %-3s %8s %8s %8s %8s %14s %10s"
          % ("模型", "组", "A∩B", "A only", "B only", "neither",
             "B only−A only", "p"))
    print("-" * 96)
    rows = [["model", "group", "rank_AB", "rank_Aonly", "rank_Bonly",
             "rank_neither", "Bonly_minus_Aonly", "p_mwu"]]
    for m in S:
        g = GROUP.get(m, "?")
        v = {c: [RANK[t][m] for t in full if CELL[t] == c] for c in cells}
        if min(len(v[c]) for c in ("A only", "B only")) < 5:
            print(f"{m}: A only / B only 样本太少，跳过"); continue
        mv = {c: float(np.mean(v[c])) if v[c] else float("nan") for c in cells}
        # B only 的名次是否显著好于 A only（名次小 = 强，所以检验 less）
        p = mannwhitneyu(v["B only"], v["A only"], alternative="less").pvalue
        d = mv["B only"] - mv["A only"]
        print("%-24s %-3s %8.2f %8.2f %8.2f %8.2f %+14.2f %10.5f"
              % (m, g, mv["A∩B"], mv["A only"], mv["B only"], mv["neither"], d, p))
        rows.append([m, g] + ["%.3f" % mv[c] for c in cells] +
                    ["%.3f" % d, "%.6f" % p])
    print("-" * 96)
    print("\n「B only−A only」为负 = 在只被 PocketAffDB 收录的靶点上相对更强。")
    print("B 组模型该为负、A 组模型该为正或接近 0，才说明是训练集专属效应；")
    print("若所有模型同号，那就是这两格靶点本身难度不同，与训练集无关。")

    out = f"{B}/results/export/T3_train_set_crossover.csv"
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\n写入 {out}")


if __name__ == "__main__":
    main()
