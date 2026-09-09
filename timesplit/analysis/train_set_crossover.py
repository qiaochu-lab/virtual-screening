"""训练集归属的交叉对照：靶点按**两个标签文件**切格，看谁在哪一格相对更强。

设计为什么要重做
----------------
第一版把靶点切成 A∩B / A only / B only / neither，前提是 A（DrugCLIP 的
train_no_test_af）和 B（LigUnity 系）是两套互斥的训练集。**这个前提是错的。**
`train_task.py:523-524` 显示 LigUnity 系同时读两个标签文件，其中
`train_label_pdbbind_seq.json` 覆盖 16,744 个 PDB ID，与 `train_no_test_af`
**完全相同**（交集 16,744，各自独有 0）。所以 **A 是 B 的真子集**，
「B only 减 A only」不能读成「PocketAffDB 成员身份的价值」。

改成按两个标签文件本身切，这样每一格的含义是明确的：

    P  train_label_pdbbind_seq.json   结构半，只有口袋，**A 和 B 都训过**
    L  train_label_blend_seq_full.json 亲和力半，带 pAff，**只有 B 训过**

四格：P∩L / 仅 P / 仅 L / 都没有。

真正要读的对比是 **仅 L vs 仅 P**：
  · 两格 B 组都训过，差别只在**标签类型**（亲和力 vs 只有结构）
  · 仅 L 那格 A 组**完全没训过**
所以如果 B 组在「仅 L」相对更强而 A 组相对更弱，说明起作用的是
**亲和力标签这一半**，不是「训练集成员身份」这个笼统的东西。

名次是靶点内十个模型的相对排位，靶点难不难会被自动消掉。
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
    return A, Bs


def load_halves():
    """分别返回结构半 P 和亲和力半 L 的 UniProt 集合。"""
    out = []
    for f in ("train_label/train_label_pdbbind_seq.json",
              "train_label_blend_seq_full.json"):
        p = f"{B}/data/raw/figshare/{f}"
        out.append({a["uniprot"] for a in json.load(open(p)) if a.get("uniprot")}
                   if os.path.exists(p) else set())
    return out[0], out[1]


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

    P, L = load_halves()
    print(f"结构半 P（pdbbind，A+B 都训过）: {len(P):,} UniProt")
    print(f"亲和力半 L（blend，只有 B 训过）: {len(L):,} UniProt")
    CELL = {}
    for t in full:
        a, b = t in P, t in L
        CELL[t] = ("P∩L" if a and b else "仅 P" if a else
                   "仅 L" if b else "都没有")
    cells = ["P∩L", "仅 P", "仅 L", "都没有"]
    print("四格靶点数（十个模型都有结果的 %d 个）：" % len(full))
    for c in cells:
        print("  %-8s %3d" % (c, sum(1 for t in full if CELL[t] == c)))

    print("\n各模型在四格里的平均靶点内名次（1 = 十个模型里最好，越小越强）")
    print("%-24s %-3s %8s %8s %8s %8s %14s %10s"
          % ("模型", "组", "P∩L", "仅 P", "仅 L", "都没有",
             "仅L−仅P", "p"))
    print("-" * 96)
    rows = [["model", "group", "rank_PL", "rank_P_only", "rank_L_only",
             "rank_neither", "Lonly_minus_Ponly", "p_mwu"]]
    for m in S:
        g = GROUP.get(m, "?")
        v = {c: [RANK[t][m] for t in full if CELL[t] == c] for c in cells}
        if min(len(v[c]) for c in ("仅 P", "仅 L")) < 5:
            print(f"{m}: 仅 P / 仅 L 样本太少，跳过"); continue
        mv = {c: float(np.mean(v[c])) if v[c] else float("nan") for c in cells}
        # 仅 L 的名次是否显著好于 仅 P（名次小 = 强，所以检验 less）
        p = mannwhitneyu(v["仅 L"], v["仅 P"], alternative="less").pvalue
        d = mv["仅 L"] - mv["仅 P"]
        print("%-24s %-3s %8.2f %8.2f %8.2f %8.2f %+14.2f %10.5f"
              % (m, g, mv["P∩L"], mv["仅 P"], mv["仅 L"], mv["都没有"], d, p))
        rows.append([m, g] + ["%.3f" % mv[c] for c in cells] +
                    ["%.3f" % d, "%.6f" % p])
    print("-" * 96)
    print("\n「仅L−仅P」为负 = 在只有亲和力标签的那批靶点上相对更强。")
    print("两格 B 组都训过，差别只在标签类型；仅 L 那格 A 组完全没训过。")
    print("所以 B 组为负、A 组为正，说明起作用的是**亲和力标签这一半**；")
    print("若所有模型同号，那就是这两格靶点本身难度不同，与训练无关。")

    out = f"{B}/results/export/T3_train_set_crossover.csv"
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\n写入 {out}")


if __name__ == "__main__":
    main()
