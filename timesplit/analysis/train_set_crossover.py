"""Training-set membership crossover control: split targets by **two label
files** and see which model is relatively stronger in which cell.

Why the design had to be redone
------------
The first version split targets into A∩B / A only / B only / neither, on
the premise that A (DrugCLIP's train_no_test_af) and B (the LigUnity
family) were two mutually exclusive training sets. **That premise is
wrong.** `train_task.py:523-524` shows the LigUnity family reads two label
files at once, and one of them, `train_label_pdbbind_seq.json`, covers
16,744 PDB IDs that are **exactly the same** as `train_no_test_af`
(intersection 16,744, nothing exclusive to either side). So **A is a true
subset of B**, and "B only minus A only" cannot be read as "the value of
PocketAffDB membership".

Switched to splitting by the two label files themselves, so each cell's
meaning is unambiguous:

    P  train_label_pdbbind_seq.json    structure half, pocket only, **both A and B trained on it**
    L  train_label_blend_seq_full.json affinity half, with pAff, **only B trained on it**

Four cells: P∩L / P only / L only / neither.

The comparison that actually matters is **L only vs P only**:
  * both cells were trained on by group B — the only difference is **label
    type** (affinity vs structure-only)
  * group A **never trained on** the L-only cell at all
So if group B is relatively stronger on "L only" while group A is
relatively weaker, that shows what's doing the work is **the affinity-label
half**, not the broad notion of "training-set membership".

Rank is each model's relative position within a target among the ten
models, so target difficulty is automatically cancelled out.
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
    """The four training sets. Set B is the union of **two** label files — see the
    comment on per_model_layers.load_B.

    train_task.py:523-524 reads both the pdbbind half (3,468 UniProt / 16,744 PDB)
    and the blend half (2,196 UniProt). The 16,744 PDB entries covered by the
    pdbbind half are exactly the same as DrugCLIP's train_no_test_af, so
    **A ⊂ B**.
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
    """Returns the UniProt sets for the structure half P and the affinity half L, respectively."""
    out = []
    for f in ("train_label/train_label_pdbbind_seq.json",
              "train_label_blend_seq_full.json"):
        p = f"{B}/data/raw/figshare/{f}"
        out.append({a["uniprot"] for a in json.load(open(p)) if a.get("uniprot")}
                   if os.path.exists(p) else set())
    return out[0], out[1]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=f"{B}/results/export/T3_vsds_matched.csv",
                    help="限定靶点子集；传 all 则用全部十模型共有的靶点。"
                         "全量版是稳健性检查：「仅 P」格从 7 个涨到 28 个，"
                         "小样本假象会在那里现形")
    args = ap.parse_args()
    # ⚠️ The key must be (layer, target), not target alone: the 350 subset is
    # 328 entries / 293 unique uniprots, and 35 uniprots appear in more than
    # one layer. Filtering by uniprot alone would pull in "this target's
    # records in other layers" too — empirically, seen+unseen then reports
    # 417, 89 more than the 328 entries in the subset itself.
    keep = None if args.subset == "all" else {
        (r["layer"], r["uniprot"]) for r in csv.DictReader(open(args.subset))}
    A, Bs = load_sets()
    S = json.load(open(f"{B}/results/t3/summary.json"))

    EF = {}
    for m in S:
        for L in LAYERS:
            for r in S[m][L]["per_target"]:
                if keep is None or (L, r["uniprot"]) in keep:
                    EF.setdefault((L, r["uniprot"]), {})[m] = r["ef1"]
    full = sorted(t for t in EF if len(EF[t]) == len(S))

    RANK = {}
    for t in full:
        ms = sorted(EF[t])
        RANK[t] = dict(zip(ms, rankdata([-EF[t][m] for m in ms], method="average"),
                          strict=True))

    P, L = load_halves()
    print(f"结构半 P（pdbbind，A+B 都训过）: {len(P):,} UniProt")
    print(f"亲和力半 L（blend，只有 B 训过）: {len(L):,} UniProt")
    CELL = {}
    for t in full:
        a, b = t[1] in P, t[1] in L
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
        # whether L-only's rank is significantly better than P-only's (smaller rank = stronger, hence the "less" alternative)
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

    out = (f"{B}/results/export/T3_train_set_crossover.csv" if keep is not None
           else f"{B}/results/export/T3_train_set_crossover_full.csv")
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\n写入 {out}")


if __name__ == "__main__":
    main()
