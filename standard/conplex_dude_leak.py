"""Target-leakage check for ConPLex on DUD-E.

Background
----
ConPLex's core method is literally "contrastive learning against DUD-E
decoys" -- ``contrastive: True`` is the shipped default in
configs/default_config.yaml, and training draws its train-half targets from
``dataset/DUDe/dude_*_train_test_split.csv``. The repo ships two splits
(cross / within), 26 targets each, 12 overlapping, union 40; the two CSVs
together name 57 DUD-E targets.

Our T1 uses **all 102** DUD-E targets, so all 40 training targets are in
there. The other nine models' training data contains no DUD-E (HypSeek's
logs show it explicitly excludes DUD-E/DEKOIS/LIT-PCBA/CASF proteins), so
they act as a natural control.

Difficulty normalization
----------
Both splits are cut along protein family lines (cross's train half is
mostly enzymes and nuclear receptors, its test half all kinases and GPCRs),
so "the seen group scores higher" could just mean those targets are easier
to begin with. Divide out target difficulty first:

    r_t = EF(model, t) / median(EF(other nine models, t))

For the control models, r's within-group/out-of-group ratio should sit near
1; only a genuinely trained-on model should come out significantly higher.

Three-way partition
--------
Cut the 102 targets into three mutually exclusive parts and check for
monotonicity:
  A  union of the two splits' train targets (40)
  B  named only as test in both CSVs (17)
  C  never named in either CSV (45) -- this is ConPLex's clean DUD-E score

Usage::

    python standard/conplex_dude_leak.py
"""
import os
import sys

import numpy as np
from scipy.stats import kruskal, mannwhitneyu

B = "/data/work/vs-benchmark"
sys.path.insert(0, f"{B}/eval")
from metrics import bedroc, enrichment_factor, roc_auc          # noqa: E402

MODELS = [
    ("hypseek_rk",       f"{B}/results/hypseek_rk/DUDE"),
    ("litenclip",        f"{B}/results/litenclip/DUDE"),
    ("ligunity_pocket",  f"{B}/results/pocket_ranking/DUDE"),
    ("ligunity_protein", f"{B}/results/protein_ranking/DUDE"),
    ("bindclip_randneg", f"{B}/results/bindclip_randneg/DUDE"),
    ("drugclip",         f"{B}/results/drugclip/DUDE"),
    ("bindclip_hardneg", f"{B}/results/bindclip_hardneg/DUDE"),
    ("conglude",         f"{B}/results/t1_raw/conglude/DUDE"),
    ("conplex",          f"{B}/results/t1_raw/conplex/DUDE"),
    ("sprint",           f"{B}/results/t1_raw/sprint/DUDE"),
]
SPLIT_CSV = [f"{B}/dude_cross_full.csv", f"{B}/dude_within_full.csv"]


def load_splits():
    """Return (union of train targets, every target named in either CSV);
    target names are lowercased throughout."""
    train, allcsv = set(), set()
    for p in SPLIT_CSV:
        for line in open(p):
            q = line.strip().split(",")
            if len(q) < 2:
                continue
            allcsv.add(q[0].lower())
            if q[1] == "train":
                train.add(q[0].lower())
    return train, allcsv


def score(d):
    """The four metrics for one target. For runs that only saved embeddings,
    reconstruct the score via the official rule: max over pocket x molecule."""
    p = f"{d}/saved_preds.npy"
    if os.path.exists(p):
        s = np.load(p).reshape(-1)
    else:
        mp, pp = f"{d}/saved_mols_embed.npy", f"{d}/saved_target_embed.npy"
        if not os.path.exists(pp):
            pp = f"{d}/saved_pocket_embed.npy"
        if not (os.path.exists(mp) and os.path.exists(pp)):
            return None
        s = (np.load(pp) @ np.load(mp).T).max(axis=0)
    y = np.load(f"{d}/saved_labels.npy")
    if len(s) != len(y) or y.sum() in (0, len(y)):
        return None
    return dict(ef1=enrichment_factor(s, y, 0.01), ef5=enrichment_factor(s, y, 0.05),
                bedroc=bedroc(s, y, 80.5), auroc=roc_auc(s, y))


def main():
    TRAIN, ALLCSV = load_splits()
    M = {}
    for name, root in MODELS:
        for t in sorted(os.listdir(root)):
            v = score(f"{root}/{t}")
            if v:
                M.setdefault(t.lower(), {})[name] = v
    T = sorted(t for t in M if len(M[t]) == len(MODELS))
    print(f"十个模型都有结果的 DUD-E 靶点：{len(T)}")

    # per-target ratio after difficulty normalization
    R = {}
    for name, _ in MODELS:
        for t in T:
            base = float(np.median([M[t][m]["ef1"] for m, _ in MODELS if m != name]))
            if base > 0.01:                      # a target none of the models can do gives a meaningless ratio
                R.setdefault(name, {})[t] = M[t][name]["ef1"] / base

    A = [t for t in T if t in TRAIN]
    Bg = [t for t in T if t in ALLCSV and t not in TRAIN]
    C = [t for t in T if t not in ALLCSV]
    print(f"  A 对比学习 train 并集 {len(A)} · B 仅作 test 出现 {len(Bg)} · "
          f"C 从未出现 {len(C)}\n")

    print("难度归一化后的三段梯度（r = EF / 其余九个模型在该靶点的中位数）")
    print("%-18s %9s %9s %9s %10s %10s %10s"
          % ("模型", "A(train)", "B(test)", "C(never)", "p A>C", "p B>C", "三组 KW"))
    print("-" * 82)
    part = ["model,r_A_train,r_B_testonly,r_C_never,p_A_vs_C,p_B_vs_C,kruskal_p"]
    for m, _ in MODELS:
        a = [R[m][t] for t in A if t in R[m]]
        b = [R[m][t] for t in Bg if t in R[m]]
        c = [R[m][t] for t in C if t in R[m]]
        pac = mannwhitneyu(a, c, alternative="greater").pvalue
        pbc = mannwhitneyu(b, c, alternative="greater").pvalue
        kw = kruskal(a, b, c).pvalue
        print("%-18s %9.3f %9.3f %9.3f %10.5f %10.5f %10.5f"
              % (m, np.median(a), np.median(b), np.median(c), pac, pbc, kw))
        part.append("%s,%.4f,%.4f,%.4f,%.6f,%.6f,%.6f"
                    % (m, np.median(a), np.median(b), np.median(c), pac, pbc, kw))

    print("\n只留 C 段 45 个靶点的干净成绩（十个模型同一批靶点，内部仍可比）")
    print("%-18s %5s %8s %8s %8s %8s %12s %8s"
          % ("模型", "靶点", "EF1%", "EF5%", "BEDROC", "AUROC", "102靶点EF1%", "变化"))
    print("-" * 82)
    clean = ["model,n_targets,EF1,EF5,BEDROC,AUROC,EF1_all102,delta_pct"]
    for name, _ in MODELS:
        cv = [M[t][name] for t in C]
        av = [M[t][name] for t in T]
        mm = {k: float(np.mean([v[k] for v in cv])) for k in cv[0]}
        a1 = float(np.mean([v["ef1"] for v in av]))
        print("%-18s %5d %8.2f %8.2f %8.4f %8.4f %12.2f %+7.1f%%"
              % (name, len(cv), mm["ef1"], mm["ef5"], mm["bedroc"], mm["auroc"],
                 a1, (mm["ef1"] - a1) / a1 * 100))
        clean.append("%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f"
                     % (name, len(cv), mm["ef1"], mm["ef5"], mm["bedroc"],
                        mm["auroc"], a1, (mm["ef1"] - a1) / a1 * 100))

    o = f"{B}/results"
    open(f"{o}/T1_conplex_dude_partition.csv", "w").write("\n".join(part) + "\n")
    open(f"{o}/T1_dude_conplex_clean.csv", "w").write("\n".join(clean) + "\n")
    print(f"\n写出 {o}/T1_conplex_dude_partition.csv 和 T1_dude_conplex_clean.csv")


if __name__ == "__main__":
    main()
