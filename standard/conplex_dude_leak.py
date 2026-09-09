"""ConPLex 在 DUD-E 上的靶点泄漏检验。

背景
----
ConPLex 的核心方法就是「用 DUD-E 诱饵做对比学习」——configs/default_config.yaml
里 ``contrastive: True`` 是默认值，训练时按 ``dataset/DUDe/dude_*_train_test_split.csv``
取 train 那一半靶点。仓库给了两套划分（cross / within），各 26 个靶点，重叠 12 个，
并集 40 个；两个 CSV 一共涉及 57 个 DUD-E 靶点。

而我们的 T1 用的是 **全部 102 个** DUD-E 靶点，那 40 个训练靶点全在里面。
其余九个模型的训练数据都不含 DUD-E（HypSeek 的日志里能看到它显式剔除了
DUD-E/DEKOIS/LIT-PCBA/CASF 的蛋白），所以它们是天然对照。

难度归一化
----------
两套划分都是按蛋白家族切的（cross 的 train 多为酶和核受体，test 全是激酶和
GPCR），所以「见过组更高」可能只是这批靶点本身好做。先把靶点难度除掉：

    r_t = EF(模型, t) / median(EF(其余九个模型, t))

对照模型的 r 组内/组外比值应当接近 1；只有真正训练过的模型才会显著偏高。

三段划分
--------
把 102 个靶点切成互斥三段，看是不是单调的：
  A 两套划分的 train 并集（40）
  B 只在 CSV 里出现、两套都标 test（17）
  C 两个 CSV 都没出现过（45）—— 这一段才是 ConPLex 干净的 DUD-E 成绩

用法::

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
    """返回 (train 并集, 两个 CSV 出现过的全部靶点)，靶点名一律小写。"""
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
    """一个靶点的四个指标。只落了 embedding 的按官方规则口袋×分子取 max。"""
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

    # 难度归一化的每靶点比值
    R = {}
    for name, _ in MODELS:
        for t in T:
            base = float(np.median([M[t][m]["ef1"] for m, _ in MODELS if m != name]))
            if base > 0.01:                      # 全员都做不了的靶点，比值没意义
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
