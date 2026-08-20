"""CASF-2016 上的打分能力与排序能力（T2 的第三套数据）。

为什么要这套
------------
T2 现在有两套数据，结论相反：FEP 基准（同一化学系列）ρ≈0.4，
自建 T3（跨系列）ρ≈0。CASF-2016 落在中间——**同一靶点、但配体骨架不同**，
而且是打分函数领域用了十年的标准集，物理/经验打分函数的数字文献里可查。

两个官方口径，分开报
--------------------
· scoring power  285 个复合物一起算相关（跨靶点，考绝对亲和力）
· ranking power  每个靶点内部 5 个配体排序，再对靶点取平均（靶点内，考排序）
两者测的不是一回事，混着报会得出互相矛盾的结论——T2 前面吃过这个亏。

打分怎么来
----------
模型只落盘了 embedding。分数 = 配对的口袋向量与分子向量的内积，
与官方 ensemble_result.py 的做法一致（同一复合物一一对应，不做交叉）。

靶点分组用 casf_label_seq.json 里的 uniprot——CASF 的 57 个簇本来就是
「同一蛋白 5 个配体」，按 uniprot 分组能复原这个结构。
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

B = "/data/work/vs-benchmark"
LAB = f"{B}/code/LigUnity/test_datasets/casf_label_seq.json"


def load_truth():
    out = {}
    for e in json.load(open(LAB)):
        pdb = e["pockets"][0]
        out[pdb] = (e["ligands"][0]["act"], e.get("uniprot", "?"))
    return out


def main(models):
    truth = load_truth()
    print("CASF-2016（285 个复合物）")
    print("=" * 78)
    print("%-24s %8s %10s %10s %12s %8s" %
          ("模型", "n", "打分ρ", "打分r", "靶点内排序ρ", "靶点数"))
    print("-" * 78)
    for m in models:
        d = f"{B}/results/{m}/PDBBind"
        try:
            ids = json.load(open(f"{d}/test_pdbbind_ids.json"))
            mol = np.load(f"{d}/test_mol_reps.npy")
            poc = np.load(f"{d}/test_pocket_reps.npy")
        except Exception as e:
            print(f"{m}: 读不到（{e}）")
            continue
        if not (len(ids) == len(mol) == len(poc)):
            print(f"{m}: 长度对不上 ids={len(ids)} mol={len(mol)} poc={len(poc)}，跳过")
            continue
        score = np.einsum("ij,ij->i", poc, mol)          # 配对内积
        y, grp = [], []
        for pdb in ids:
            a, up = truth.get(pdb, (np.nan, "?"))
            y.append(a); grp.append(up)
        y = np.array(y, dtype=float)
        ok = ~np.isnan(y)
        sp = stats.spearmanr(score[ok], y[ok]).statistic
        pr = stats.pearsonr(score[ok], y[ok]).statistic

        per = defaultdict(list)
        for s, a, g in zip(score, y, grp):
            if not np.isnan(a):
                per[g].append((s, a))
        rhos = [stats.spearmanr([x[0] for x in v], [x[1] for x in v]).statistic
                for v in per.values() if len(v) >= 3 and np.std([x[0] for x in v]) > 0]
        rhos = [r for r in rhos if not np.isnan(r)]
        print("%-24s %8d %10.3f %10.3f %12.3f %8d" %
              (m, int(ok.sum()), sp, pr, float(np.mean(rhos)) if rhos else float("nan"),
               len(rhos)))

    print("-" * 78)
    print("\n怎么读")
    print("· 打分 ρ/r = 跨靶点比绝对结合强度（CASF 的 scoring power）")
    print("· 靶点内排序 ρ = 同一靶点 5 个不同骨架的配体谁强谁弱（ranking power）")
    print("· 与 T2 另外两套对照：FEP 同系列 ρ≈0.4，T3 跨系列 ρ≈0；")
    print("  CASF 是「同靶点、跨骨架」，正好落在中间，用来定位分界线在哪")


if __name__ == "__main__":
    main(sys.argv[1:] or ["pocket_ranking", "protein_ranking"])
