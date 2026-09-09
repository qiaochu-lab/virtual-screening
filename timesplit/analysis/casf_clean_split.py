"""CASF-2016 拆成「训练集里有的」和「没有的」两半，分别算打分力与排序力。

为什么要做
----------
`casf_train_overlap.py` 查出 **285 个 CASF 复合物里 148 个（51.9%）的 PDB ID
直接出现在 PocketAffDB 的训练文件里**——不是「相似」，是同一条 deposition。
四个模型（LigUnity ×2 / HypSeek / LiTENCLIP）都训在这份数据上。

这直接威胁 T2 的两条结论：
1. CASF 上 ρ = 0.42–0.55 而 T3 上只有 0.09–0.26；
2. 我们把这个差距归因于 T3 自己的 `pAff ≥ 6` 门限造成的范围受限
   （`t2_gap.py`，解释掉 75–91%）。

「CASF 高是因为一半是训练数据」是个**竞争解释**，和范围受限分不开。
这个脚本把它分开：如果干净那半的 ρ 掉下来，泄漏解释成立；
如果基本不动，范围受限那条反而被加固。

两个口径，分开报
----------------
· **打分力（scoring power）**：跨复合物比绝对亲和力。按**复合物**是否在训练集
  里拆两半——这一半干净不干净是复合物自己的属性，拆得干净。
· **排序力（ranking power）**：同一靶点内 5 个配体排序。一个靶点的 5 个复合物
  可能一半在训练集一半不在，所以按**靶点**分类：
    - 全脏 = 5 个全在训练集
    - 全净 = 一个都不在  ← 这一档才是真正的干净对照
    - 混合 = 其余
  只在「全净」上重算，才是没有泄漏的排序力。

⚠️ 这是**观察性**拆分，不是干预。干净/脏两半的靶点本身可能难度不同
（PDBbind 收录得早的往往是研究得透的经典靶点）。真正的干预版本是换
LigUnity 的三档 `--protein-similarity-thres` 权重重跑同一批复合物，
那需要 GPU，见 T2 文档。
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

B = "/data/work/vs-benchmark"
LAB = f"{B}/code/LigUnity/test_datasets/casf_label_seq.json"


def load_overlap(path):
    for line in open(path):
        if line.startswith("overlapping_pdb_ids"):
            return {x.strip().lower() for x in line.split("\t", 1)[1].split(",")
                    if x.strip()}
    raise SystemExit(f"{path} 里没有 overlapping_pdb_ids 行")


def load_truth():
    out = {}
    for e in json.load(open(LAB)):
        pdb = e["pockets"][0]
        out[pdb] = (e["ligands"][0]["act"], e.get("uniprot", "?"))
    return out


def scoring(score, y):
    ok = ~np.isnan(y)
    if ok.sum() < 5 or np.std(score[ok]) == 0:
        return None
    return (int(ok.sum()),
            float(stats.spearmanr(score[ok], y[ok]).statistic),
            float(stats.pearsonr(score[ok], y[ok]).statistic))


def ranking(per):
    """per: uniprot -> [(score, act)]；返回 (平均 ρ, 靶点数)。"""
    rs = []
    for v in per.values():
        if len(v) < 3:
            continue
        s = [x[0] for x in v]
        if np.std(s) == 0:
            continue
        r = stats.spearmanr(s, [x[1] for x in v]).statistic
        if not np.isnan(r):
            rs.append(r)
    return (float(np.mean(rs)), len(rs)) if rs else (float("nan"), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["hypseek_rk", "pocket_ranking", "protein_ranking",
                             "litenclip"])
    ap.add_argument("--overlap",
                    default=f"{B}/results/export/T2_casf_train_overlap.txt")
    ap.add_argument("--out", default=f"{B}/results/export/T2_casf_clean_split.csv")
    args = ap.parse_args()

    over = load_overlap(args.overlap)
    truth = load_truth()

    # 靶点分类：全脏 / 全净 / 混合
    tcx = defaultdict(list)
    for pdb, (_a, up) in truth.items():
        tcx[up].append(pdb in over)
    tclass = {}
    for up, flags in tcx.items():
        tclass[up] = "全脏" if all(flags) else ("全净" if not any(flags) else "混合")
    n_all = len(tclass)
    print(f"CASF：{len(truth)} 个复合物 / {n_all} 个靶点簇")
    print(f"复合物在训练集里：{len(over & set(truth))}")
    for c in ("全脏", "混合", "全净"):
        ups = [u for u, k in tclass.items() if k == c]
        print(f"  {c} 靶点 {len(ups):3d}  复合物 "
              f"{sum(len(tcx[u]) for u in ups):3d}")

    rows = [["model", "metric", "split", "n", "spearman", "pearson"]]
    print("\n打分力（跨复合物）与排序力（靶点内）")
    print("=" * 96)
    print("%-22s %10s %10s %10s %10s %10s %10s" %
          ("模型", "打分ρ 全部", "脏复合物", "净复合物", "排序ρ 全部",
           "全脏靶点", "全净靶点"))
    print("-" * 96)

    for m in args.models:
        d = f"{B}/results/{m}/PDBBind"
        try:
            ids = json.load(open(f"{d}/test_pdbbind_ids.json"))
            mol = np.load(f"{d}/test_mol_reps.npy")
            poc = np.load(f"{d}/test_pocket_reps.npy")
        except Exception as e:
            print(f"{m}: 读不到（{e}）")
            continue
        if not (len(ids) == len(mol) == len(poc)):
            print(f"{m}: 长度对不上，跳过")
            continue
        score = np.einsum("ij,ij->i", poc, mol)
        y = np.array([truth.get(p, (np.nan, "?"))[0] for p in ids], dtype=float)
        up = [truth.get(p, (np.nan, "?"))[1] for p in ids]
        dirty = np.array([p in over for p in ids])

        cells, res = [], {}
        for name, sel in (("全部", np.ones(len(ids), bool)),
                          ("脏复合物", dirty), ("净复合物", ~dirty)):
            r = scoring(score[sel], y[sel])
            res[name] = r
            cells.append(f"{r[1]:+.3f}({r[0]})" if r else "—")
            if r:
                rows.append([m, "scoring", name, r[0], f"{r[1]:.4f}", f"{r[2]:.4f}"])

        for name, keep in (("全部", None), ("全脏靶点", "全脏"), ("全净靶点", "全净")):
            per = defaultdict(list)
            for s, a, g in zip(score, y, up):
                if np.isnan(a):
                    continue
                if keep is not None and tclass.get(g) != keep:
                    continue
                per[g].append((s, a))
            rr, nt = ranking(per)
            cells.append(f"{rr:+.3f}({nt})" if nt else "—")
            if nt:
                rows.append([m, "ranking", name, nt, f"{rr:.4f}", ""])

        print("%-22s %10s %10s %10s %10s %10s %10s" % (m, *cells))

    print("-" * 96)
    print("括号里是 n（打分力=复合物数，排序力=靶点数）")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\n写入 {args.out}")
    print("\n读法：净复合物 / 全净靶点这两列掉得多，说明 CASF 的成绩靠泄漏撑；"
          "基本不动，则范围受限（t2_gap.py）仍是 CASF–T3 差距的主解释。")


if __name__ == "__main__":
    main()
