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


def ranking(per, boot=2000, seed=0):
    """per: uniprot -> [(score, act)]；返回 (平均 ρ, 靶点数, (lo, hi))。

    靶点数只有十几个，均值没有区间就没法读，所以对靶点做自助重采样。
    """
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
    if not rs:
        return float("nan"), 0, (float("nan"), float("nan"))
    a = np.array(rs)
    rng = np.random.default_rng(seed)
    bs = a[rng.integers(0, len(a), size=(boot, len(a)))].mean(axis=1)
    return float(a.mean()), len(a), (float(np.percentile(bs, 2.5)),
                                     float(np.percentile(bs, 97.5)))


def thorndike(r, k):
    """范围受限校正（case II）：把在窄展布上测到的 r 折算到宽 k 倍的展布上。"""
    return r * k / np.sqrt(1 + r * r * (k * k - 1))


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
    tcx = defaultdict(list)                     # uniprot -> [pdb]
    for pdb, (_a, up) in truth.items():
        tcx[up].append(pdb)
    tclass = {}
    for up, pdbs in tcx.items():
        flags = [p in over for p in pdbs]
        tclass[up] = "全脏" if all(flags) else ("全净" if not any(flags) else "混合")
    n_all = len(tclass)
    print(f"CASF：{len(truth)} 个复合物 / {n_all} 个靶点簇")
    print(f"复合物在训练集里：{len(over & set(truth))}")
    for c in ("全脏", "混合", "全净"):
        ups = [u for u, k in tclass.items() if k == c]
        print(f"  {c} 靶点 {len(ups):3d}  复合物 "
              f"{sum(len(tcx[u]) for u in ups):3d}")

    # ⚠️ 必须先排掉的混杂：干净靶点会不会本身亲和力展布就窄？
    # 那样排序 ρ 低就又是范围受限（t2_gap.py 那条），不是泄漏。
    print("\n混杂检查：各类靶点的靶点内 pAff 展布")
    print("%-8s %8s %10s %12s %10s" % ("靶点类", "靶点数", "配体中位", "SD 中位", "极差中位"))
    spread = {}
    for c in ("全脏", "混合", "全净"):
        sds, rngs, ns = [], [], []
        for u, k in tclass.items():
            if k != c:
                continue
            acts = [truth[p][0] for p in tcx[u] if not np.isnan(truth[p][0])]
            if len(acts) < 3:
                continue
            sds.append(np.std(acts, ddof=1))
            rngs.append(max(acts) - min(acts))
            ns.append(len(acts))
        if sds:
            spread[c] = (np.median(sds), np.median(rngs))
            print("%-8s %8d %10.0f %12.3f %10.3f" %
                  (c, len(sds), np.median(ns), np.median(sds), np.median(rngs)))
    if "全脏" in spread and "全净" in spread:
        ratio = spread["全脏"][0] / spread["全净"][0]
        print(f"  展布比（全脏/全净）= {ratio:.2f}"
              f"{'  ← 接近 1，排序 ρ 的差不是展布造成的' if 0.85 < ratio < 1.18 else '  ← 偏离 1，两组不可直接比，须先做范围受限校正'}")

    rows = [["model", "metric", "split", "n", "spearman", "pearson"]]
    rank_res = {}
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
            # strict=True：三者由同一个 ids 构造，长度必须相等。裸 zip 遇到
            # 不等长会静默截断——本项目的四次索引 bug 有三次是这样躲过检查的。
            for s, a, g in zip(score, y, up, strict=True):
                if np.isnan(a):
                    continue
                if keep is not None and tclass.get(g) != keep:
                    continue
                per[g].append((s, a))
            rr, nt, ci = ranking(per)
            cells.append(f"{rr:+.3f}({nt})" if nt else "—")
            if nt:
                rows.append([m, "ranking", name, nt, f"{rr:.4f}", ""])
                rank_res.setdefault(m, {})[name] = (rr, nt, ci)

        print("%-22s %10s %10s %10s %10s %10s %10s" % (m, *cells))

    print("-" * 96)
    print("括号里是 n（打分力=复合物数，排序力=靶点数）")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\n写入 {args.out}")
    # 全净靶点的展布比全脏窄，所以直接比 ρ 不公平——先按 Thorndike 折算到同一展布
    k = spread["全脏"][0] / spread["全净"][0] if ("全脏" in spread and "全净" in spread) else None
    if k:
        print(f"\n把「全净靶点」的 ρ 按展布比 k={k:.2f} 校正到「全脏靶点」的展布上")
        print("（自助区间 = 对靶点重采样 2000 次的 95%）")
        print("%-22s %22s %22s %12s" %
              ("模型", "全脏靶点 ρ [95%]", "全净靶点 ρ [95%]", "全净校正后"))
        print("-" * 84)
        for m, d in rank_res.items():
            if "全脏靶点" not in d or "全净靶点" not in d:
                continue
            (rd, nd, cd), (rc, nc, cc) = d["全脏靶点"], d["全净靶点"]
            corr = thorndike(rc, k)
            rows.append([m, "ranking", "全净靶点_校正", nc, f"{corr:.4f}", ""])
            gap = "泄漏解释成立" if corr < cd[0] else "区间重叠，分不开"
            print("%-22s %22s %22s %12s  %s" %
                  (m, f"{rd:+.3f} [{cd[0]:+.3f},{cd[1]:+.3f}]",
                   f"{rc:+.3f} [{cc[0]:+.3f},{cc[1]:+.3f}]",
                   f"{corr:+.3f}", gap))
        print("-" * 84)
        print("「泄漏解释成立」= 校正后的全净值仍落在全脏的 95% 区间之下")

    print("\n读法：净复合物 / 全净靶点这两列掉得多，说明 CASF 的成绩靠泄漏撑；"
          "基本不动，则范围受限（t2_gap.py）仍是 CASF–T3 差距的主解释。")


if __name__ == "__main__":
    main()
