"""Split CASF-2016 into "in the training set" and "not in it", and compute
scoring power and ranking power separately on each half.

Why this is needed
-------------------
`casf_train_overlap.py` found that **148 of 285 CASF complexes (51.9%) have a
PDB ID that appears directly in PocketAffDB's training file** — not "similar",
the same deposition. Four models (LigUnity x2 / HypSeek / LiTENCLIP) all
trained on this data.

This directly threatens two T2 conclusions:
1. CASF gives rho = 0.42-0.55 while T3 gives only 0.09-0.26;
2. We attributed that gap to the restriction of range caused by T3's own
   `pAff >= 6` cutoff (`t2_gap.py`, explaining away 75-91% of it).

"CASF is high because half of it is training data" is a **competing
explanation**, and it is entangled with restriction of range. This script
separates them: if rho collapses on the clean half, the leakage explanation
holds; if it barely moves, the restriction-of-range explanation is instead
reinforced.

Two conventions, reported separately
-------------------------------------
- **Scoring power**: absolute affinity compared across complexes. Split into
  two halves by whether the **complex** is in the training set — whether
  this half is clean is a property of the complex itself, so the split is
  clean.
- **Ranking power**: ranking the 5 ligands within one target. A target's 5
  complexes may be half in the training set and half not, so classify by
  **target** instead:
    - fully dirty = all 5 in the training set
    - fully clean = none in it  <- this tier is the real clean control
    - mixed = everything else
  Only recomputing on "fully clean" gives ranking power with no leakage.

Warning: this is an **observational** split, not an intervention. The
targets in the clean/dirty halves may themselves differ in difficulty
(targets deposited early in PDBbind tend to be well-studied, classic
targets). The real interventional version would be to rerun the same
complexes with LigUnity's three `--protein-similarity-thres` weight tiers,
which needs a GPU — see the T2 documentation.
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
    """per: uniprot -> [(score, act)]; returns (mean rho, n_targets, (lo, hi)).

    There are only a dozen or so targets, so a mean with no interval can't be
    read; bootstrap-resample over targets instead.
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
    """Restriction-of-range correction (case II): rescale an r measured on a
    narrow spread to what it would be on a spread k times wider."""
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

    # Classify targets: fully dirty / fully clean / mixed
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

    # Warning: confound to rule out first -- could the clean targets simply
    # have a narrower affinity spread themselves? In that case a low ranking
    # rho would again be restriction of range (the t2_gap.py finding), not
    # leakage.
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
            # strict=True: all three are built from the same ids, so their
            # lengths must match. A bare zip silently truncates on unequal
            # lengths -- three of this project's four indexing bugs slipped
            # past review exactly this way.
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
    # The fully-clean targets' spread is narrower than the fully-dirty ones',
    # so comparing rho directly isn't fair -- first rescale to the same
    # spread via Thorndike
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
