"""T2 排序能力 × 配体新颖度——把「靶点变新」和「分子变新」拆开。

为什么要做
----------
T2 现在只报 L1–L4，也就是只按**靶点**新颖度分层。但 L1 的活性里有 53.9%
是训练配体的近复制品（Tanimoto ≥ 0.7），L4 只有 6.4%——所以 ρ 从 L1 的
+0.26 掉到 L4 的 +0.10，这条曲线里同时有两件事在变，我们却全算在靶点头上。

T1/T3 那边拆过一次同样的混杂：主表 L4 富集 9.38，按配体新颖度分档之后是
「见过的化学 27.1 / 全新化学 4.5」。T2 从来没拆过。这个脚本补上。

口径
----
· 新颖度 = 该分子对训练集配体的最大 Tanimoto（ECFP4），四档与
  `ligand_novelty.py` / `novelty_tiered_ef.py` 完全一致，直接复用它们的缓存。
· 在**每个靶点内部**，只用落在同一档的活性算 Spearman(模型分数, 实测 pAff)，
  该档少于 --min-tier 个活性就跳过这个靶点的这一档。
· 再对靶点取平均。同时给出 pooled 列（同一批靶点、不分档），
  这样「分档后」和「分档前」是在同一个靶点集合上比较的。

⚠️ 两个必须硬做的检查
--------------------
1. **分子顺序**。模型读 lmdb，游标是字典序（0,1,10,100,…），与评测集 jsonl
   顺序不同而长度相同；只比长度会静默错配，这个坑在本项目里出现过三次
   （PATCHES.md）。这里复用 `novelty_tiered_ef.py` 的硬校验：还原顺序后
   必须验「标签为 1 的位置上确实是该靶点的 active」，验不过就跳过并报出来。
2. **每档的 n 必须打出来**。Spearman 比 EF 对小样本敏感得多，T3 上 L1 的
   「全新 <0.35」档只有 18 个靶点，是全表最薄的一格——只报 ρ 不报 n 会
   让人把噪声当结论。

已知局限
--------
新颖度是相对 **PocketAffDB 的配体**算的（`train_label_blend_seq_full.json`），
不是相对各模型自己的训练配体。所以对非 PocketAffDB 训练的模型（DrugCLIP 系、
ConPLex、ConGLUDe、SPRINT），这一档划分只是个近似。要做逐模型版本需要各模型
自己的训练配体清单，目前只有 SPRINT 和 ConPLex 拿得到。
"""
import argparse
import collections
import csv
import json
import os
import pickle

import numpy as np
from scipy import stats

B = "/data/work/vs-benchmark"
TIERS = [(0.0, 0.35, "全新 <0.35"), (0.35, 0.50, "远 0.35–0.5"),
         (0.50, 0.70, "近 0.5–0.7"), (0.70, 1.01, "极近 ≥0.7")]
TIER_NAMES = [t[2] for t in TIERS]


def tier_of(v):
    for lo, hi, name in TIERS:
        if lo <= v < hi:
            return name
    return None


def model_order(up, L, n, rec, labels):
    """还原模型看到的分子顺序并硬校验；对不上返回 None。"""
    act = {x["smiles"] for x in rec["actives"]}

    def ok(seq):
        if seq is None or len(seq) != n:
            return None
        got = {seq[i] for i in range(n) if labels[i] == 1}
        return seq if got == act else None

    r = ok([x["smiles"] for x in rec["actives"]] +
           [x["smiles"] for x in rec["decoys"]])
    if r is not None:
        return r
    path = f"{B}/data/T3_6A/{L}/{up}/{up}_lig.lmdb"
    if not os.path.exists(path):
        return None
    try:
        import lmdb
        e = lmdb.open(path, subdir=False, readonly=True, lock=False)
        out = []
        with e.begin() as t:
            for _k, v in t.cursor():
                out.append(pickle.loads(v)["smi"])
        e.close()
    except Exception:
        return None
    return ok(out)


def rho(pairs):
    """pairs = [(score, paff)]；方差为 0 或点太少返回 None。"""
    s = np.array([p[0] for p in pairs], dtype=float)
    a = np.array([p[1] for p in pairs], dtype=float)
    if np.std(s) == 0 or np.std(a) == 0:
        return None
    r = stats.spearmanr(s, a).statistic
    return float(r) if np.isfinite(r) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--novelty", default=f"{B}/data/t3/ligand_novelty.json")
    ap.add_argument("--eval-dir", default=f"{B}/data/t3/eval")
    ap.add_argument("--raw", default=f"{B}/results/t3_raw")
    ap.add_argument("--models", nargs="+",
                    default=["hypseek_rk", "ligunity_protein_ranking",
                             "ligunity_pocket_ranking", "litenclip",
                             "conglude", "drugclip"])
    ap.add_argument("--layers", nargs="+", default=["L1", "L2", "L3", "L4"])
    ap.add_argument("--min-tier", type=int, default=5,
                    help="该靶点该档至少这么多活性才算；Spearman 比 EF 对小 n 敏感，"
                         "所以比 novelty_tiered_ef.py 的 3 高")
    ap.add_argument("--subset", default=None)
    ap.add_argument("--out", default=f"{B}/results/export/T2_novelty_tiers.csv")
    ap.add_argument("--out-paired", default=f"{B}/results/export/T2_novelty_paired.csv")
    args = ap.parse_args()

    keep = None
    if args.subset:
        keep = {(r["layer"], r["uniprot"]) for r in csv.DictReader(open(args.subset))}
        print(f"子集过滤：{len(keep)} 条")

    nov = json.load(open(args.novelty))
    recs = {L: [json.loads(x) for x in open(f"{args.eval_dir}/{L}.jsonl")]
            for L in args.layers}

    rows = [["model", "layer", "tier", "n_targets", "n_actives_median",
             "spearman_mean", "spearman_sem", "frac_positive"]]
    prows = [["model", "layer", "contrast", "n_paired_targets",
              "rho_familiar", "rho_novel", "delta", "wilcoxon_p",
              "n_familiar_wins"]]

    print("\nT2 排序能力 × 配体新颖度")
    print(f"（每靶点每档 ≥{args.min_tier} 个活性才计入；pooled = 同一批靶点不分档）")
    print("=" * 108)

    for m in args.models:
        # bucket[层][档] = [每靶点 ρ]；paired[层] = [(熟 ρ, 生 ρ)]
        bucket = collections.defaultdict(lambda: collections.defaultdict(list))
        nact = collections.defaultdict(lambda: collections.defaultdict(list))
        pooled = collections.defaultdict(list)
        paired = collections.defaultdict(list)
        paired_half = collections.defaultdict(list)
        n_bad = collections.Counter()

        for L in args.layers:
            d = f"{args.raw}/{m}/T3/{L}"
            if not os.path.isdir(d):
                d = f"{args.raw}/{m}/{L}"
            if not os.path.isdir(d):
                continue
            for r in recs[L]:
                up = r["uniprot"]
                if keep is not None and (L, up) not in keep:
                    continue
                try:
                    p = np.load(f"{d}/{up}/saved_preds.npy").reshape(-1)
                    y = np.load(f"{d}/{up}/saved_labels.npy")
                except Exception:
                    continue
                if len(p) != len(y):
                    continue
                order = model_order(up, L, len(y), r, y)
                if order is None:
                    n_bad[L] += 1
                    continue
                aff = {a["smiles"]: float(a["paff"]) for a in r["actives"]}

                by_tier = collections.defaultdict(list)  # 档名 -> [(score, paff)]
                allp = []
                for i in range(len(y)):
                    if y[i] != 1:
                        continue
                    smi = order[i]
                    if smi not in aff:
                        continue
                    t = tier_of(nov.get(smi, -1))
                    if t is None:
                        continue
                    by_tier[t].append((float(p[i]), aff[smi]))
                    allp.append((float(p[i]), aff[smi]))

                # 两半：生 <0.5 / 熟 ≥0.5。极端两档在同一靶点内同时够 5 个的很少，
                # 逐靶点配对会退化到 n=5~9；按两半分能让配对样本回到几十。
                half = {"生半 <0.5": by_tier[TIER_NAMES[0]] + by_tier[TIER_NAMES[1]],
                        "熟半 ≥0.5": by_tier[TIER_NAMES[2]] + by_tier[TIER_NAMES[3]]}
                hgot = {}
                for h, pr in half.items():
                    if len(pr) >= args.min_tier:
                        v = rho(pr)
                        if v is not None:
                            hgot[h] = v
                if len(hgot) == 2:
                    paired_half[L].append((hgot["熟半 ≥0.5"], hgot["生半 <0.5"]))

                got = {}
                for t, pr in by_tier.items():
                    if len(pr) < args.min_tier:
                        continue
                    v = rho(pr)
                    if v is None:
                        continue
                    bucket[L][t].append(v)
                    nact[L][t].append(len(pr))
                    got[t] = v
                if got:
                    v = rho(allp)
                    if v is not None:
                        pooled[L].append(v)
                # 逐靶点配对：同一个靶点内，熟化学 vs 全新化学
                if TIER_NAMES[3] in got and TIER_NAMES[0] in got:
                    paired[L].append((got[TIER_NAMES[3]], got[TIER_NAMES[0]]))

        print(f"\n{m}")
        print("-" * 108)
        print("%-5s %14s" % ("层", "pooled") +
              "".join("%20s" % t for t in TIER_NAMES))
        for L in args.layers:
            if L not in bucket:
                continue
            pv = pooled[L]
            cells = []
            for t in TIER_NAMES:
                v = bucket[L].get(t)
                if not v:
                    cells.append("—")
                    continue
                a = np.array(v)
                sem = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else float("nan")
                cells.append(f"{a.mean():+.3f} (n={len(a)})")
                rows.append([m, L, t, len(a), int(np.median(nact[L][t])),
                             f"{a.mean():.4f}", f"{sem:.4f}",
                             f"{(a > 0).mean():.3f}"])
            print("%-5s %14s" % (L, f"{np.mean(pv):+.3f} (n={len(pv)})" if pv else "—")
                  + "".join("%20s" % c for c in cells))
            if pv:
                rows.append([m, L, "pooled", len(pv), "",
                             f"{np.mean(pv):.4f}",
                             f"{np.std(pv, ddof=1)/np.sqrt(len(pv)):.4f}"
                             if len(pv) > 1 else "",
                             f"{(np.array(pv) > 0).mean():.3f}"])

        # 配对检验：均值会骗人，必须逐靶点配
        for L in args.layers:
            pr = paired[L]
            if len(pr) < 5:
                continue
            fam = np.array([x[0] for x in pr])
            new = np.array([x[1] for x in pr])
            try:
                w = stats.wilcoxon(fam, new).pvalue
            except Exception:
                w = float("nan")
            prows.append([m, L, "极近≥0.7 vs 全新<0.35", len(pr),
                          f"{fam.mean():.4f}", f"{new.mean():.4f}",
                          f"{(fam - new).mean():.4f}", f"{w:.4g}",
                          int((fam > new).sum())])
            print(f"  配对 极近≥0.7 vs 全新<0.35  {L}: "
                  f"n={len(pr)}  {fam.mean():+.3f} vs {new.mean():+.3f}  "
                  f"Δ={fam.mean()-new.mean():+.3f}  p={w:.3g}  "
                  f"熟胜 {int((fam>new).sum())}/{len(pr)}")

        for L in args.layers:
            pr = paired_half[L]
            if len(pr) < 5:
                continue
            fam = np.array([x[0] for x in pr])
            new = np.array([x[1] for x in pr])
            try:
                w = stats.wilcoxon(fam, new).pvalue
            except Exception:
                w = float("nan")
            prows.append([m, L, "熟半≥0.5 vs 生半<0.5", len(pr),
                          f"{fam.mean():.4f}", f"{new.mean():.4f}",
                          f"{(fam - new).mean():.4f}", f"{w:.4g}",
                          int((fam > new).sum())])
            print(f"  配对 熟半≥0.5 vs 生半<0.5   {L}: "
                  f"n={len(pr)}  {fam.mean():+.3f} vs {new.mean():+.3f}  "
                  f"Δ={fam.mean()-new.mean():+.3f}  p={w:.3g}  "
                  f"熟胜 {int((fam>new).sum())}/{len(pr)}")

        if n_bad:
            print("  ⚠️ 分子顺序校验未通过而跳过：" +
                  "  ".join(f"{L} {n}" for L, n in sorted(n_bad.items())))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    with open(args.out_paired, "w", newline="") as f:
        csv.writer(f).writerows(prows)
    print(f"\n写入 {args.out}")
    print(f"写入 {args.out_paired}")
    print("\n读法：如果 pooled 的 L1→L4 衰减在分档之后大幅变小，"
          "说明原来那条曲线主要是配体新颖度在动，不是靶点新颖度。")


if __name__ == "__main__":
    main()
