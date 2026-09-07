"""从 T3 里选一个「活性 ≥50 且类别构成对齐 VSDS-vd」的子集。

背景
----
导师 2026-09-04 定的两条：① 每个靶点活性分子 ≥50；② 靶点类别参考
VSDS-vd（Gu et al., Nat Mach Intell 7:509-520, 2025，DOI 10.1038/s42256-025-00993-0）。

VSDS-vd 的类别构成不是论文里现成的表，是把 Zenodo 上的数据集
(https://zenodo.org/records/13684010) 下下来、按 DTEBV-D 子集的 147 个
UniProt 目录、用**与本仓库 annotate_target_class3.py 完全相同**的 ChEMBL
分类树口径重新标注得到的。两边同口径是这个比对唯一有意义的前提。

两个改不掉的缺口
----------------
核受体和 P450 在 ≥50 的池子里只有 6 个和 2 个，按 VSDS-vd 比例分别需要
17 和 9 个。**全部拿光也不够**——不是筛选口径的问题：这两个家族研究得早、
成员少，时间切分之后基本不出新靶点（L3+L4 两层里它们都是 0 个）。
如实报成 limitation，不要为了配平这两类把整体规模砍到 49（P450 的天花板）。

为什么要在「类别」和「分层」两个方向同时配平
--------------------------------------------
第一版只按类别配额、类内优先塞 L3/L4，结果「其他酶」46 个名额被 L4 一层吃光，
L1/L2 里这个最大的类变成 0 个——层间构成完全失衡，分层比较就没意义了。
现在用带容量上限的迭代比例配平（IPF）：行边际=VSDS-vd 的类别比例，
列边际=候选池本身的分层比例，每格不超过实际存货，最后按最大余数取整。

类内为什么用随机抽样而不是「取活性数最多的」
--------------------------------------------
两种都能填满配额，但按活性数取会系统性偏向被研究透的热门靶点，
并且把每靶点活性数的跨度进一步拉大（池子里最大 3262 个）。
固定种子随机抽样保持活性数分布不失真。
"""
import argparse
import collections
import csv
import json
import random

# VSDS-vd DTEBV-D 147 个靶点的类别构成（同口径重标；见模块 docstring）
VSDS = {"激酶": 35, "其他酶": 27, "GPCR": 23, "蛋白酶": 23, "核受体": 11,
        "表观": 10, "其他/未分类": 7, "P450": 6, "离子通道": 3, "转运体": 2}
ORDER = ["激酶", "其他酶", "GPCR", "蛋白酶", "核受体", "表观",
         "离子通道", "P450", "转运体", "其他/未分类"]
LAYERS = ["L1", "L2", "L3", "L4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="results/T3_targets.csv")
    ap.add_argument("--min-actives", type=int, default=50)
    ap.add_argument("--quota", type=int, default=250,
                    help="名义配额；供给不足的类别会少于配额，所以实得数更小")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/T3_vsds_matched.csv")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.targets))
            if int(r["n_actives"]) >= args.min_actives]
    nv = sum(VSDS.values())
    p = {c: VSDS[c] / nv for c in VSDS}

    pool = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        pool[r["protein_class"]][r["layer"]].append(r)

    cap = {(c, L): len(pool[c].get(L, [])) for c in ORDER for L in LAYERS}
    # L3 是最薄的一层（新靶点 / 家族见过），池子里本来就只有二十来个，
    # 按比例分只会剩个位数、那一层直接作废，所以整层全留，不参与比例分配。
    n_layer = collections.Counter(r["layer"] for r in rows)
    rest = [L for L in LAYERS if L != "L3"]
    n_rest = sum(n_layer[L] for L in rest)
    q_layer = {"L3": float(n_layer["L3"])}
    for L in rest:
        q_layer[L] = (args.quota - n_layer["L3"]) * n_layer[L] / n_rest
    q_class = {c: args.quota * p.get(c, 0) for c in ORDER}

    # 带上限的迭代比例配平：行=类别，列=分层，每格不超过存货
    x = {k: min(cap[k], 1.0) for k in cap}
    for _ in range(200):
        for c in ORDER:                                   # 行归一
            s_ = sum(x[(c, L)] for L in LAYERS)
            if s_ > 0:
                for L in LAYERS:
                    x[(c, L)] = min(cap[(c, L)], x[(c, L)] * q_class[c] / s_)
        for L in LAYERS:                                  # 列归一
            s_ = sum(x[(c, L)] for c in ORDER)
            if s_ > 0:
                for c in ORDER:
                    x[(c, L)] = min(cap[(c, L)], x[(c, L)] * q_layer[L] / s_)

    # 最大余数法取整，仍受存货上限约束
    alloc = {k: min(cap[k], int(v)) for k, v in x.items()}
    frac = sorted(((x[k] - int(x[k]), k) for k in x if alloc[k] < cap[k]), reverse=True)
    need = round(sum(min(cap[k], x[k]) for k in x)) - sum(alloc.values())
    for _, k in frac[:max(0, need)]:
        alloc[k] += 1

    rng = random.Random(args.seed)
    keep, short = [], []
    for c in ORDER:
        got_c = 0
        for L in LAYERS:
            avail = list(pool[c].get(L, []))
            rng.shuffle(avail)
            keep += avail[:alloc[(c, L)]]
            got_c += alloc[(c, L)]
        if got_c < round(q_class[c]):
            short.append((c, round(q_class[c]), got_c))

    n = len(keep)
    got = collections.Counter(r["protein_class"] for r in keep)
    lay = collections.Counter(r["layer"] for r in keep)

    print(f"活性门槛 ≥{args.min_actives}：候选池 {len(rows)} 个靶点")
    print(f"名义配额 {args.quota} → 实得 {n} 个\n")
    print("%-11s %9s %14s %8s" % ("类别", "VSDS-vd", "本子集", "偏差"))
    print("-" * 46)
    for c in ORDER:
        pv, pb = 100 * p.get(c, 0), 100 * got[c] / n
        print("%-11s %8.1f%% %14s %+7.1f" %
              (c, pv, f"{got[c]} ({pb:.0f}%)", pb - pv))
    print("-" * 46)
    dev = max(abs(100 * got[c] / n - 100 * p.get(c, 0)) for c in ORDER)
    print("%-11s %9s %14s %+7.1fpp" % ("最大偏差", "", "", dev))

    print("\n分层：" + " · ".join(f"{L} {lay[L]}" for L in LAYERS))
    na = sorted(int(r["n_actives"]) for r in keep)
    print(f"每靶点活性数：最小 {na[0]} · 中位 {na[len(na)//2]} · 最大 {na[-1]}"
          f"   （VSDS-vd 中位 28）")

    if short:
        print("\n⚠️ 供给不足、已全部取用的类别（写进 limitation）：")
        for c, q, g in short:
            print(f"   {c:<8} 需 {q:>3}，池子里只有 {g:>3}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(keep, key=lambda r: (r["layer"], r["protein_class"],
                                                r["uniprot"])))
    print(f"\n靶点清单写入 {args.out}")


if __name__ == "__main__":
    main()
