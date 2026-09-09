"""NEW-4：靶点新颖度 × 配体新颖度的**正式**交互检验（任务清单 3.2 / Fig. 3）。

要检验的模型
------------
    Performance ~ TargetNovelty + LigandNovelty + TargetNovelty × LigandNovelty

之前只做了描述性 2×2（比「新/熟比」在 seen 和 unseen 两组的中位数），
那说明不了交互项是不是统计上为零。这里做正式的。

为什么用二项 GLM 而不是对 EF 做回归
-----------------------------------
EF_t = recall_t / 0.01 有大量精确的 0（某档活性一个都没进 top-1%），
取对数会变成 -inf，加个 epsilon 又让结果依赖 epsilon 怎么选。

回到 EF 的定义本身：它就是「该档活性进 top-1% 的比例」除以 0.01。
所以直接把**每个活性是否进 top-1%** 当伯努利响应，做 logit 回归：

    logit P(进 top-1%) = β0 + β1·TargetSeen + β2·LigandTier + β3·(TargetSeen × LigandTier)

零天然被处理，不需要任何变换。β3 就是交互项。

为什么 bootstrap 而不是 GLM 自带的标准误
----------------------------------------
同一个靶点的活性不独立（同系列），GLM 假定独立会把标准误算得过小。
任务清单允许「target-level bootstrap 或 mixed-effects」，这里用前者：
**按靶点有放回重抽**，每次重抽后重新拟合，取 β3 的 2.5/97.5 分位。
这把靶点内相关性吸收进了重抽单元，不需要假定随机效应的分布形式。

⚠️ 参照系必须逐模型
------------------
配体新颖度必须按**各模型自己的训练配体**算：结构半那三个模型用
`ligand_novelty_drugclip.json`（13,590 个配体），亲和力半那四个用
`ligand_novelty.json`（428,767 个）。混用会让交互项是假的——同一个分子在
两套参照系下能差 0.4 个 Tanimoto。ConPLex / SPRINT 的缓存还没有，先不进模型。
"""
import argparse
import collections
import csv
import json
import os
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression

B = "/data/work/vs-benchmark"
TIERS = [(0.0, 0.35), (0.35, 0.50), (0.50, 0.70), (0.70, 1.01)]
TIER_NAME = ["全新 <0.35", "远 0.35–0.5", "近 0.5–0.7", "极近 ≥0.7"]
FRAC = 0.01
MIN_T = 3

# 模型 → (训练靶点集合的来源, 配体新颖度缓存)
AREF = "ligand_novelty_drugclip.json"      # 13,590 个训练配体
BREF = "ligand_novelty.json"               # 428,767
CREF = "ligand_novelty_conplex.json"       # 3,814
DREF = "ligand_novelty_sprint.json"        # 1,390,031
MODELS = {
    "drugclip": ("A", AREF), "bindclip_randneg": ("A", AREF),
    "bindclip_hardneg": ("A", AREF),
    "ligunity_pocket_ranking": ("B", BREF), "ligunity_protein_ranking": ("B", BREF),
    "litenclip": ("B", BREF), "hypseek_rk": ("B", BREF),
    "conplex": ("C", CREF), "sprint": ("D", DREF),
}
# ⚠️ 训练配体池差两个数量级（3,814 → 1,390,031），「新颖度」在不同模型之间
# 根本不是同一个量。ConPLex 有 87.9% 的 T3 分子落在最低档、只有 0.5% 在最高档，
# 这个自变量几乎没有方差——即使 z 标准化也不改变偏度。所以下面把每个模型
# novelty 的 IQR 一起报出来，宽 CI 要能读成「功效不足」而不是「没有交互」。


def tier_of(v):
    for i, (lo, hi) in enumerate(TIERS):
        if lo <= v < hi:
            return i
    return None


def model_order(up, L, n, rec, labels, root):
    """还原模型看到的分子顺序并硬校验；对不上返回 None。"""
    act = {x["smiles"] for x in rec["actives"]}

    def ok(seq):
        if seq is None or len(seq) != n:
            return None
        return seq if {seq[i] for i in range(n) if labels[i] == 1} == act else None

    r = ok([x["smiles"] for x in rec["actives"]] + [x["smiles"] for x in rec["decoys"]])
    if r is not None:
        return r
    p = f"{root}/data/T3_6A/{L}/{up}/{up}_lig.lmdb"
    if not os.path.exists(p):
        return None
    try:
        import lmdb
        e = lmdb.open(p, subdir=False, readonly=True, lock=False)
        out = []
        with e.begin() as t:
            for _k, v in t.cursor():
                out.append(pickle.loads(v)["smi"])
        e.close()
    except Exception:
        return None
    return ok(out)


def load_train_sets(root):
    lab = []
    for f in ("train_label_blend_seq_full.json", "train_label/train_label_pdbbind_seq.json"):
        p = f"{root}/data/raw/figshare/{f}"
        if os.path.exists(p):
            lab += json.load(open(p))
    Bs = {a["uniprot"] for a in lab if a.get("uniprot")}
    import lmdb
    pdb2up = json.load(open(f"{root}/data/t3/drugclip_pdb2uniprot.json"))
    e = lmdb.open(f"{root}/data/train_no_test_af/train.lmdb", subdir=False,
                  readonly=True, lock=False)
    A = set()
    with e.begin() as t:
        for _k, v in t.cursor():
            pk = pickle.loads(v).get("pocket")
            if pk:
                A |= set(pdb2up.get(str(pk).split("_")[0].upper()[:4], []))
    e.close()
    C = set()
    pc = f"{root}/results/export/T3_conplex_train_coverage.csv"
    if os.path.exists(pc):
        C = {r["uniprot"] for r in csv.DictReader(open(pc))
             if float(r["best_identity_conplex_bindingdb"]) >= 0.95
             or r["in_dude_57"] == "1"}
    D = set()
    pd_ = f"{root}/data/sprint_train_uniprots.txt"
    if os.path.exists(pd_):
        D = {l.strip() for l in open(pd_) if l.strip()}
    return {"A": A, "B": Bs, "C": C, "D": D}


def fit(rows):
    """加权 logit：每个 (靶点, 档) 贡献 hit 次成功和 tot-hit 次失败。

    返回 (β_target, β_tier, β_interaction)。用 sample_weight 而不是把每个活性
    展开成一行，等价但快得多。
    """
    X, y, w = [], [], []
    for seen, tier, hit, tot in rows:
        if tot <= 0:
            continue
        f = [seen, tier, seen * tier]
        if hit > 0:
            X.append(f); y.append(1); w.append(hit)
        if tot - hit > 0:
            X.append(f); y.append(0); w.append(tot - hit)
    if len({v for v in y}) < 2:
        return None
    m = LogisticRegression(penalty=None, max_iter=2000, solver="lbfgs")
    m.fit(np.array(X, float), np.array(y), sample_weight=np.array(w, float))
    return m.coef_[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=B)
    ap.add_argument("--eval-dir", default=f"{B}/data/t3/eval")
    ap.add_argument("--raw", default=f"{B}/results/t3_raw")
    ap.add_argument("--subset", default=f"{B}/results/export/T3_vsds_matched.csv")
    ap.add_argument("--layers", nargs="+", default=["L1", "L2", "L3", "L4"])
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default=f"{B}/results/export/T3_novelty_interaction.csv")
    ap.add_argument("--cells-out",
                    default=f"{B}/results/export/T3_novelty_interaction_cells.csv")
    args = ap.parse_args()

    # ⚠️ 键必须是 (层, 靶点)：350 子集 328 条 / 293 个唯一 uniprot，
    # 35 个 uniprot 出现在多个层，只按 uniprot 过滤会多算别的层的记录。
    keep = {(r["layer"], r["uniprot"]) for r in csv.DictReader(open(args.subset))}
    SETS = load_train_sets(args.root)
    # 从 MODELS 里收集实际用到的缓存，别硬编码——加模型时最容易漏这一行
    novc = {f: json.load(open(f"{args.root}/data/t3/{f}"))
            for f in {ref for _tag, ref in MODELS.values()}}
    recs = {L: [json.loads(x) for x in open(f"{args.eval_dir}/{L}.jsonl")]
            for L in args.layers}

    # cells[model] = [(seen, tier, hit, tot, uniprot), ...]
    cells = collections.defaultdict(list)
    novvals = collections.defaultdict(list)   # 该模型参照系下的活性新颖度分布
    iqr = {}
    for m, (tag, ref) in MODELS.items():
        nov = novc[ref]
        T = SETS[tag]
        n_bad = 0
        for L in args.layers:
            # 结果目录布局不统一：口袋类模型写 results/t3_raw/{m}/T3/{层}/，
            # 序列类模型（ConPLex 等）少一层 T3 且在 results/t3/ 下。
            # 不列全候选路径，ConPLex 会静默产出 0 个格子而不报错。
            d = None
            for cand in (f"{args.raw}/{m}/T3/{L}",
                         f"{args.raw}/{m}/{L}",
                         f"{os.path.dirname(args.raw)}/t3/{m}/{L}"):
                if os.path.isdir(cand):
                    d = cand
                    break
            if d is None:
                continue
            for r in recs[L]:
                up = r["uniprot"]
                if (L, up) not in keep:
                    continue
                try:
                    p = np.load(f"{d}/{up}/saved_preds.npy").reshape(-1)
                    yv = np.load(f"{d}/{up}/saved_labels.npy")
                except Exception:
                    continue
                if len(p) != len(yv):
                    continue
                order = model_order(up, L, len(yv), r, yv, args.root)
                if order is None:
                    n_bad += 1
                    continue
                k = max(1, int(np.ceil(FRAC * len(yv))))
                top = set(np.argsort(-p)[:k].tolist())
                tot = collections.Counter()
                hit = collections.Counter()
                for i in range(len(yv)):
                    if yv[i] != 1:
                        continue
                    t = tier_of(nov.get(order[i], -1))
                    if t is None:
                        continue
                    tot[t] += 1
                    if i in top:
                        hit[t] += 1
                seen = 1 if up in T else 0
                for i in range(len(yv)):
                    if yv[i] == 1:
                        v = nov.get(order[i], -1)
                        if v >= 0:
                            novvals[m].append(v)
                for t in tot:
                    if tot[t] >= MIN_T:
                        cells[m].append((seen, t, hit[t], tot[t], (L, up)))
        v = np.array(novvals[m]) if novvals[m] else np.array([0.0])
        q1, q2, q3 = np.percentile(v, [25, 50, 75])
        print(f"{m:26} {tag} 格子 {len(cells[m]):5d}  活性新颖度 "
              f"中位 {q2:.3f} IQR [{q1:.3f}, {q3:.3f}] 宽 {q3-q1:.3f}"
              f"  ⚠️{n_bad} 顺序校验不过", flush=True)
        iqr[m] = (q1, q2, q3)

    rows = [["model", "train_set", "n_cells", "n_targets", "novelty_iqr_width",
             "beta_target", "beta_tier", "beta_interaction",
             "ci_lo", "ci_hi", "p_two_sided", "interaction_sign"]]
    cell_rows = [["model", "uniprot", "target_seen", "tier", "hit", "tot", "ef_tier"]]
    print("\n正式交互检验：logit P(活性进 top-1%) ~ 靶点见过 + 配体档 + 交互")
    print(f"靶点级 bootstrap {args.boot} 次，2.5/97.5 分位\n")
    print("%-26s %6s %7s %10s %10s %12s %18s %7s"
          % ("模型", "格子", "靶点", "β靶点", "β配体档", "β交互", "95% CI", "p"))
    print("-" * 108)
    rng = np.random.default_rng(1)
    for m in MODELS:
        C = cells[m]
        if len(C) < 20:
            print(f"{m}: 格子太少，跳过"); continue
        for seen, t, h, n, up in C:
            cell_rows.append([m, up[1], seen, TIER_NAME[t], h, n,
                              f"{(h/n)/FRAC:.4f}"])
        base = fit([(s, t, h, n) for s, t, h, n, _ in C])
        if base is None:
            continue
        ups = sorted({u for *_, u in C})
        by_up = collections.defaultdict(list)
        for s, t, h, n, u in C:
            by_up[u].append((s, t, h, n))
        bs = []
        for _ in range(args.boot):
            pick = rng.choice(len(ups), len(ups), replace=True)
            rr = [x for i in pick for x in by_up[ups[i]]]
            f = fit(rr)
            if f is not None:
                bs.append(f[2])
        bs = np.array(bs)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        # 双侧 p：bootstrap 分布中越过 0 的比例 ×2
        p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
        sign = "无交互" if lo <= 0 <= hi else ("放大" if base[2] > 0 else "削弱")
        print("%-26s %6d %7d %10.3f %10.3f %12.4f  [%+.3f, %+.3f] %7.4f"
              % (m, len(C), len(ups), base[0], base[1], base[2], lo, hi, p))
        rows.append([m, MODELS[m][0], len(C), len(ups),
                     f"{iqr[m][2]-iqr[m][0]:.4f}",
                     f"{base[0]:.4f}", f"{base[1]:.4f}", f"{base[2]:.4f}",
                     f"{lo:.4f}", f"{hi:.4f}", f"{p:.5f}", sign])
    print("-" * 108)

    # ---- 合并拟合 ----
    # 七个模型单独看都不显著，但 β 的符号如果一致，可能只是单模型功效不足。
    # 合并成一个回归（模型作固定效应），靶点级 bootstrap——同一个靶点在所有模型
    # 里的记录一起重抽，这样模型之间共享靶点造成的相关性被正确吸收。
    pooled = [(s_, t_, h_, n_, u_, mi)
              for mi, m in enumerate(MODELS) for s_, t_, h_, n_, u_ in cells[m]]
    if pooled:
        nm = len(MODELS)

        def fit_pooled(rr):
            X, yv, w = [], [], []
            for s_, t_, h_, n_, _u, mi in rr:
                if n_ <= 0:
                    continue
                dummies = [1.0 if j == mi else 0.0 for j in range(1, nm)]
                f = [s_, t_, s_ * t_] + dummies
                if h_ > 0:
                    X.append(f); yv.append(1); w.append(h_)
                if n_ - h_ > 0:
                    X.append(f); yv.append(0); w.append(n_ - h_)
            if len({v for v in yv}) < 2:
                return None
            mm = LogisticRegression(penalty=None, max_iter=3000, solver="lbfgs")
            mm.fit(np.array(X, float), np.array(yv), sample_weight=np.array(w, float))
            return mm.coef_[0]

        base = fit_pooled(pooled)
        ups = sorted({u for *_, u, _ in pooled})
        by_up = collections.defaultdict(list)
        for rec in pooled:
            by_up[rec[4]].append(rec)
        bs = []
        for _ in range(args.boot):
            pick = rng.choice(len(ups), len(ups), replace=True)
            rr = [x for i in pick for x in by_up[ups[i]]]
            f = fit_pooled(rr)
            if f is not None:
                bs.append(f[2])
        bs = np.array(bs)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        pp = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
        neg = sum(1 for r in rows[1:] if float(r[7]) < 0)
        print("\n合并拟合（七个模型 + 模型固定效应，靶点级 bootstrap）")
        print("  β靶点 %+.3f   β配体档 %+.3f   **β交互 %+.4f**  95%% CI [%+.3f, %+.3f]  p=%.4f"
              % (base[0], base[1], base[2], lo, hi, pp))
        print(f"  逐模型 β交互 为负的：{neg}/{len(rows)-1}")
        print("  （逐模型符号一致不能当独立检验——七个模型跑的是同一批靶点，"
              "彼此高度相关。合并拟合才是正确的合并方式。）")
        rows.append(["POOLED", "all", len(pooled), len(ups), "",
                     f"{base[0]:.4f}", f"{base[1]:.4f}", f"{base[2]:.4f}",
                     f"{lo:.4f}", f"{hi:.4f}", f"{pp:.5f}",
                     "无交互" if lo <= 0 <= hi else ("放大" if base[2] > 0 else "削弱")])

    print("\nβ交互 = 0（CI 含 0）：靶点新颖度和配体新颖度是**可加的**，各自独立起作用。")
    print("β交互 > 0：靶点见过时，配体新颖度的收益被放大（两者互相加强）。")
    print("β交互 < 0：靶点见过时，配体新颖度的收益反而变小。")
    print("\n注意 β 在 logit 尺度上，不能直接读成 EF 的倍数；符号和显著性是重点。")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    with open(args.cells_out, "w", newline="") as f:
        csv.writer(f).writerows(cell_rows)
    print(f"\n写入 {args.out}")
    print(f"写入 {args.cells_out}（逐靶点逐档的原始格子，可复现回归）")


if __name__ == "__main__":
    main()
