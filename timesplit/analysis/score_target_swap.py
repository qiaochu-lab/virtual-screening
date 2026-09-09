"""Target swap 的结果汇总：换掉靶点身份之后，模型还剩多少区分力。

配对关系
--------
swap 结果的目录名是**替身靶点 T'**，因为模型的蛋白序列按目录名查
（build_target_swap.py 的注释解释了为什么必须这样搭）。所以配对是：

    正确： results/t3_raw/{模型}/T3/{层}/{T}        T 的口袋 + T 的配体池
    swap： results/t3_raw/swap_{模型}_r{N}/T3/{层}/{T'}   T' 的口袋 + T 的配体池

两边打的是同一批分子，唯一变量是靶点身份。

⚠️ 必须校验标签一致
------------------
试跑时我按 T 去查 swap 目录，取到的是「T 的口袋 + 别人的配体」，
两边分子数完全不同却没被发现，差点把无效比较当成结果报出去。
所以这里对每一对都比标签数组，不一致就剔除并报出来。

多轮
----
random swap 每个靶点抽多个替身（round1/2/3），逐靶点取各轮中位数再做配对检验，
压掉单次抽样的运气。
"""
import argparse, csv, json, os, sys
import numpy as np

B = "/data/work/vs-benchmark"
LAYERS = ["L1", "L4"]
METRICS = ["ef1", "bedroc", "auroc"]


def find_dir(base_candidates, layer, target):
    """结果目录的布局不统一，按候选依次找。

    口袋类模型（UniMol / LigUnity / HypSeek）写 results/t3_raw/{m}/T3/{层}/{靶点}；
    序列类模型（ConPLex / ConGLUDe / SPRINT）少一层 T3，而且基线在 results/t3/ 下。
    与其在调用处到处写 if，不如在这里一次性把候选路径列全。
    """
    for b in base_candidates:
        for mid in (f"/T3/{layer}/{target}", f"/{layer}/{target}"):
            d = b + mid
            if os.path.isdir(d):
                return d
    return None


def load(d):
    try:
        p = np.load(f"{d}/saved_preds.npy").reshape(-1)
        y = np.load(f"{d}/saved_labels.npy").reshape(-1)
    except Exception:
        return None, None
    if len(p) != len(y) or y.sum() == 0 or y.sum() == len(y):
        return None, None
    return p, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--manifest", default=f"{B}/data/T3_swap_full/swap_manifest.json")
    ap.add_argument("--raw", default=f"{B}/results/t3_raw")
    ap.add_argument("--metrics-dir", default=f"{B}/eval")
    ap.add_argument("--out", default=f"{B}/results/export/T3_target_swap.csv")
    args = ap.parse_args()

    sys.path.insert(0, args.metrics_dir)
    from metrics import enrichment_factor, bedroc, roc_auc
    from scipy import stats

    man = json.load(open(args.manifest))
    rows = [["model", "layer", "n_targets", "metric",
             "correct_mean", "swap_mean", "delta", "delta_pct", "p_wilcoxon"]]

    print("Target swap：候选配体池不变，只换靶点身份")
    print("=" * 96)
    for m in args.models:
        printed = False
        for L in LAYERS:
            # 逐靶点收集各轮的 swap 指标
            per = {}          # 原靶点 -> {"correct": (...), "swap": [各轮]}
            skipped = 0
            for rnd in range(1, args.rounds + 1):
                key = f"round{rnd}/{L}"
                if key not in man:
                    continue
                for pr in man[key]["pairs"]:
                    o, s = pr["ligand_pool_from"], pr["identity_from"]
                    base = os.path.dirname(args.raw)   # .../results
                    dc = find_dir([f"{args.raw}/{m}", f"{base}/t3/{m}"], L, o)
                    ds = find_dir([f"{args.raw}/swap_{m}_r{rnd}"], L, s)
                    if dc is None or ds is None:
                        continue
                    pc, yc = load(dc)
                    ps, ys = load(ds)
                    if pc is None or ps is None:
                        continue
                    if len(yc) != len(ys) or not np.array_equal(yc, ys):
                        skipped += 1
                        continue
                    e = per.setdefault(o, {"correct": None, "swap": []})
                    if e["correct"] is None:
                        e["correct"] = (enrichment_factor(pc, yc, 0.01),
                                        bedroc(pc, yc, 80.5), roc_auc(pc, yc))
                    e["swap"].append((enrichment_factor(ps, ys, 0.01),
                                      bedroc(ps, ys, 80.5), roc_auc(ps, ys)))
            per = {k: v for k, v in per.items() if v["correct"] and v["swap"]}
            if len(per) < 5:
                continue
            if not printed:
                print(f"\n{m}")
                print("-" * 96)
                printed = True

            A = np.array([v["correct"] for v in per.values()])
            Sw = np.array([np.median(v["swap"], axis=0) for v in per.values()])
            nr = np.median([len(v["swap"]) for v in per.values()])
            note = f"  ⚠️ {skipped} 对标签不一致已剔除" if skipped else ""
            print(f"  {L}  n={len(per)}  每靶点 {nr:.0f} 轮{note}")
            for i, nm in enumerate(METRICS):
                d = Sw[:, i] - A[:, i]
                try:
                    p = stats.wilcoxon(A[:, i], Sw[:, i]).pvalue
                except Exception:
                    p = float("nan")
                pct = 100 * d.mean() / A[:, i].mean() if A[:, i].mean() else float("nan")
                star = " *" if p < 0.05 else ""
                print("      %-8s 正确 %8.3f → swap %8.3f   Δ %+8.3f (%+6.1f%%)  p=%.4f%s"
                      % (nm, A[:, i].mean(), Sw[:, i].mean(), d.mean(), pct, p, star))
                rows.append([m, L, len(per), nm,
                             f"{A[:, i].mean():.4f}", f"{Sw[:, i].mean():.4f}",
                             f"{d.mean():.4f}", f"{pct:.1f}", f"{p:.4g}"])
            # 有信号的靶点单独看：正确口袋下 EF>0 的那批
            has = A[:, 0] > 0
            if has.sum() >= 3:
                lost = ((Sw[has, 0] == 0)).sum()
                print("      有富集的靶点 %d 个，swap 后掉到 0 的 %d 个；"
                      "AUROC %.3f → %.3f"
                      % (has.sum(), lost, A[has, 2].mean(), Sw[has, 2].mean()))
    print("\n" + "=" * 96)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"写入 {args.out}")


if __name__ == "__main__":
    main()
