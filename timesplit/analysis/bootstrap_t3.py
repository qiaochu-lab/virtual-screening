"""给 T3 主表补 bootstrap 置信区间。

为什么需要
----------
主表现在只有均值。L3 只有 43–49 个靶点，L4 也就 176–234 个，
「A 模型 8.83 vs B 模型 8.46」这种差距到底算不算差距，没有区间就判断不了。
交给别人 review 时，没有区间的均值很难引用。

口径
----
在**靶点层面**重采样（分析单位是靶点，不是分子）。
把各靶点的分子拼起来再重采样会低估方差，而且改变了 EF 的定义——
`eval/README.md` 里写过这一点，这里保持一致。
"""
import argparse
import json
import os

import numpy as np

B = "/data/work/vs-benchmark"


def ci(vals, n_boot=2000, seed=0):
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="ef1", choices=["ef1", "ef5", "bedroc", "auroc"])
    ap.add_argument("--summary", default=f"{B}/results/t3/summary.json")
    ap.add_argument("--out", default=f"{B}/results/export/T3_main_ci.csv")
    args = ap.parse_args()

    s = json.load(open(args.summary))
    rows = ["model,layer,n_targets,metric,mean,ci_lo,ci_hi"]
    print(f"T3 主表 · {args.metric} · 靶点层面 bootstrap（2000 次重采样，95% 区间）")
    print("=" * 72)
    print("%-26s %-4s %7s %10s %22s" % ("模型", "层", "靶点", "均值", "95% 置信区间"))
    print("-" * 72)
    for m in sorted(s):
        for L in ["L1", "L2", "L3", "L4"]:
            d = s[m].get(L)
            if not d or "per_target" not in d:
                continue
            vals = [t.get(args.metric) for t in d["per_target"] if t.get(args.metric) is not None]
            mean, lo, hi = ci(vals)
            if not np.isfinite(mean):
                continue
            print("%-26s %-4s %7d %10.2f %22s" %
                  (m, L, len(vals), mean, f"[{lo:.2f}, {hi:.2f}]"))
            rows.append(f"{m},{L},{len(vals)},{args.metric},{mean:.4f},{lo:.4f},{hi:.4f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(rows) + "\n")
    print("-" * 72)
    print(f"写入 {args.out}")
    print("读法：两个模型的区间明显不重叠才算真的有差距；"
          "L3 靶点少，区间会宽到很多比较都下不了结论。")


if __name__ == "__main__":
    main()
