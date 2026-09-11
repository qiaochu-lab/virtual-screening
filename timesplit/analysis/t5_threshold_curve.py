"""T5: full pocket-threshold sensitivity curve (4Å / 6Å / 8Å).

6Å is the convention the models were trained under. Tightening to 4Å
(0.58×) and loosening to 8Å (1.85×) together let us tell whether this is
"bigger is better" or "must match the training convention" — the two have
completely different practical implications.

⚠️ 10.8% of 8Å's pockets exceed --max-pocket-atoms 511 and get truncated;
this must be noted when reporting results, otherwise the truncation effect
gets misread as a pocket-size effect.
"""
import argparse
import json, os
import numpy as np

from _subset import add_subset_arg, load_subset

B = "/data/work/vs-benchmark"

_ap = argparse.ArgumentParser()
add_subset_arg(_ap)
_args = _ap.parse_args()
KEEP = load_subset(_args.subset)


def ef1(bucket, L):
    """Get the mean EF1% for a layer; if a subset is given, compute it on the fly from per_target instead of using the whole-layer summary value."""
    if not bucket:
        return None
    if KEEP is None:
        return bucket.get("ef1")
    pt = [x["ef1"] for x in bucket.get("per_target", [])
          if (L, x["uniprot"]) in KEEP]
    return float(np.mean(pt)) if pt else None

S = {}
for tag, f in [("6Å", "summary.json"), ("4Å", "summary_4a.json"), ("8Å", "summary_8a.json")]:
    p = f"{B}/results/t3/{f}"
    S[tag] = json.load(open(p)) if os.path.exists(p) else {}
    if not S[tag]:
        print(f"  （{tag} 结果尚未生成：{f}）")

MODELS = ["drugclip", "bindclip_randneg", "bindclip_hardneg"]
print("\nT5 口袋阈值敏感性（EF1%，6Å 为训练口径）")
print("=" * 78)
print("%-20s %-4s %9s %9s %9s %10s %10s" %
      ("模型", "层", "4Å", "6Å(基准)", "8Å", "4Å vs 6Å", "8Å vs 6Å"))
print("-" * 78)
for m in MODELS:
    for L in ["L1", "L2", "L3", "L4"]:
        a = S["4Å"].get(m + "_4a", {}).get(L)
        b = S["6Å"].get(m, {}).get(L)
        c = S["8Å"].get(m + "_8a", {}).get(L)
        av, bv, cv = ef1(a, L), ef1(b, L), ef1(c, L)
        if bv is None:
            continue
        f4 = f"{av:.2f}" if av is not None else "—"
        f8 = f"{cv:.2f}" if cv is not None else "—"
        d4 = f"{(av-bv)/bv*100:+.0f}%" if av is not None else "—"
        d8 = f"{(cv-bv)/bv*100:+.0f}%" if cv is not None else "—"
        print("%-20s %-4s %9s %9.2f %9s %10s %10s" % (m, L, f4, bv, f8, d4, d8))

print("\n" + "=" * 78)
print("关键判断：6Å 是不是最优？")
print("=" * 78)
n_best = {"4Å": 0, "6Å": 0, "8Å": 0}
for m in MODELS:
    for L in ["L1", "L2", "L3", "L4"]:
        v = {}
        for tag, x in (("4Å", ef1(S["4Å"].get(m+"_4a", {}).get(L), L)),
                       ("6Å", ef1(S["6Å"].get(m, {}).get(L), L)),
                       ("8Å", ef1(S["8Å"].get(m+"_8a", {}).get(L), L))):
            if x is not None:
                v[tag] = x
        if len(v) >= 2:
            n_best[max(v, key=v.get)] += 1
tot = sum(n_best.values())
for k, n in n_best.items():
    print(f"  {k} 最优的格子数: {n}/{tot}")
print("\n若 6Å 明显最优 → 支持「必须匹配训练口径」，而不是「口袋越大越好」")
print("若 8Å 更优     → 说明模型受益于更大上下文，训练口径未必最佳")
print("\n注：8Å 有 10.8% 的口袋超过 511 原子上限被截断，其结果偏保守")
