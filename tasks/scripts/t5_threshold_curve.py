"""T5：口袋阈值敏感性完整曲线（4Å / 6Å / 8Å）。

6Å 是模型训练时的口径。4Å 收紧到 0.58×、8Å 放宽到 1.85×，
一收一放才能看出这是"越大越好"还是"必须匹配训练口径"——
这两种情况的实践含义完全不同。

⚠️ 8Å 有 10.8% 的口袋超过 --max-pocket-atoms 511 会被截断，
报结果时要注明，否则会把截断效应误读成口袋效应。
"""
import json, os
import numpy as np
B = "/data/yicheng/xqc/vs-benchmark"

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
        if not b:
            continue
        f4 = f"{a['ef1']:.2f}" if a else "—"
        f8 = f"{c['ef1']:.2f}" if c else "—"
        d4 = f"{(a['ef1']-b['ef1'])/b['ef1']*100:+.0f}%" if a else "—"
        d8 = f"{(c['ef1']-b['ef1'])/b['ef1']*100:+.0f}%" if c else "—"
        print("%-20s %-4s %9s %9.2f %9s %10s %10s" % (m, L, f4, b["ef1"], f8, d4, d8))

print("\n" + "=" * 78)
print("关键判断：6Å 是不是最优？")
print("=" * 78)
n_best = {"4Å": 0, "6Å": 0, "8Å": 0}
for m in MODELS:
    for L in ["L1", "L2", "L3", "L4"]:
        v = {}
        if S["4Å"].get(m+"_4a", {}).get(L): v["4Å"] = S["4Å"][m+"_4a"][L]["ef1"]
        if S["6Å"].get(m, {}).get(L):       v["6Å"] = S["6Å"][m][L]["ef1"]
        if S["8Å"].get(m+"_8a", {}).get(L): v["8Å"] = S["8Å"][m+"_8a"][L]["ef1"]
        if len(v) >= 2:
            n_best[max(v, key=v.get)] += 1
tot = sum(n_best.values())
for k, n in n_best.items():
    print(f"  {k} 最优的格子数: {n}/{tot}")
print("\n若 6Å 明显最优 → 支持「必须匹配训练口径」，而不是「口袋越大越好」")
print("若 8Å 更优     → 说明模型受益于更大上下文，训练口径未必最佳")
print("\n注：8Å 有 10.8% 的口袋超过 511 原子上限被截断，其结果偏保守")
