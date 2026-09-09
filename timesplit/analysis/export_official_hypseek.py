"""导出官方 HypSeek 权重的评测结果，与我们原有的数字并排。

背景：此前所有 HypSeek 数字用的都是官方 _rk（当时只有它公开）。合作者
2026-09-07 指出 _rk 是按 benchmark 选出的 checkpoint，用它测 DUD-E/PCBA 有泄漏，
应当 _vs 跑虚筛、_rk 跑排序。issue #4 里作者同时给出两个权重。
同时 alpha_prot（蛋白序列通路权重）训练默认 1、评测默认 0，两档都测。

只导数字，不下结论——结论要人看过再改。
"""
import csv, json, os, sys
import numpy as np

B = "/data/work/vs-benchmark"
sys.path.insert(0, f"{B}/eval")
from metrics import bedroc, enrichment_factor, roc_auc

BENCH = ["DUDE", "DEKOIS", "PCBA"]
# (标签, 结果目录) —— 目录名由 run_official_hypseek.sh 决定
CAND = [
    ("hypseek_rk (我们原用)",            f"{B}/results/hypseek_rk"),
    ("hypseek_official_vs a=0",          f"{B}/results/hypseek_official_vs"),
    ("hypseek_official_vs a=1",          f"{B}/results/hypseek_official_vs_a1"),
    ("hypseek_official_rk a=0",          f"{B}/results/hypseek_official_rk"),
    ("hypseek_official_rk a=1",          f"{B}/results/hypseek_official_rk_a1"),
    ("hypseek_vs_collab (合作者训)",     f"{B}/results/hypseek_vs_collab"),
    ("hypseek_vs_s1 (我们训)",           f"{B}/results/hypseek_vs_seed1"),
]


def score_target(d):
    try:
        p = np.load(f"{d}/saved_preds.npy").reshape(-1)
        y = np.load(f"{d}/saved_labels.npy").reshape(-1)
    except Exception:
        return None
    if len(p) != len(y) or y.sum() == 0 or y.sum() == len(y):
        return None
    return dict(ef1=enrichment_factor(p, y, 0.01),
                ef5=enrichment_factor(p, y, 0.05),
                bedroc=bedroc(p, y, 80.5), auroc=roc_auc(p, y))


def main():
    rows = [["variant", "benchmark", "n_targets", "ef1", "ef5", "bedroc", "auroc"]]
    print("%-30s %-8s %7s %9s %9s %9s %9s" %
          ("权重 / alpha_prot", "基准", "靶点", "EF1%", "EF5%", "BEDROC", "AUROC"))
    print("-" * 88)
    seen_any = False
    for tag, root in CAND:
        for b in BENCH:
            d = f"{root}/{b}"
            if not os.path.isdir(d):
                continue
            vals = [v for up in sorted(os.listdir(d))
                    if (v := score_target(f"{d}/{up}")) is not None]
            if not vals:
                continue
            seen_any = True
            m = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
            print("%-30s %-8s %7d %9.2f %9.2f %9.4f %9.4f" %
                  (tag, b, len(vals), m["ef1"], m["ef5"], m["bedroc"], m["auroc"]))
            rows.append([tag, b, len(vals)] +
                        [f"{m[k]:.4f}" for k in ("ef1", "ef5", "bedroc", "auroc")])
        print()
    if not seen_any:
        print("⚠️ 没有找到任何结果目录，评测可能还没跑完")
        return 1

    print("\n参照（GitHub issue #4，作者与复现者的数字）")
    print("  论文 / 作者 _vs (alpha_prot=0)  DUD-E EF1% = 51.44 / 51.43")
    print("  作者 _vs (alpha_prot=1)         DUD-E EF1% = 53.04")
    print("  复现者自训 _vs                  DUD-E EF1% = 46.05")

    out = f"{B}/results/export/T1_hypseek_official.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\n写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
