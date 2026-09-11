"""Export evaluation results for the official HypSeek weights, side by side
with our existing numbers.

Background: every HypSeek number so far used the official `_rk` (the only
one public at the time). On 2026-09-07 a collaborator pointed out that `_rk`
is a checkpoint selected against the benchmark, so using it to score
DUD-E/PCBA leaks; `_vs` should run screening and `_rk` should run ranking.
Issue #4 has the author releasing both weights together. Also test both
settings of `alpha_prot` (the protein-sequence pathway weight), which
defaults to 1 in training and 0 in evaluation.

Only exports numbers, draws no conclusions -- conclusions get revised only
after a human has looked at them.
"""
import csv, json, os, sys
import numpy as np

B = "/data/work/vs-benchmark"
sys.path.insert(0, f"{B}/eval")
from metrics import bedroc, enrichment_factor, roc_auc

BENCH = ["DUDE", "DEKOIS", "PCBA"]
# (label, results dir) -- the directory name is decided by run_official_hypseek.sh
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
