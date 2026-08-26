"""T5 apo 对照的配对比较：同一批靶点，holo 口袋 vs apo 口袋。

为什么这个对照重要
------------------
所有公开虚筛基准（DUD-E、DEKOIS、LIT-PCBA）和我们的 T3，用的都是
**holo 口袋**——从共晶复合物截出来的，侧链已经为配体让好了位。
真实虚筛拿到的常常是 apo 构象。如果模型在 apo 上明显变差，
说明现有基准（包括我们自己的）系统性高估了这类方法的实用表现。

口径
----
· 同一批靶点、同一批分子、同一套推理参数，**只有口袋构象不同**
· 按靶点配对做 Wilcoxon，分析单位是靶点
· 同时报叠合 RMSD 与 apo/holo 口袋原子数之比，用来看「差异是否随构象偏离增大」
"""
import json
import os

import numpy as np
import sys
from scipy import stats

B = "/data/work/vs-benchmark"
sys.path.insert(0, f"{B}/eval")
from metrics import bedroc, enrichment_factor, roc_auc  # noqa: E402

PAIRS = [("drugclip", "drugclip_apo"), ("bindclip_randneg", "bindclip_randneg_apo")]


def score_dir(d):
    out = {}
    if not os.path.isdir(d):
        return out
    for up in os.listdir(d):
        try:
            p = np.load(f"{d}/{up}/saved_preds.npy").reshape(-1)
            y = np.load(f"{d}/{up}/saved_labels.npy")
        except Exception:
            continue
        if len(p) != len(y) or y.sum() == 0 or y.sum() == len(y):
            continue
        out[up] = dict(ef1=enrichment_factor(p, y, 0.01),
                       bedroc=bedroc(p, y, 80.5), auroc=roc_auc(p, y))
    return out


def main():
    man = json.load(open(f"{B}/data/t3/apo_pocket_manifest.json"))
    print("T5 · apo 构象对照（同靶点配对，仅口袋构象不同）")
    print("=" * 78)
    print("%-22s %7s %10s %10s %10s %10s" %
          ("模型", "靶点", "holo EF1", "apo EF1", "holo AUROC", "apo AUROC"))
    print("-" * 78)
    rows = ["model,layer,uniprot,apo_pdb,holo_pdb,align_rmsd,metric,holo,apo"]
    for holo_m, apo_m in PAIRS:
        h, a = {}, {}
        for L in ["L3", "L4"]:
            h |= {f"{L}/{k}": v for k, v in score_dir(f"{B}/results/t3_raw/{holo_m}/T3/{L}").items()}
            a |= {f"{L}/{k}": v for k, v in score_dir(f"{B}/results/t3_raw/{apo_m}/T3/{L}").items()}
        common = sorted(set(h) & set(a))
        if len(common) < 5:
            print(f"{holo_m}: 共同靶点只有 {len(common)} 个，先等跑完")
            continue
        hv = {k: np.array([h[c][k] for c in common]) for k in ("ef1", "bedroc", "auroc")}
        av = {k: np.array([a[c][k] for c in common]) for k in ("ef1", "bedroc", "auroc")}
        print("%-22s %7d %10.2f %10.2f %10.4f %10.4f" %
              (holo_m, len(common), hv["ef1"].mean(), av["ef1"].mean(),
               hv["auroc"].mean(), av["auroc"].mean()))
        for k in ("ef1", "bedroc", "auroc"):
            try:
                p = stats.wilcoxon(hv[k], av[k]).pvalue
            except ValueError:
                p = float("nan")
            d = (av[k].mean() - hv[k].mean()) / max(hv[k].mean(), 1e-9) * 100
            print(f"      {k:7s} 变化 {d:+6.1f}%   配对 p={p:.4f}")
        for c in common:
            L, up = c.split("/")
            mm = man.get(up, {})
            for k in ("ef1", "auroc"):
                rows.append(f"{holo_m},{L},{up},{mm.get('apo_pdb','')},{mm.get('holo_pdb','')},"
                            f"{mm.get('align_rmsd','')},{k},{h[c][k]:.4f},{a[c][k]:.4f}")
        # 差异是否随构象偏离增大
        rmsd = np.array([man.get(c.split("/")[1], {}).get("align_rmsd", np.nan) for c in common])
        ok = ~np.isnan(rmsd)
        if ok.sum() >= 8:
            delta = av["auroc"][ok] - hv["auroc"][ok]
            r = stats.spearmanr(rmsd[ok], delta).statistic
            print(f"      AUROC 变化 vs 叠合 RMSD 的相关 ρ={r:+.3f}"
                  "（负值 = 构象偏离越大掉得越多）")
    out = f"{B}/results/export/T5_apo.csv"
    open(out, "w").write("\n".join(rows) + "\n")
    print(f"\n逐靶点写入 {out}")


if __name__ == "__main__":
    main()
