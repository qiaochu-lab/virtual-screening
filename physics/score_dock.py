"""Evaluation of the docking results: compare "retrieval's original order vs. docking rerank vs. fusion" within the same top-N shortlist.

Relationship to the three Boltz-2 rounds
--------------------------------------------
Same question, same set of metrics, but a different physics method and the
depth raised from 50 to 200 (recall@50 on L4 is only 17.5%, recall@200 is
34.0% -- the ceiling doubles). If docking also shows no benefit, then
"cascade reranking does not work" is no longer a property of Boltz-2 alone;
if docking helps while Boltz-2 does not, that points at co-folding scoring
being the problem rather than the cascade idea itself.

Warning: scope limitation -- this is **pocket-slice docking**: the receptor
only has the atoms inside the 6A pocket, and residues at the slice's edge
lack their neighbor constraints, biasing absolute affinity. But this is only
used for the **relative ranking** of the same batch of molecules, where that
bias has limited effect.
"""
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy import stats

B = "/data/work/vs"


def parse_scores(path):
    """smina's table output -> [affinity], in the same order as the input SDF."""
    out = []
    for m in re.finditer(r"^\s*1\s+(-?\d+\.\d+)\s", open(path).read(), re.M):
        out.append(float(m.group(1)))
    return out


def prec_at_k(lab, k):
    k = min(k, len(lab))
    return float(np.sum(lab[:k])) / k if k else float("nan")


def main():
    man = json.load(open(f"{B}/dock/manifest.json"))
    print(f"粗筛模型 {man['model']}   层 {man['layer']}   深度 top-{man['topn']}")
    rows = ["target,n,n_ligands,coverage,n_actives,method,p_at_10,p_at_20,mean_active_rank,auroc"]
    per = {}
    covs = {}
    for up in man["targets"]:
        d = f"{B}/dock/{up}"
        sp = f"{d}/scores.txt"
        if not os.path.exists(sp):
            continue
        aff = parse_scores(sp)
        info = json.load(open(f"{d}/manifest.json"))["ligands"]
        n = min(len(aff), len(info))
        cov = n / len(info) if info else 0.0
        if n < 20:
            continue
        lab = np.array([info[i]["label"] for i in range(n)])
        if lab.sum() < 2 or (lab == 0).sum() < 2:
            continue
        ret = np.array([info[i]["retrieval_score"] for i in range(n)])
        dock = -np.array(aff[:n])          # more negative affinity is better; flip sign to align direction
        r1, r2 = stats.rankdata(-ret), stats.rankdata(-dock)
        fus = -(r1 + r2) / 2

        def metr(sc):
            o = np.argsort(-sc)
            lo = lab[o]
            ranks = np.where(lo == 1)[0] + 1
            auc = stats.mannwhitneyu(sc[lab == 1], sc[lab == 0],
                                     alternative="greater").statistic / \
                ((lab == 1).sum() * (lab == 0).sum())
            return prec_at_k(lo, 10), prec_at_k(lo, 20), float(ranks.mean()), float(auc)

        covs[up] = (cov, len(info))
        per[up] = {"检索原序": metr(ret), "对接重排": metr(dock), "排名融合": metr(fus)}
        for name, key in (("retrieval", "检索原序"), ("smina_rerank", "对接重排"),
                          ("rank_fusion", "排名融合")):
            v = per[up][key]
            rows.append(f"{up},{n},{len(info)},{cov:.3f},{int(lab.sum())},{name},"
                        f"{v[0]:.3f},{v[1]:.3f},{v[2]:.2f},{v[3]:.4f}")

    if not per:
        print("还没有可用结果")
        return
    part = [u for u in per if covs[u][0] < 0.99]
    if part:
        print("\n⚠ 以下靶点因超时只跑完一部分配体，先跑完的配体系统性偏小/偏易，")
        print("  其 EF 不可与完整靶点直接混算：")
        for u in part:
            c, tot = covs[u]
            print(f"    {u}  {int(c*tot)}/{tot}  ({c:.0%})")
    full = [u for u in per if covs[u][0] >= 0.99]
    print(f"\n可用靶点 {len(per)}（其中完整 {len(full)}）")
    def table(keys, title):
        if len(keys) < 3:
            print(f"\n[{title}] 靶点不足，跳过")
            return
        print(f"\n[{title}]  n={len(keys)}")
        print("%-12s %8s %8s %14s %10s" % ("排序方式", "P@10", "P@20", "active 平均名次", "AUROC"))
        print("-" * 58)
        base = np.array([per[u]["检索原序"] for u in keys])
        for key in ("检索原序", "对接重排", "排名融合"):
            v = np.array([per[u][key] for u in keys])
            print("%-12s %8.3f %8.3f %14.1f %10.3f" %
                  (key, v[:, 0].mean(), v[:, 1].mean(), v[:, 2].mean(), v[:, 3].mean()))
            if key != "检索原序":
                line = []
                for j, nm in enumerate(("P@10", "P@20", "平均名次", "AUROC")):
                    a, b = base[:, j], v[:, j]
                    try:
                        p = stats.wilcoxon(a, b).pvalue
                    except ValueError:
                        p = float("nan")
                    line.append(f"{nm} {b.mean()-a.mean():+.3f} p={p:.3f}")
                print("      vs 基线: " + " | ".join(line))
    table(list(per), "全部靶点")
    table(full, "仅完整靶点")
    out = f"{B}/results/export/T6_dock.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write("\n".join(rows) + "\n")
    print(f"\n逐靶点写入 {out}")


if __name__ == "__main__":
    main()
