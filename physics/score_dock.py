"""对接结果的评测：在同一个 top-N shortlist 内比「检索原序 vs 对接重排 vs 融合」。

与 Boltz-2 那三轮的关系
-----------------------
同一个问题、同一套指标，但换了物理方法并把深度从 50 加到 200
（recall@50 在 L4 只有 17.5%，@200 是 34.0%，天花板翻倍）。
如果对接也无收益，那「串联 rerank 不奏效」就不再是 Boltz-2 一家的性质；
如果对接有效而 Boltz-2 无效，说明问题出在共折叠打分而非级联思路。

⚠️ 口径限制：这是**口袋切片对接**——受体只有 6Å 口袋内的原子，
切片边缘残基缺少邻接约束，绝对亲和力会有偏差。
但我们只用它做同一批分子的**相对排序**，这个偏差影响有限。
"""
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy import stats

B = "/data/yicheng/xqc/vs-benchmark"


def parse_scores(path):
    """smina 的表格输出 -> [affinity]，顺序与输入 SDF 一致。"""
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
        dock = -np.array(aff[:n])          # affinity 越负越好，取负同向
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
