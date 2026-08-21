"""串联 rerank 的评测：检索粗筛 top-N 之后，物理重排到底有没有用。

比什么
------
在**同一个 top-N 子集内部**比三种排序：
  A 检索原序        —— 就是模型自己的打分顺序（基线）
  B Boltz-2 重排    —— 只按物理分数排
  C 排名融合        —— 两者名次平均（最省事、也最像实践里的做法）

⚠️ 只能在子集内部比。粗筛已经把 active 比例从约 2% 抬到 27%，
这里的 precision@k 不能和全库 EF 放在一起说。

看哪些指标
----------
· precision@5 / @10  —— 送去做实验的前几个里有几个真的是 active，最贴近实际决策
· active 平均名次    —— 整体是否被推上去了，不只看头部
· 子集内 AUROC       —— 与 k 的选择无关
配对检验按**靶点**做（分析单位是靶点，不是分子）。

已知的偏差来源
--------------
Boltz-2 亲和力模块训练时配体上限 56 个重原子，超过会不准。
这批里有一部分超限，脚本会单独把「全部 ≤56 重原子」的子集再算一遍对照。
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from rdkit import Chem, RDLogger
from scipy import stats

RDLogger.DisableLog("rdApp.*")
B = "/data/work/vs-benchmark"
HEAVY_CAP = 56


def load_affinity():
    out = {}
    for p in glob.glob(f"{B}/boltz_rerank_out/shard_*/*/predictions/*/affinity_*.json"):
        name = os.path.basename(p)[len("affinity_"):-len(".json")]
        try:
            out[name] = json.load(open(p))["affinity_pred_value"]
        except Exception:
            pass
    return out


def prec_at_k(labels_in_order, k):
    k = min(k, len(labels_in_order))
    return float(np.sum(labels_in_order[:k])) / k if k else float("nan")


def summarize(per_target, tag):
    """per_target: uniprot -> dict(方法 -> (p5, p10, meanrank, auroc))"""
    methods = ["检索原序", "Boltz 重排", "排名融合"]
    print(f"\n{tag}（{len(per_target)} 个靶点）")
    print("-" * 74)
    print("%-12s %10s %10s %12s %10s" % ("排序方式", "P@5", "P@10", "active 平均名次", "子集 AUROC"))
    for m in methods:
        v = np.array([per_target[u][m] for u in per_target if m in per_target[u]])
        if not len(v):
            continue
        print("%-12s %10.3f %10.3f %12.1f %10.3f" %
              (m, v[:, 0].mean(), v[:, 1].mean(), v[:, 2].mean(), v[:, 3].mean()))
    # 配对检验：重排 / 融合 相对基线
    base = np.array([per_target[u]["检索原序"] for u in per_target])
    for m in methods[1:]:
        alt = np.array([per_target[u][m] for u in per_target])
        line = []
        for j, nm in enumerate(["P@5", "P@10", "平均名次", "AUROC"]):
            a, b = base[:, j], alt[:, j]
            if np.allclose(a, b):
                line.append(f"{nm} 无变化")
                continue
            try:
                p = stats.wilcoxon(a, b).pvalue
            except ValueError:
                p = float("nan")
            d = b.mean() - a.mean()
            better = (d < 0) if nm == "平均名次" else (d > 0)
            line.append(f"{nm} {d:+.3f}{'↑' if better else '↓'} p={p:.3f}")
        print(f"  {m} vs 检索原序: " + " | ".join(line))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, nargs="+", default=[5, 10])
    args = ap.parse_args()

    man = json.load(open(f"{B}/data/t3/rerank_manifest.json"))
    aff = load_affinity()
    print(f"粗筛模型: {man['model']}   层: {man['layer']}   top-{man['topn']}")
    print(f"复合物 {len(man['entries'])}，已出亲和力 {len(aff)}"
          f"（{len(aff)/max(len(man['entries']),1)*100:.1f}%）")

    by_t = defaultdict(list)
    for e in man["entries"]:
        by_t[e["uniprot"]].append(e)

    per_all, per_small, skipped = {}, {}, []
    for up, items in by_t.items():
        items = [e for e in items if e["name"] in aff]
        n_act = sum(e["label"] for e in items)
        # 两类都要有：全是 active 的子集算不出 AUROC（早先版本会得到 nan）
        if len(items) < 10 or n_act < 2 or (len(items) - n_act) < 2:
            skipped.append(up)
            continue
        lab = np.array([e["label"] for e in items])
        ret = np.array([e["pred"] for e in items])          # 检索分数，越大越好
        bz = -np.array([aff[e["name"]] for e in items])     # 取负同向：越大越好
        # 排名融合：各自转成名次（1 最好）再平均
        r1 = stats.rankdata(-ret); r2 = stats.rankdata(-bz)
        fus = -(r1 + r2) / 2

        def metrics(score):
            order = np.argsort(-score)
            lo = lab[order]
            ranks = np.where(lo == 1)[0] + 1
            auc = stats.mannwhitneyu(score[lab == 1], score[lab == 0],
                                     alternative="greater").statistic / \
                (max((lab == 1).sum() * (lab == 0).sum(), 1))
            return (prec_at_k(lo, args.topk[0]), prec_at_k(lo, args.topk[1]),
                    float(ranks.mean()), float(auc))

        d = {"检索原序": metrics(ret), "Boltz 重排": metrics(bz), "排名融合": metrics(fus)}
        per_all[up] = d

        # 只留 ≤56 重原子的对照（Boltz 亲和力模块的训练上限）
        keep = []
        for e in items:
            m = Chem.MolFromSmiles(e["smi"])
            keep.append(m is not None and m.GetNumHeavyAtoms() <= HEAVY_CAP)
        keep = np.array(keep)
        if keep.sum() >= 10 and lab[keep].sum() >= 2 and (lab[keep] == 0).sum() >= 2:
            lab2, ret2, bz2 = lab[keep], ret[keep], bz[keep]
            r1 = stats.rankdata(-ret2); r2 = stats.rankdata(-bz2)
            fus2 = -(r1 + r2) / 2

            def metrics2(score):
                order = np.argsort(-score)
                lo = lab2[order]
                ranks = np.where(lo == 1)[0] + 1
                auc = stats.mannwhitneyu(score[lab2 == 1], score[lab2 == 0],
                                         alternative="greater").statistic / \
                    (max((lab2 == 1).sum() * (lab2 == 0).sum(), 1))
                return (prec_at_k(lo, args.topk[0]), prec_at_k(lo, args.topk[1]),
                        float(ranks.mean()), float(auc))
            per_small[up] = {"检索原序": metrics2(ret2), "Boltz 重排": metrics2(bz2),
                             "排名融合": metrics2(fus2)}

    print("=" * 74)
    summarize(per_all, "全部配体")
    if per_small:
        summarize(per_small, f"仅 ≤{HEAVY_CAP} 重原子的配体（Boltz 亲和力模块的训练范围内）")
    if skipped:
        print(f"\n跳过 {len(skipped)} 个靶点（结果不足或 active 太少）")

    print("\n怎么读")
    print("· P@5/P@10 升高且 p 显著 → 串联 rerank 在实践意义上有用")
    print("· 只有 AUROC 升、P@k 不升 → 物理分数整体有信息，但没把 active 推到最前面")
    print("· 融合优于单独重排 → 两类方法的错误模式不同，互补，这正是 T6 的假设")


if __name__ == "__main__":
    main()
