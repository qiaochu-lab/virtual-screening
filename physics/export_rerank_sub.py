"""Boltz-2 重排在 350 子集 L4 靶点上的结果（T6-RE 的 Boltz 那半）。

和前四轮的两个关键区别，读结果时必须带着：

1. **靶点全部来自最终的 350 子集**。前四轮跑在全量 L4 上，12 个靶点里只有 5 个
   落在子集里，所以那几轮的数不能直接当主结论。

2. **召回按构造 = 100%**（`prep_rerank.py --inject-actives`）。不在 top-200 里的
   活性被补回了候选池。**这是为了让实验有意义**：子集上 L4 的 recall@200 只有
   22.6%（top-50 更只有 9.4%），重排一个近八成活性都不在里面的列表，
   无论物理方法多准都做不出什么——结果是预定的，而且分不清「物理重排没用」
   和「候选里没东西可捞」。

⚠️ **因此绝对指标不可与前四轮或全库 EF 比较。**
补回活性把候选池的活性占比抬到了约 48%，P@5 / P@10 的随机基线也跟着抬到 0.48。
唯一可比的是**同一批分子上「检索原序」和「Boltz 重排」的配对差**——
这正是要测的量。

三个排序一起报：
  · retrieval    检索模型的原始排序（对照组）
  · boltz        纯 Boltz-2 亲和力打分
  · rank_fusion  两者名次相加（前几轮里唯一偶尔赢过检索的组合）
"""
import glob
import json
import os

import numpy as np
from scipy import stats

B = "/data/work/vs-benchmark"
# prep_rerank.py 的 manifest 路径是写死的，不跟随 --out
MAN = f"{B}/data/t3/rerank_manifest.json"


def load(out_root):
    aff = {}
    for p in glob.glob(f"{out_root}/shard_*/*/predictions/*/affinity_*.json"):
        n = os.path.basename(p)[9:-5]
        try:
            aff[n] = json.load(open(p))["affinity_pred_value"]
        except Exception:
            pass
    return aff


def metrics(lab, sc):
    """P@5 / P@10 / 活性平均名次 / AUROC。分数越大越靠前。"""
    o = np.argsort(-sc)
    lo = lab[o]
    ranks = np.where(lo == 1)[0] + 1
    na, nd = int((lab == 1).sum()), int((lab == 0).sum())
    auc = (stats.mannwhitneyu(sc[lab == 1], sc[lab == 0],
                              alternative="greater").statistic / (na * nd)
           if na and nd else float("nan"))
    return float(lo[:5].mean()), float(lo[:10].mean()), float(ranks.mean()), float(auc)


def main():
    man = json.load(open(MAN))
    aff = load(f"{B}/boltz_rerank_sub_out")
    print(f"Boltz-2 出分 {len(aff):,} / {len(man['entries']):,}")

    by = {}
    for e in man["entries"]:
        by.setdefault(e["uniprot"], []).append(e)

    rows = ["target,n_shortlist,n_actives,frac_active,method,"
            "p_at_5,p_at_10,mean_active_rank,auroc"]
    per = {}
    for up, items in sorted(by.items()):
        items = [e for e in items if e["name"] in aff]
        if len(items) < 20:
            continue
        lab = np.array([e["label"] for e in items], dtype=int)
        if lab.sum() < 3 or lab.sum() == len(lab):
            continue
        # 检索分数越大越好；Boltz 的 affinity_pred_value 越小越好（预测的 log Kd）
        # "pred" 就是检索模型给这个分子的分数
        ret = np.array([e["pred"] for e in items], dtype=float)
        bol = -np.array([aff[e["name"]] for e in items], dtype=float)
        fus = -(stats.rankdata(-ret) + stats.rankdata(-bol))
        for name, sc in (("retrieval", ret), ("boltz", bol), ("rank_fusion", fus)):
            m = metrics(lab, sc)
            per.setdefault(name, []).append(m)
            rows.append("%s,%d,%d,%.3f,%s,%.3f,%.3f,%.2f,%.4f"
                        % (up, len(items), int(lab.sum()), lab.mean(), name, *m))

    n_t = len(per.get("retrieval", []))
    print(f"\n可评的靶点 {n_t}")
    print("%-14s %8s %8s %14s %8s" % ("排序", "P@5", "P@10", "活性平均名次", "AUROC"))
    print("-" * 58)
    for name in ("retrieval", "boltz", "rank_fusion"):
        v = np.array(per.get(name, []))
        if len(v):
            print("%-14s %8.3f %8.3f %14.1f %8.4f"
                  % (name, v[:, 0].mean(), v[:, 1].mean(), v[:, 2].mean(), v[:, 3].mean()))
    print("-" * 58)

    if n_t >= 5:
        print("\n逐靶点配对（Wilcoxon signed-rank，对 retrieval）")
        R = np.array(per["retrieval"])
        for name in ("boltz", "rank_fusion"):
            X = np.array(per[name])
            print(f"  {name}")
            for i, lab in enumerate(("P@5", "P@10", "活性平均名次", "AUROC")):
                a, b = R[:, i], X[:, i]
                if np.allclose(a, b):
                    print(f"    {lab:12} 完全相同"); continue
                p = stats.wilcoxon(a, b).pvalue
                # 名次是越小越好，其余越大越好
                better = (b < a).sum() if i == 2 else (b > a).sum()
                print(f"    {lab:12} {a.mean():+.3f} → {b.mean():+.3f}   "
                      f"赢 {better}/{len(a)}   p={p:.4f}")

    out = f"{B}/results/export/T6_rerank_subset.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write("\n".join(rows) + "\n")
    print(f"\n写入 {out}")
    print("\n⚠️ 绝对指标不可与前四轮或全库 EF 比：补回活性后候选池的活性占比约 48%，")
    print("   P@5/P@10 的随机基线也是 0.48。可比的只有同一批分子上的配对差。")


if __name__ == "__main__":
    main()
