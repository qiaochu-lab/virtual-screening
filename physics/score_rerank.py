"""Evaluation of the cascade rerank: after retrieval's coarse top-N, does physics reranking actually help.

What is compared
-------------------
Three orderings compared **within the same top-N subset**:
  A retrieval's original order -- the model's own score order (baseline)
  B Boltz-2 rerank             -- ordered purely by the physics score
  C rank fusion                -- average of the two ranks (the simplest
                                   option, and the one closest to practice)

Warning: this can only be compared within the subset. Coarse retrieval has
already raised the active fraction from about 2% to 27%, so precision@k
here cannot be stated alongside the full-pool EF.

Which metrics to look at
---------------------------
. precision@5 / @10 -- of the first few sent for experimental follow-up, how
  many are really active; closest to the actual decision
. mean active rank -- whether actives are pushed up overall, not only at the
  very top
. within-subset AUROC -- independent of the choice of k
The paired test is done **per target** (the unit of analysis is the target,
not the molecule).

Known source of bias
------------------------
Boltz-2's affinity module was trained with a ligand cap of 56 heavy atoms;
beyond that it becomes inaccurate. Some ligands in this batch exceed that,
so the script separately recomputes the comparison on the subset where
"every ligand is <=56 heavy atoms".
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
    """per_target: uniprot -> dict(method -> (p5, p10, meanrank, auroc))"""
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
    # paired test: rerank / fusion against the baseline
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
        # both classes must be present: a subset that is all-active cannot yield an AUROC (an earlier version got nan here)
        if len(items) < 10 or n_act < 2 or (len(items) - n_act) < 2:
            skipped.append(up)
            continue
        lab = np.array([e["label"] for e in items])
        ret = np.array([e["pred"] for e in items])          # retrieval score, higher is better
        bz = -np.array([aff[e["name"]] for e in items])     # sign flipped to align direction: higher is better
        # rank fusion: convert each to rank (1 = best), then average
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

        # keep only the <=56 heavy-atom control set (Boltz's affinity module training cap)
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
