"""rerank4（diffusion_samples=5）对 rerank3（=1）的配对比较。

两轮用的是同一批 yaml（boltz_rerank3/），同一个 manifest，同样的 MSA，
亲和力阶段同样是默认的 5 个扩散样本。唯一的差别是结构阶段采样几次。

为什么这个差别可能重要：Boltz 结构阶段采样 N 个复合物后按 confidence 排名，
只有 rank-0 会写成 pre_affinity_*.npz 交给亲和力模型
（boltz/data/write/writer.py:177）。N=1 等于没有筛选，把一次未经挑选的
随机抽样直接交给打分模型；N=5 至少有个 best-of-5。

输出 results/export/T6_rerank4.csv，并打印逐靶点配对检验。
"""
import glob, json, os
import numpy as np
from scipy import stats
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
B = "/data/work/vs"
man = json.load(open(f"{B}/data/t3/rerank3_manifest.json"))


def load(out_root):
    aff = {}
    for p in glob.glob(f"{out_root}/shard_*/*/predictions/*/affinity_*.json"):
        n = os.path.basename(p)[9:-5]
        try:
            aff[n] = json.load(open(p))["affinity_pred_value"]
        except Exception:
            pass
    return aff


a3 = load(f"{B}/boltz_rerank3_out")
a4 = load(f"{B}/boltz_rerank4_out")
print(f"rerank3 (N=1) 出分 {len(a3)}   rerank4 (N=5) 出分 {len(a4)}")
both = set(a3) & set(a4)
print(f"两轮都有分的复合物 {len(both)}")

by = {}
for e in man["entries"]:
    by.setdefault(e["uniprot"], []).append(e)


def metrics(lab, sc):
    o = np.argsort(-sc)
    lo = lab[o]
    ranks = np.where(lo == 1)[0] + 1
    auc = stats.mannwhitneyu(sc[lab == 1], sc[lab == 0],
                             alternative="greater").statistic / \
        ((lab == 1).sum() * (lab == 0).sum())
    return lo[:5].mean(), lo[:10].mean(), float(ranks.mean()), float(auc)


rows = ["target,n_shortlist,n_actives,method,p_at_5,p_at_10,mean_active_rank,auroc"]
per = {}
for up, items in by.items():
    # 只保留两轮都有分的复合物，保证是严格配对
    items = [e for e in items if e["name"] in both]
    if len(items) < 10:
        continue
    lab = np.array([e["label"] for e in items])
    if lab.sum() < 2 or (lab == 0).sum() < 2:
        continue
    ret = np.array([e["pred"] for e in items])
    b3 = -np.array([a3[e["name"]] for e in items])
    b4 = -np.array([a4[e["name"]] for e in items])
    r1 = stats.rankdata(-ret)
    fus3 = -(r1 + stats.rankdata(-b3)) / 2
    fus4 = -(r1 + stats.rankdata(-b4)) / 2
    per[up] = {"retrieval": metrics(lab, ret),
               "boltz_N1": metrics(lab, b3), "boltz_N5": metrics(lab, b4),
               "fusion_N1": metrics(lab, fus3), "fusion_N5": metrics(lab, fus4)}
    for nm in per[up]:
        v = per[up][nm]
        rows.append(f"{up},{len(items)},{int(lab.sum())},{nm},"
                    f"{v[0]:.3f},{v[1]:.3f},{v[2]:.2f},{v[3]:.4f}")

os.makedirs(f"{B}/results/export", exist_ok=True)
open(f"{B}/results/export/T6_rerank4.csv", "w").write("\n".join(rows) + "\n")

print(f"\n可用靶点 {len(per)}")
print("%-12s %8s %8s %14s %10s" % ("排序方式", "P@5", "P@10", "active 平均名次", "AUROC"))
print("-" * 58)
for nm in ("retrieval", "boltz_N1", "boltz_N5", "fusion_N1", "fusion_N5"):
    v = np.array([per[u][nm] for u in per])
    print("%-12s %8.3f %8.3f %14.1f %10.3f" %
          (nm, v[:, 0].mean(), v[:, 1].mean(), v[:, 2].mean(), v[:, 3].mean()))

print("\n配对检验（逐靶点 Wilcoxon）")
print("-" * 58)
for a, b, label in (("boltz_N1", "boltz_N5", "多采样有没有改善 Boltz 重排"),
                    ("retrieval", "boltz_N5", "N=5 的 Boltz 重排 vs 检索基线"),
                    ("retrieval", "fusion_N5", "N=5 的排名融合 vs 检索基线")):
    line = []
    for j, mn in enumerate(("P@5", "P@10", "平均名次", "AUROC")):
        x = np.array([per[u][a][j] for u in per])
        y = np.array([per[u][b][j] for u in per])
        try:
            p = stats.wilcoxon(x, y).pvalue
        except ValueError:
            p = float("nan")
        line.append(f"{mn} {y.mean()-x.mean():+.3f} p={p:.3f}")
    print(f"{label}\n  " + " | ".join(line))

print(f"\n写入 {B}/results/export/T6_rerank4.csv")
