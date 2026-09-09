"""化学系列 oracle 上界（target-conditioned ligand-similarity oracle）。

⚠️ **文件名是历史遗留，这个量不是「纯配体基线」。** 它虽然不看蛋白结构，
却读了**该靶点的已知活性分子**——任何被评测的模型都拿不到这个信息。
所以它量的不是「不看蛋白能做多好」，而是「**已经知道什么能结合这个靶点时，
纯化学相似度能走多远**」，是个**上界**。

真正的纯配体基线在 `ligand_only_learned.py`：它连靶点是谁都不知道，
结果低于随机。两者差两个数量级，名字混用会把结论讲反。

原始描述：只用 2D 指纹相似度，能把 T3 的诱饵分开吗。

为什么要做
----------
Mattsson & Walters (bioRxiv 2026.06.29.735309) 指出，蛋白–配体亲和力基准普遍
被数据泄漏撑着，一个**不看蛋白的纯配体模型**在 FEP+ 上就能拿到 r=0.66。
如果一个虚筛基准能被纯配体信号解决，那它测的就不是「口袋–配体匹配」。

这个脚本给 T3 做同样的检查：把每个候选分子按「与该靶点已知活性的最大 Tanimoto」
打分（活性分子留一，不许拿自己），然后照常算 EF/BEDROC/AUROC。

注意这个基线比模型多拿了信息——它看得到该靶点的真实活性分子，而模型看不到。
所以它是**上界**：它测的是「这批活性分子在化学空间里有多聚集」，
也就是 analogous-series bias 有多严重，不是模型能达到什么。

怎么读结果
----------
· 接近随机（EF≈1、AUROC≈0.5）→ 诱饵设计成功，跨靶点真实活性确实不能靠
  「像不像药」区分，T3 的富集只能来自靶点特异性信息。
· 明显高于随机 → 这一层的活性分子自己就抱团，模型可能靠记化学系列就能得分，
  报 EF 时必须减掉这个底。
"""
import argparse
import csv
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")

B = "/data/work/vs-benchmark"
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fp(smi):
    m = Chem.MolFromSmiles(smi)
    return GEN.GetFingerprint(m) if m else None


def one_target(rec):
    """返回 (uniprot, layer, scores, labels)；活性分子对自己留一。"""
    af = [fp(a["smiles"]) for a in rec["actives"]]
    df = [fp(d["smiles"]) for d in rec["decoys"]]
    keep_a = [i for i, x in enumerate(af) if x is not None]
    keep_d = [i for i, x in enumerate(df) if x is not None]
    A = [af[i] for i in keep_a]
    D = [df[i] for i in keep_d]
    if len(A) < 2 or not D:
        return None
    s_a = []
    for i, x in enumerate(A):                       # 留一：排除自己
        others = A[:i] + A[i + 1:]
        s_a.append(max(DataStructs.BulkTanimotoSimilarity(x, others)))
    s_d = [max(DataStructs.BulkTanimotoSimilarity(x, A)) for x in D]
    scores = np.array(s_a + s_d, dtype=np.float64)
    labels = np.array([1] * len(s_a) + [0] * len(s_d), dtype=np.int8)
    return rec["uniprot"], rec["layer"], scores, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="+", default=["L1", "L2", "L3", "L4"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-dir", default=f"{B}/data/t3/eval")
    ap.add_argument("--metrics", default=f"{B}/eval")
    ap.add_argument("--out", default=f"{B}/results/export/T3_ligand_only.csv")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, args.metrics)
    from metrics import enrichment_factor as ef_at, bedroc, roc_auc, pr_auc

    rows = [["layer", "uniprot", "n_actives", "n_decoys",
             "ef1", "ef5", "bedroc", "auroc", "pr_auc"]]
    per_layer = defaultdict(list)
    for L in args.layers:
        recs = [json.loads(l) for l in open(f"{args.eval_dir}/{L}.jsonl")]
        with ProcessPoolExecutor(args.workers) as ex:
            for r in ex.map(one_target, recs, chunksize=4):
                if r is None:
                    continue
                up, lay, s, y = r
                m = (ef_at(s, y, 0.01), ef_at(s, y, 0.05),
                     bedroc(s, y), roc_auc(s, y), pr_auc(s, y))
                rows.append([lay, up, int(y.sum()), int((1 - y).sum())]
                            + [f"{v:.4f}" for v in m])
                per_layer[lay].append(m)
        print(f"{L} 完成 {len(per_layer[L])} 个靶点", flush=True)

    print("\n纯配体基线（只用 2D 指纹，不看蛋白）")
    print("=" * 72)
    print("%-5s %8s %9s %9s %9s %9s %9s" %
          ("层", "靶点", "EF1%", "EF5%", "BEDROC", "AUROC", "PR-AUC"))
    print("-" * 72)
    for L in args.layers:
        v = per_layer.get(L)
        if not v:
            continue
        a = np.array(v)
        print("%-5s %8d %9.2f %9.2f %9.4f %9.4f %9.4f"
              % (L, len(v), *a.mean(axis=0)))
    print("-" * 72)
    print("随机基线： EF=1.00  BEDROC≈0.02  AUROC=0.500  PR-AUC≈0.020")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\n逐靶点写入 {args.out}")


if __name__ == "__main__":
    main()
