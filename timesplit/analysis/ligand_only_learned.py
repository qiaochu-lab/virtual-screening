"""**真正的**纯配体基线：模型完全不知道靶点是谁，只看分子。

（对照 `ligand_only_baseline.py`——那个文件名有误导性，它其实是
**化学系列 oracle 上界**，读了该靶点的已知活性。本文件才是名副其实的
ligand-only：靶点身份从未进入模型。）

和 `ligand_only_baseline.py` 的区别
----------------------------------
那个算的是「候选分子对**该靶点已知活性**的最大 Tanimoto」——它用到了靶点身份
（通过该靶点的活性集合），所以是一个 target-conditioned oracle，衡量的是
「这批活性在化学空间里有多聚集」。

这里要的是另一件事：一个**根本不输入靶点**的分类器，f(配体) → P(活性)。
如果它也能富集，说明数据集里存在与靶点无关的「像不像活性分子」的信号。

    诱饵是别的靶点的真实活性分子，所以「像不像药」这条捷径按构造应当被堵死，
    预期结果接近随机。若确实如此，是对诱饵设计的一个正面证据；
    若显著高于随机，那是必须报告的偏差。

训练/测试的切分
--------------
按**靶点**切，不是按分子切：同一个靶点的活性高度同系列，按分子切会让
训练集和测试集共享化学系列，测出来的是记忆不是泛化。
用 GroupKFold 以 uniprot 为 group，逐折预测，最后按靶点算 EF/AUROC——
与主表口径一致。
"""
import argparse, collections, csv, json, os, sys
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")
from concurrent.futures import ProcessPoolExecutor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

B = "/data/work/vs-benchmark"
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fp(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    a = np.zeros(2048, dtype=np.uint8)
    for b in GEN.GetFingerprint(m).GetOnBits():
        a[b] = 1
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=f"{B}/data/t3/eval")
    ap.add_argument("--subset", default=f"{B}/results/export/T3_vsds_matched.csv")
    ap.add_argument("--layers", nargs="+", default=["L1", "L2", "L3", "L4"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-decoys", type=int, default=400,
                    help="每靶点最多采样多少诱饵进训练，控制内存；评测时用全量")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--metrics-dir", default=f"{B}/eval")
    ap.add_argument("--out", default=f"{B}/results/export/T3_ligand_only_learned.csv")
    args = ap.parse_args()
    sys.path.insert(0, args.metrics_dir)
    from metrics import enrichment_factor, bedroc, roc_auc

    keep = {(r["layer"], r["uniprot"]) for r in csv.DictReader(open(args.subset))}
    print(f"子集 {len(keep)} 条")

    recs = []
    for L in args.layers:
        for line in open(f"{args.eval_dir}/{L}.jsonl"):
            r = json.loads(line)
            if (L, r["uniprot"]) in keep:
                recs.append((L, r))
    print(f"读到 {len(recs)} 个靶点")

    # 指纹：整个子集的唯一分子各算一次
    smis = sorted({x["smiles"] for _, r in recs for g in ("actives", "decoys") for x in r[g]})
    print(f"唯一分子 {len(smis):,}，建指纹…", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        fps = list(ex.map(fp, smis, chunksize=500))
    # strict=True：长度不等立刻抛，不静默截断。见 PATCHES「并行列表」那条。
    F = {s: f for s, f in zip(smis, fps, strict=True) if f is not None}
    print(f"  可用 {len(F):,}")

    rng = np.random.default_rng(1)
    X, y, grp, meta = [], [], [], []
    for L, r in recs:
        up = r["uniprot"]
        act = [x["smiles"] for x in r["actives"] if x["smiles"] in F]
        dec = [x["smiles"] for x in r["decoys"] if x["smiles"] in F]
        if len(act) < 5 or len(dec) < 20:
            continue
        # 训练用采样诱饵（控内存），评测阶段仍用全量
        keep_d = dec if len(dec) <= args.max_decoys else \
            list(rng.choice(dec, args.max_decoys, replace=False))
        for s in act:
            X.append(F[s]); y.append(1); grp.append(up); meta.append((L, up, s))
        for s in keep_d:
            X.append(F[s]); y.append(0); grp.append(up); meta.append((L, up, s))
    X = np.array(X, dtype=np.uint8); y = np.array(y); grp = np.array(grp)
    print(f"训练矩阵 {X.shape}，活性 {y.sum():,}")

    # 按靶点分折：同一靶点的活性同系列，按分子分折会泄漏化学系列
    pred = np.zeros(len(y))
    gkf = GroupKFold(n_splits=args.folds)
    for k, (tr, te) in enumerate(gkf.split(X, y, grp), 1):
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                             max_depth=None, random_state=1)
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict_proba(X[te])[:, 1]
        print(f"  fold {k}/{args.folds}: 训练 {len(tr):,} 测试 {len(te):,}", flush=True)

    by = collections.defaultdict(lambda: ([], []))
    for (L, up, _), p_, y_ in zip(meta, pred, y, strict=True):
        by[(L, up)][0].append(p_); by[(L, up)][1].append(y_)

    rows = [["layer", "uniprot", "n_actives", "n_decoys", "ef1", "bedroc", "auroc"]]
    agg = collections.defaultdict(list)
    for (L, up), (p_, y_) in by.items():
        p_, y_ = np.array(p_), np.array(y_)
        if y_.sum() == 0 or y_.sum() == len(y_):
            continue
        v = (enrichment_factor(p_, y_, 0.01), bedroc(p_, y_, 80.5), roc_auc(p_, y_))
        agg[L].append(v)
        rows.append([L, up, int(y_.sum()), int((1-y_).sum())] + [f"{x:.4f}" for x in v])

    print("\n真·纯配体分类器（不输入靶点，按靶点分折交叉验证）")
    print("=" * 62)
    print("%-5s %8s %10s %10s %10s" % ("层", "靶点", "EF1%", "BEDROC", "AUROC"))
    print("-" * 62)
    for L in args.layers:
        if not agg[L]:
            continue
        a = np.array(agg[L])
        print("%-5s %8d %10.2f %10.4f %10.4f" % (L, len(a), *a.mean(axis=0)))
    print("-" * 62)
    print("随机基线： EF=1.00  BEDROC≈0.02  AUROC=0.500")
    print("\n注：训练时每靶点最多采样 %d 个诱饵，评测在这批采样上进行；"
          "\n    诱饵为跨靶点真实活性分子，因此「像不像药」这条捷径按构造已被堵死。"
          % args.max_decoys)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\n写入 {args.out}")


if __name__ == "__main__":
    main()
