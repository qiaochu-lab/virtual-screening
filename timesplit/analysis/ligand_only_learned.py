"""The **real** pure-ligand baseline: the model has no idea which target it
is, and sees only the molecule.

(Compare `ligand_only_baseline.py` -- that file's name is misleading; it is
actually the **chemical-series oracle ceiling**, which reads that target's
known actives. This file is the one that is genuinely ligand-only: target
identity never enters the model.)

Difference from `ligand_only_baseline.py`
---------------------------------------------
That one computes "the candidate molecule's maximum Tanimoto to **that
target's known actives**" -- it uses target identity (via that target's
active set), so it is a target-conditioned oracle, measuring "how clustered
this batch of actives is in chemical space".

What is wanted here is a different thing: a classifier that **never receives
the target as input at all**, f(ligand) -> P(active). If it can still enrich,
that means the dataset contains a target-independent "does this look like an
active" signal.

    Decoys are real active molecules from other targets, so the "does it
    look drug-like" shortcut should be blocked by construction, and the
    expected result is close to random. If that holds, it is positive
    evidence for the decoy design; if it is significantly above random,
    that is a bias that must be reported.

Train/test split
-------------------
Split by **target**, not by molecule: one target's actives are highly
same-series, so splitting by molecule would let the training and test sets
share a chemical series, measuring memorization rather than generalization.
Use GroupKFold with uniprot as the group, predict fold by fold, then compute
EF/AUROC per target -- consistent with the main table's convention.
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
    ap.add_argument("--clf", choices=["gbdt", "mlp"], default="gbdt",
                    help="任务清单 4.2 要求两个实现：GBDT 和 2 层 MLP。"
                         "两个都跑是为了确认结论不是某个模型族的特例——"
                         "树模型和神经网络在稀疏二值指纹上的归纳偏置很不一样，"
                         "如果两边都低于随机，那就不是模型选择的问题。")
    ap.add_argument("--out", default=None,
                    help="不给就按 --clf 自动命名")
    args = ap.parse_args()
    if args.out is None:
        suffix = "" if args.clf == "gbdt" else f"_{args.clf}"
        args.out = f"{B}/results/export/T3_ligand_only_learned{suffix}.csv"
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

    # Fingerprints: computed once per unique molecule across the whole subset
    smis = sorted({x["smiles"] for _, r in recs for g in ("actives", "decoys") for x in r[g]})
    print(f"唯一分子 {len(smis):,}，建指纹…", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        fps = list(ex.map(fp, smis, chunksize=500))
    # strict=True: raises immediately on unequal length, no silent
    # truncation. See the "parallel lists" entry in PATCHES.
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
        # Sample decoys for training (to control memory); evaluation still uses the full set
        keep_d = dec if len(dec) <= args.max_decoys else \
            list(rng.choice(dec, args.max_decoys, replace=False))
        for s in act:
            X.append(F[s]); y.append(1); grp.append(up); meta.append((L, up, s))
        for s in keep_d:
            X.append(F[s]); y.append(0); grp.append(up); meta.append((L, up, s))
    X = np.array(X, dtype=np.uint8); y = np.array(y); grp = np.array(grp)
    print(f"训练矩阵 {X.shape}，活性 {y.sum():,}，分类器 {args.clf}")

    # Split folds by target: one target's actives are same-series, so
    # splitting by molecule would leak the chemical series
    pred = np.zeros(len(y))
    gkf = GroupKFold(n_splits=args.folds)
    def make_clf():
        if args.clf == "mlp":
            # 2-layer MLP, hidden layers (512, 128). early_stopping carves
            # another 10% out of the training set internally, never touching
            # the test fold -- otherwise it would be tuning the stopping
            # point on test data.
            from sklearn.neural_network import MLPClassifier
            return MLPClassifier(hidden_layer_sizes=(512, 128), activation="relu",
                                 alpha=1e-4, batch_size=256, learning_rate_init=1e-3,
                                 max_iter=60, early_stopping=True,
                                 n_iter_no_change=5, validation_fraction=0.1,
                                 random_state=1)
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                              max_depth=None, random_state=1)

    for k, (tr, te) in enumerate(gkf.split(X, y, grp), 1):
        clf = make_clf()
        Xtr = X[tr].astype(np.float32) if args.clf == "mlp" else X[tr]
        Xte = X[te].astype(np.float32) if args.clf == "mlp" else X[te]
        clf.fit(Xtr, y[tr])
        pred[te] = clf.predict_proba(Xte)[:, 1]
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
