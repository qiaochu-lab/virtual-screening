"""测试配体离训练集有多远——按配体新颖度分档，看模型捞到的是新分子还是熟分子。

背景
----
Mattsson & Walters (bioRxiv 2026.06.29.735309) 的核心提议是
**Novelty-Tiered Benchmark**：把测试数据按配体新颖度分档，最难那档
（对训练配体 Tanimoto < 0.35）才是检验真泛化的地方。

我们原来的 L1/L2 分界用的是 Bemis-Murcko 骨架**是否见过**（二值）。
两个分子可以骨架不同但 Tanimoto 0.7，所以这个分界偏粗。这里换成连续相似度。

两部分
------
A. 每个 T3 分子对训练集 42.6 万个配体的最大 Tanimoto，按层看分布；
   诱饵作对照——诱饵是别的靶点的真实活性，它们的新颖度就是"背景水平"。
B. 更要紧的一问：**模型排进 top-1% 的活性里，有多少是新的？**
   如果模型只捞回熟分子，那 EF 高就只说明它记性好。

只对 T3 的 14.7 万个唯一分子各算一次（不是逐"分子×靶点"对），
所以 42.6 万 × 14.7 万的比较跑一遍就够。
"""
import argparse
import collections
import json
import os
import pickle

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")
from concurrent.futures import ProcessPoolExecutor

B = "/data/work/vs-benchmark"
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
TIERS = [(0.0, 0.35, "全新 <0.35"), (0.35, 0.50, "远 0.35–0.5"),
         (0.50, 0.70, "近 0.5–0.7"), (0.70, 1.01, "极近 ≥0.7")]
TRAIN_FPS = None


def fp(smi):
    m = Chem.MolFromSmiles(smi)
    return GEN.GetFingerprint(m) if m else None


def _init(fps):
    global TRAIN_FPS
    TRAIN_FPS = fps


def max_sim(smi):
    f = fp(smi)
    if f is None:
        return -1.0
    return max(DataStructs.BulkTanimotoSimilarity(f, TRAIN_FPS))


def model_order(up, L, n, rec, labels, b_root):
    """还原模型看到的分子顺序并**硬校验**；对不上返回 None。

    ⚠️ 只比长度是不够的。模型读 lmdb，游标是字典序（0, 1, 10, 100, …），
    与评测集 jsonl 的「活性+诱饵」顺序不同而**长度相同**。本项目已经因为
    只比长度而静默错配三次（PATCHES.md）。所以还原顺序后必须验
    「标签为 1 的位置上确实是该靶点的 active」，验不过就跳过该靶点。
    """
    act = {x["smiles"] for x in rec["actives"]}

    def ok(seq):
        if seq is None or len(seq) != n:
            return None
        got = {seq[i] for i in range(n) if labels[i] == 1}
        return seq if got == act else None

    r = ok([x["smiles"] for x in rec["actives"]] +
           [x["smiles"] for x in rec["decoys"]])
    if r is not None:
        return r
    path = f"{b_root}/data/T3_6A/{L}/{up}/{up}_lig.lmdb"
    if not os.path.exists(path):
        return None
    try:
        import lmdb, pickle
        e = lmdb.open(path, subdir=False, readonly=True, lock=False)
        out = []
        with e.begin() as t:
            for _k, v in t.cursor():
                out.append(pickle.loads(v)["smi"])
        e.close()
    except Exception:
        return None
    return ok(out)


def tier_of(v):
    for lo, hi, name in TIERS:
        if lo <= v < hi:
            return name
    return "未知"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-label",
                    default=f"{B}/code/LigUnity/test_datasets/train_label_blend_seq_full.json")
    ap.add_argument("--eval-dir", default=f"{B}/data/t3/eval")
    ap.add_argument("--raw", default=f"{B}/results/t3_raw",
                    help="逐靶点打分目录，用于 B 部分")
    ap.add_argument("--models", nargs="+",
                    default=["ligunity_protein_ranking", "hypseek_rk", "drugclip"])
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--cache", default=f"{B}/data/t3/ligand_novelty.json")
    ap.add_argument("--out", default=f"{B}/results/export/T3_ligand_novelty.csv")
    args = ap.parse_args()

    # ---------- 训练集指纹 ----------
    lab = json.load(open(args.train_label))
    tsmi = sorted({l["smi"] for a in lab for l in (a.get("ligands") or [])
                   if isinstance(l, dict) and l.get("smi")})
    print(f"训练集去重 SMILES {len(tsmi):,}，建指纹…", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        tfps = [f for f in ex.map(fp, tsmi, chunksize=1000) if f is not None]
    print(f"训练集指纹 {len(tfps):,}", flush=True)

    # ---------- T3 唯一分子的新颖度 ----------
    recs = {L: [json.loads(x) for x in open(f"{args.eval_dir}/{L}.jsonl")]
            for L in ("L1", "L2", "L3", "L4")}
    mols = {}
    for L, rs in recs.items():
        for r in rs:
            for grp in ("actives", "decoys"):
                for x in r[grp]:
                    mols.setdefault(x["smiles"], None)
    smis = sorted(mols)
    print(f"T3 唯一分子 {len(smis):,}，算最大 Tanimoto…", flush=True)

    if os.path.exists(args.cache):
        nov = json.load(open(args.cache))
        print(f"  复用缓存 {args.cache}（{len(nov):,} 条）")
    else:
        nov = {}
        with ProcessPoolExecutor(args.workers, initializer=_init,
                                 initargs=(tfps,)) as ex:
            for i, (s, v) in enumerate(zip(smis, ex.map(max_sim, smis,
                                                        chunksize=200),
                                           strict=True)):
                nov[s] = v
                if (i + 1) % 20000 == 0:
                    print(f"  {i+1:,}/{len(smis):,}", flush=True)
        json.dump(nov, open(args.cache, "w"))
        print(f"  写入 {args.cache}")

    # ---------- A. 分布 ----------
    print("\n" + "=" * 78)
    print("A. 配体新颖度分布（对训练集 42.6 万配体的最大 Tanimoto）")
    print("=" * 78)
    rows = [["layer", "group", "n", "median_sim"]
            + [t[2] for t in TIERS]]
    for L in ("L1", "L2", "L3", "L4"):
        for grp in ("actives", "decoys"):
            v = [nov[x["smiles"]] for r in recs[L] for x in r[grp]
                 if nov.get(x["smiles"], -1) >= 0]
            if not v:
                continue
            c = collections.Counter(tier_of(x) for x in v)
            n = len(v)
            print("%-4s %-8s %9d  中位 %.3f   " % (L, grp, n, np.median(v))
                  + "  ".join(f"{t[2]} {100*c[t[2]]/n:4.1f}%" for t in TIERS))
            rows.append([L, grp, n, f"{np.median(v):.4f}"]
                        + [f"{100*c[t[2]]/n:.2f}" for t in TIERS])
        print()

    # ---------- B. 模型捞回的是新分子还是熟分子 ----------
    print("=" * 78)
    print("B. 模型排进 top-1% 的活性里，各新颖度档占多少")
    print("   （对照：该层活性本身的档位构成，见 A）")
    print("=" * 78)
    rows.append([])
    rows.append(["model", "layer", "n_found", "median_sim_found"]
                + [t[2] for t in TIERS])
    for m in args.models:
        for L in ("L1", "L4"):
            d = f"{args.raw}/{m}/T3/{L}"
            if not os.path.isdir(d):
                continue
            found = []
            n_bad = 0
            for r in recs[L]:
                up = r["uniprot"]
                try:
                    p = np.load(f"{d}/{up}/saved_preds.npy").reshape(-1)
                    y = np.load(f"{d}/{up}/saved_labels.npy")
                except Exception:
                    continue
                if len(p) != len(y):
                    continue
                order = model_order(up, L, len(y), r, y, B)
                if order is None:          # 顺序验不过，跳过而不是猜
                    n_bad += 1
                    continue
                k = int(np.ceil(0.01 * len(y)))
                top = np.argsort(-p)[:k]
                found += [nov[order[i]] for i in top
                          if y[i] == 1 and nov.get(order[i], -1) >= 0]
            if len(found) < 20:
                continue
            c = collections.Counter(tier_of(x) for x in found)
            n = len(found)
            print("%-26s %-4s 捞回 %6d  中位 %.3f   " % (m, L, n, np.median(found))
                  + "  ".join(f"{t[2]} {100*c[t[2]]/n:4.1f}%" for t in TIERS))
            rows.append([m, L, n, f"{np.median(found):.4f}"]
                        + [f"{100*c[t[2]]/n:.2f}" for t in TIERS])
            if n_bad:
                print("%-26s %-4s ⚠️ 分子顺序校验未通过而跳过 %d 个靶点"
                      % ("", L, n_bad))
        print()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import csv
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"写入 {args.out}")


if __name__ == "__main__":
    main()
