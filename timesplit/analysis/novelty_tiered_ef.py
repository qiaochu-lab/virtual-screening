"""按配体新颖度分档，分别算富集——模型在「全新化学」上到底能富集多少。

`ligand_novelty.py` 回答了两件事：测试活性离训练集有多远，以及模型捞回的
活性偏熟还是偏生。但它没回答最直接的那一问——**在全新分子这一档上，
模型的富集是多少**。Mattsson & Walters (bioRxiv 2026.06.29.735309) 提的
Novelty-Tiered Benchmark 要的就是这个数。

指标定义
--------
对某一新颖度档 t（按对训练集配体的最大 Tanimoto 划分）：

    recall_t = 该档活性落进 top-1% 的个数 / 该档活性总数
    EF_t     = recall_t / 0.01

EF_t = 1 表示这一档的活性被捞到的概率和随机一样。这样定义的好处是
**各档之间直接可比**——不受该档活性数量多少的影响。

逐靶点算再取均值（只算该档至少有 MIN_T 个活性的靶点），
与主表的口径一致。

⚠️ 分子顺序必须硬校验
--------------------
模型读 lmdb，游标是字典序（0, 1, 10, 100, …），和评测集 jsonl 顺序不同，
而两者长度相同。只比长度会静默错配——这个坑在本项目里出现过三次。
所以还原顺序后必须验「标签为 1 的位置上确实是该靶点的 active」，
验不过就跳过该靶点并报出来。
"""
import argparse
import collections
import csv
import json
import os
import pickle

import numpy as np

B = "/data/work/vs-benchmark"
TIERS = [(0.0, 0.35, "全新 <0.35"), (0.35, 0.50, "远 0.35–0.5"),
         (0.50, 0.70, "近 0.5–0.7"), (0.70, 1.01, "极近 ≥0.7")]
MIN_T = 3          # 该档至少这么多活性才算这个靶点
FRAC = 0.01


def tier_of(v):
    for lo, hi, name in TIERS:
        if lo <= v < hi:
            return name
    return None


def model_order(up, L, n, rec, labels):
    """还原模型看到的分子顺序并硬校验；对不上返回 None。"""
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
    path = f"{B}/data/T3_6A/{L}/{up}/{up}_lig.lmdb"
    if not os.path.exists(path):
        return None
    try:
        import lmdb
        e = lmdb.open(path, subdir=False, readonly=True, lock=False)
        out = []
        with e.begin() as t:
            for _k, v in t.cursor():
                out.append(pickle.loads(v)["smi"])
        e.close()
    except Exception:
        return None
    return ok(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--novelty", default=f"{B}/data/t3/ligand_novelty.json")
    ap.add_argument("--eval-dir", default=f"{B}/data/t3/eval")
    ap.add_argument("--raw", default=f"{B}/results/t3_raw")
    ap.add_argument("--models", nargs="+",
                    default=["ligunity_protein_ranking", "ligunity_pocket_ranking",
                             "hypseek_rk", "litenclip", "drugclip", "conglude"])
    ap.add_argument("--layers", nargs="+", default=["L1", "L2", "L3", "L4"])
    ap.add_argument("--subset", default=None)
    ap.add_argument("--out", default=f"{B}/results/export/T3_novelty_tiered_ef.csv")
    args = ap.parse_args()

    keep = None
    if args.subset:
        keep = {(r["layer"], r["uniprot"]) for r in csv.DictReader(open(args.subset))}
        print(f"子集过滤：{len(keep)} 条")

    nov = json.load(open(args.novelty))
    recs = {L: [json.loads(x) for x in open(f"{args.eval_dir}/{L}.jsonl")]
            for L in args.layers}

    rows = [["model", "layer", "tier", "n_targets", "n_actives",
             "recall_at_1pct", "ef_tier"]]
    print("\n各新颖度档的富集（EF_t = 该档 recall@1% / 0.01；1.0 = 随机）")
    print("=" * 96)
    hdr = "%-24s %-4s" % ("模型", "层") + "".join("%18s" % t[2] for t in TIERS)
    for m in args.models:
        print("\n" + hdr)
        print("-" * 96)
        for L in args.layers:
            d = f"{args.raw}/{m}/T3/{L}"
            if not os.path.isdir(d):
                continue
            per = collections.defaultdict(list)   # tier -> [ef per target]
            cnt = collections.Counter()
            n_bad = 0
            for r in recs[L]:
                up = r["uniprot"]
                if keep is not None and (L, up) not in keep:
                    continue
                try:
                    p = np.load(f"{d}/{up}/saved_preds.npy").reshape(-1)
                    y = np.load(f"{d}/{up}/saved_labels.npy")
                except Exception:
                    continue
                if len(p) != len(y):
                    continue
                order = model_order(up, L, len(y), r, y)
                if order is None:
                    n_bad += 1
                    continue
                k = int(np.ceil(FRAC * len(y)))
                top = set(np.argsort(-p)[:k].tolist())
                tot = collections.Counter()
                hit = collections.Counter()
                for i in range(len(y)):
                    if y[i] != 1:
                        continue
                    t = tier_of(nov.get(order[i], -1))
                    if t is None:
                        continue
                    tot[t] += 1
                    if i in top:
                        hit[t] += 1
                for t in tot:
                    if tot[t] >= MIN_T:
                        per[t].append((hit[t] / tot[t]) / FRAC)
                        cnt[t] += tot[t]
            if not per:
                continue
            cells = []
            for _, _, t in TIERS:
                v = per.get(t)
                cells.append(f"{np.mean(v):.1f} (n={len(v)})" if v else "—")
                if v:
                    rows.append([m, L, t, len(v), cnt[t],
                                 f"{np.mean(v)*FRAC:.4f}", f"{np.mean(v):.2f}"])
            print("%-24s %-4s" % (m, L) + "".join("%18s" % c for c in cells)
                  + (f"   ⚠️{n_bad} 个靶点顺序校验不过" if n_bad else ""))
    print("=" * 96)
    print("读法：括号里是该档有 ≥3 个活性、因而参与统计的靶点数。")
    print("     同一行左右对比 = 同一模型对「新化学」和「熟化学」的富集差距。")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\n写入 {args.out}")


if __name__ == "__main__":
    main()
