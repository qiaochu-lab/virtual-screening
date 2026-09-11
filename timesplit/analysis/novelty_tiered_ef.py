"""Bin by ligand novelty and compute enrichment per tier -- how much can a
model actually enrich on "completely novel chemistry".

`ligand_novelty.py` answers two things: how far the test actives are from
the training set, and whether the actives a model retrieves skew familiar or
novel. But it never answers the most direct question -- **what is the
model's enrichment on the completely-novel tier**. That is exactly the
number the Novelty-Tiered Benchmark proposed by Mattsson & Walters (bioRxiv
2026.06.29.735309) calls for.

Metric definition
--------------------
For a novelty tier t (binned by maximum Tanimoto to the training ligands):

    recall_t = number of that tier's actives landing in top-1% / total actives in that tier
    EF_t     = recall_t / 0.01

EF_t = 1 means actives in that tier are retrieved with the same probability
as random. The benefit of this definition is that **tiers are directly
comparable to each other** -- unaffected by how many actives that tier has.

Computed per target and then averaged (only over targets with at least
MIN_T actives in that tier), consistent with the main table's convention.

Warning: molecule order must be strictly validated
-----------------------------------------------------
The model reads an lmdb whose cursor order is lexicographic (0, 1, 10, 100,
...), which differs from the eval-set jsonl order while having the same
length. Comparing only the length silently produces mismatches -- this
pitfall has bitten this project three times. So after reconstructing the
order it must be validated that "the positions labeled 1 really are that
target's actives"; skip and report the target if that check fails.
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
MIN_T = 3          # a tier needs at least this many actives to count for that target
FRAC = 0.01


def tier_of(v):
    for lo, hi, name in TIERS:
        if lo <= v < hi:
            return name
    return None


def model_order(up, L, n, rec, labels):
    """Reconstruct the molecule order the model actually saw and strictly
    validate it; returns None if it doesn't check out."""
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
    ap.add_argument("--target-groups", default=None,
                    help="csv：uniprot,group。给了就按这一列分组而不是按 L1–L4，"
                         "用来做「每个模型按自己训练集判 seen/unseen」的版本")
    ap.add_argument("--out", default=f"{B}/results/export/T3_novelty_tiered_ef.csv")
    args = ap.parse_args()

    keep = None
    if args.subset:
        keep = {(r["layer"], r["uniprot"]) for r in csv.DictReader(open(args.subset))}
        print(f"子集过滤：{len(keep)} 条")

    # Grouping: by layer by default; if --target-groups is given, group by
    # that column instead (merging same-group targets across layers)
    groups = None
    if args.target_groups:
        groups = {r["uniprot"]: r["group"]
                  for r in csv.DictReader(open(args.target_groups))}
        print(f"按 {args.target_groups} 分组，{len(set(groups.values()))} 组 / "
              f"{len(groups)} 个靶点")

    nov = json.load(open(args.novelty))
    recs = {L: [json.loads(x) for x in open(f"{args.eval_dir}/{L}.jsonl")]
            for L in args.layers}

    rows = [["model", "group", "tier", "n_targets", "n_actives",
             "recall_at_1pct", "ef_tier"]]
    print("\n各新颖度档的富集（EF_t = 该档 recall@1% / 0.01；1.0 = 随机）")
    print("=" * 96)
    col = "组" if groups else "层"
    hdr = "%-24s %-8s" % ("模型", col) + "".join("%18s" % t[2] for t in TIERS)
    for m in args.models:
        print("\n" + hdr)
        print("-" * 96)
        # bucket: group name -> tier -> [per-target EF_t]; iterate by layer but accumulate by group
        bucket = collections.defaultdict(lambda: collections.defaultdict(list))
        count = collections.defaultdict(collections.Counter)
        n_bad = 0
        for L in args.layers:
            d = f"{args.raw}/{m}/T3/{L}"
            if not os.path.isdir(d):
                continue
            for r in recs[L]:
                up = r["uniprot"]
                if keep is not None and (L, up) not in keep:
                    continue
                g = groups.get(up) if groups else L
                if g is None:
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
                        bucket[g][t].append((hit[t] / tot[t]) / FRAC)
                        count[g][t] += tot[t]
        order_g = (sorted(bucket) if groups else
                   [L for L in args.layers if L in bucket])
        for g in order_g:
            cells = []
            for _, _, t in TIERS:
                v = bucket[g].get(t)
                cells.append(f"{np.mean(v):.1f} (n={len(v)})" if v else "—")
                if v:
                    rows.append([m, g, t, len(v), count[g][t],
                                 f"{np.mean(v)*FRAC:.4f}", f"{np.mean(v):.2f}"])
            print("%-24s %-8s" % (m, g) + "".join("%18s" % c for c in cells))
        if n_bad:
            print("%-24s %-8s ⚠️ %d 个靶点顺序校验不过，已跳过" % ("", "", n_bad))
    print("=" * 96)
    print("读法：括号里是该档有 ≥3 个活性、因而参与统计的靶点数。")
    print("     同一行左右对比 = 同一模型对「新化学」和「熟化学」的富集差距。")
    if groups:
        print("     上下对比 = 同一化学新颖度下，靶点见没见过带来的差距。")
        print("     ⚠️ 新颖度档本身是按 PocketAffDB 的配体算的，不是按各模型自己的"
              "训练配体，")
        print("        所以这张表解决了靶点侧的循环性，没解决配体侧的。")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\n写入 {args.out}")


if __name__ == "__main__":
    main()
