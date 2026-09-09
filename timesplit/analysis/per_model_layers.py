"""逐模型分层：每个模型按**它自己的**训练集判 seen/unseen，各算各的衰减。

为什么
------
现在的 L1–L4 是按 PocketAffDB（B）画的，而 B 只是十个模型里四个的训练集。
逐模型审计显示这套标签对别人错配得很厉害：A 在 L1 只覆盖 77%，在 L3/L4
却覆盖 21–23%。用这套标签比较不同训练集的模型再谈「训练数据 vs 架构」，
是循环论证。

这里换成每个模型自己的口径：靶点在它训练集里 = seen，否则 = unseen，
然后算 seen→unseen 的衰减，跨模型直接可比。

为什么只做两层不做四层
----------------------
L1/L2 的分界是配体骨架见没见过，L3/L4 是家族见没见过。前者要每个模型的
**训练配体**清单，后者要按每个模型训练集重跑一次聚类。配体清单十个模型
里只有两三个拿得到，所以四层的逐模型版做不实。靶点级的 seen/unseen 是
能对所有已知训练集的模型一致执行的最细粒度，先把这层做干净。

四套训练集
----------
  A  train_no_test_af  →  DrugCLIP、BindCLIP ×2
  B  PocketAffDB       →  LigUnity ×2、LiTENCLIP、HypSeek
  C  ConPLex BindingDB →  ConPLex（mmseqs 反查，≥95% 同一性）
  D  SPRINT MERGED     →  SPRINT
  ConGLUDe 的清单仍未获得，跳过。

衰减一律按「超出随机」算：EF 的随机底是 1.0，AUROC 是 0.5。
"""
import argparse
import csv
import json
import os

import numpy as np
from scipy.stats import mannwhitneyu

B = "/data/work/vs-benchmark"

MODEL_SET = {
    "drugclip": "A", "bindclip_randneg": "A", "bindclip_hardneg": "A",
    "ligunity_pocket_ranking": "B", "ligunity_protein_ranking": "B",
    "litenclip": "B", "hypseek_rk": "B",
    "conplex": "C",
    "sprint": "D",
    # conglude: 清单未获得
}
LAYERS = ["L1", "L2", "L3", "L4"]


def decay(hi, lo, floor):
    """超出随机基线的损失比例。floor 是该指标的随机值。"""
    base = hi - floor
    return float("nan") if base <= 0 else (base - (lo - floor)) / base


def load_B():
    lab = json.load(open(f"{B}/data/raw/figshare/train_label_blend_seq_full.json"))
    return {a["uniprot"] for a in lab if a.get("uniprot")}


def load_A():
    import pickle

    import lmdb
    pdb2up = json.load(open(f"{B}/data/t3/drugclip_pdb2uniprot.json"))
    e = lmdb.open(f"{B}/data/train_no_test_af/train.lmdb",
                  subdir=False, readonly=True, lock=False)
    ups = set()
    with e.begin() as t:
        for _, v in t.cursor():
            pk = pickle.loads(v).get("pocket")
            if pk:
                ups |= set(pdb2up.get(str(pk).split("_")[0].upper()[:4], []))
    e.close()
    return ups


def load_C(thr=0.95):
    p = f"{B}/results/export/T3_conplex_train_coverage.csv"
    return {r["uniprot"] for r in csv.DictReader(open(p))
            if float(r["best_identity_conplex_bindingdb"]) >= thr
            or r["in_dude_57"] == "1"}


def load_D():
    p = f"{B}/data/sprint_train_uniprots.txt"
    return {l.strip() for l in open(p) if l.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=f"{B}/results/t3/summary.json")
    ap.add_argument("--subset", default=f"{B}/results/export/T3_vsds_matched.csv")
    ap.add_argument("--out", default=f"{B}/results/export/T3_per_model_layers.csv")
    args = ap.parse_args()

    keep = {r["uniprot"] for r in csv.DictReader(open(args.subset))}
    orig = {r["uniprot"]: r["layer"] for r in csv.DictReader(open(args.subset))}
    print(f"子集靶点 {len(keep)}")

    SETS = {"A": load_A(), "B": load_B(), "C": load_C(), "D": load_D()}
    for k, v in SETS.items():
        print(f"  训练集 {k}: {len(v):,} UniProt，覆盖子集 "
              f"{len(v & keep)}/{len(keep)} ({100*len(v & keep)/len(keep):.0f}%)")

    S = json.load(open(args.summary))
    rows = [["model", "train_set", "n_seen", "n_unseen", "ef1_seen", "ef1_unseen",
             "ef1_decay", "auroc_seen", "auroc_unseen", "auroc_decay", "p_mwu",
             "ef1_L1", "ef1_L4", "ef1_decay_L1L4"]]

    print("\n每个模型按自己的训练集分 seen/unseen（350 子集）")
    print("%-24s %-3s %5s %6s %8s %8s %8s %8s %8s %9s"
          % ("模型", "集", "seen", "unseen", "EF1 seen", "EF1 un", "EF1 衰减",
             "AUC seen", "AUC un", "p"))
    print("-" * 100)

    for m, tag in MODEL_SET.items():
        if m not in S:
            print(f"{m}: summary.json 里没有，跳过"); continue
        T = SETS[tag]
        seen, unseen = [], []
        l1, l4 = [], []
        for L in LAYERS:
            for r in S[m][L]["per_target"]:
                u = r["uniprot"]
                if u not in keep:
                    continue
                (seen if u in T else unseen).append(r)
                if orig.get(u) == "L1":
                    l1.append(r)
                elif orig.get(u) == "L4":
                    l4.append(r)
        if not seen or not unseen:
            print(f"{m}: seen/unseen 有一边是空的，跳过"); continue
        g = lambda a, k: float(np.mean([x[k] for x in a]))
        es, eu = g(seen, "ef1"), g(unseen, "ef1")
        as_, au = g(seen, "auroc"), g(unseen, "auroc")
        p = mannwhitneyu([x["ef1"] for x in seen], [x["ef1"] for x in unseen],
                         alternative="greater").pvalue
        d_ef, d_au = decay(es, eu, 1.0), decay(as_, au, 0.5)
        e1, e4 = (g(l1, "ef1") if l1 else float("nan"),
                  g(l4, "ef1") if l4 else float("nan"))
        print("%-24s %-3s %5d %6d %8.2f %8.2f %7.0f%% %8.3f %8.3f %9.5f"
              % (m, tag, len(seen), len(unseen), es, eu, 100 * d_ef, as_, au, p))
        rows.append([m, tag, len(seen), len(unseen), f"{es:.4f}", f"{eu:.4f}",
                     f"{d_ef:.4f}", f"{as_:.4f}", f"{au:.4f}", f"{d_au:.4f}",
                     f"{p:.6f}", f"{e1:.4f}", f"{e4:.4f}",
                     f"{decay(e1, e4, 1.0):.4f}"])

    print("-" * 100)
    print("\n和现行 L1→L4 分层的对照（同一批靶点、同一批分数，只是换了标签）")
    print("%-24s %-3s %12s %14s %10s"
          % ("模型", "集", "自己的衰减", "L1→L4 衰减", "差"))
    print("-" * 70)
    for r in rows[1:]:
        own, l14 = float(r[6]), float(r[13])
        print("%-24s %-3s %11.0f%% %13.0f%% %9.1f pt"
              % (r[0], r[1], 100 * own, 100 * l14, 100 * (own - l14)))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\n写入 {args.out}")


if __name__ == "__main__":
    main()
