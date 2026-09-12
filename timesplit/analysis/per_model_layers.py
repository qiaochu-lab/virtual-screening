"""Per-model layering: judge seen/unseen against **each model's own**
training set, and compute each model's own decay.

Why
----
The current L1-L4 is drawn against PocketAffDB (B), and B is the training
set for only four of the ten models. A per-model audit shows this label set
badly mismatches the others: A only covers 77% at L1, yet covers just 21-23%
at L3/L4. Using this label set to compare models with different training
sets and then claiming "training data vs. architecture" is circular.

This switches to each model's own convention instead: a target is seen if
it is in that model's training set, unseen otherwise, and then computes the
seen->unseen decay, which is directly comparable across models.

Why only two tiers, not four
-------------------------------
L1/L2's boundary is whether the ligand scaffold has been seen, L3/L4's is
whether the family has been seen. The former needs each model's **training
ligand** list, the latter needs re-running clustering against each model's
training set. The ligand list is only obtainable for two or three of the
ten models, so a four-layer per-model version isn't practical. Target-level
seen/unseen is the finest granularity that can be applied consistently to
every model whose training set is known, so get this layer clean first.

Four training sets
---------------------
  A  train_no_test_af  ->  DrugCLIP, BindCLIP x2
  B  PocketAffDB       ->  LigUnity x2, LiTENCLIP, HypSeek
  C  ConPLex BindingDB ->  ConPLex (mmseqs reverse lookup, >=95% identity)
  D  SPRINT MERGED     ->  SPRINT
  ConGLUDe's list is still unobtained, skipped.

Decay is always computed as "excess over random": EF's random floor is 1.0,
AUROC's is 0.5.
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
    "litenclip": "B", "hypseek_official_vs": "B",
    "conplex": "C",
    "sprint": "D",
    # conglude: list not obtained
}
LAYERS = ["L1", "L2", "L3", "L4"]


def decay(hi, lo, floor):
    """Fractional loss beyond the random baseline. floor is that metric's
    random value."""
    base = hi - floor
    return float("nan") if base <= 0 else (base - (lo - floor)) / base


def load_B():
    """Group B's (LigUnity x2 / LiTENCLIP / HypSeek) training targets = the
    union of two label files.

    train_task.py:523-524 reads both train_label_pdbbind_seq.json (the
    structure half, 3,468 UniProt / 16,744 PDB) and
    train_label_blend_seq_full.json (the affinity half, 2,196). Only the
    latter was counted previously.

    And the structure half's 16,744 PDB entries are **exactly identical** to
    DrugCLIP's train_no_test_af (intersection 16,744, neither has any
    exclusive entries), so group A's training structures are a proper
    subset of group B's.
    """
    ups = set()
    for f in ("train_label_blend_seq_full.json",
              f"train_label/train_label_pdbbind_seq.json"):
        p = f"{B}/data/raw/figshare/{f}"
        if not os.path.exists(p):
            continue
        for a in json.load(open(p)):
            if a.get("uniprot"):
                ups.add(a["uniprot"])
    return ups


def load_B_blend_only():
    """The affinity half only -- used to separate "affinity labels" from
    "structure"."""
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

    # Warning: the key must be (layer, uniprot), not uniprot alone: the
    # 350-quota subset has 328 entries / 293 unique uniprots, and 35
    # uniprots appear in more than one layer. Filtering by uniprot alone
    # would also pull in "that target's records in other layers" -- measured
    # in practice, seen+unseen came out to 417, 89 more than the subset's
    # own 328 entries.
    keep = {(r["layer"], r["uniprot"]) for r in csv.DictReader(open(args.subset))}
    orig = {r["uniprot"]: r["layer"] for r in csv.DictReader(open(args.subset))}
    print(f"子集靶点 {len(keep)}")

    SETS = {"A": load_A(), "B": load_B(), "C": load_C(), "D": load_D()}
    ups = {u for _, u in keep}          # keep holds (layer, target) pairs; coverage must be computed per target
    for k, v in SETS.items():
        n = len(v & ups)
        print(f"  训练集 {k}: {len(v):,} UniProt，覆盖子集 "
              f"{n}/{len(ups)} 个唯一靶点 ({100*n/len(ups):.0f}%)")

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
                if (L, u) not in keep:
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
