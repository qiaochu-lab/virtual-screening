"""Per-model difficulty control for the seen/unseen comparison — the division-free version.

The first version normalised difficulty with
r_t = EF(model,t) / median(EF(other models,t)), but the ratio blows up when
the denominator is near 0: drugclip's fold ratio has a median of 1.85 but a
mean of 0.57 — opposite directions — showing that this ratio's distribution
is dominated by outliers and can't be reported.

Switch to two statistics that don't need division:

1. **P(seen > unseen)** — the probability that a randomly drawn seen target
   has a larger r than a randomly drawn unseen target. This is just
   Mann-Whitney U / (n1·n2), bounded in [0,1], with 0.5 meaning no
   difference. It is the same test as the p-value, just read for effect size
   rather than significance.

2. **Within-target rank** — at each target, rank the ten models by EF1
   (1 = best), then compare a model's average rank on its own seen targets
   against its unseen targets. This sidesteps division entirely and
   automatically cancels out target difficulty (rank is a within-target
   relative quantity). A better (numerically smaller) rank is what indicates
   a training-set advantage.

The conclusion only holds if both statistics agree in direction.
"""
import csv
import json
import os

import numpy as np
from scipy.stats import mannwhitneyu, rankdata

B = "/data/work/vs-benchmark"
MODEL_SET = {
    "drugclip": "A", "bindclip_randneg": "A", "bindclip_hardneg": "A",
    "ligunity_pocket_ranking": "B", "ligunity_protein_ranking": "B",
    "litenclip": "B", "hypseek_rk": "B",
    "conplex": "C", "sprint": "D",
}
LAYERS = ["L1", "L2", "L3", "L4"]


def load_sets():
    """The four training sets. Set B is the union of **two** label files — see the
    comment on per_model_layers.load_B.

    train_task.py:523-524 reads both the pdbbind half (3,468 UniProt / 16,744 PDB)
    and the blend half (2,196 UniProt). The 16,744 PDB entries covered by the
    pdbbind half are exactly the same as DrugCLIP's train_no_test_af, so
    **A ⊂ B**.
    """
    import pickle

    import lmdb
    Bs = set()
    for f in ("train_label_blend_seq_full.json",
              "train_label/train_label_pdbbind_seq.json"):
        p = f"{B}/data/raw/figshare/{f}"
        if os.path.exists(p):
            Bs |= {a["uniprot"] for a in json.load(open(p)) if a.get("uniprot")}
    pdb2up = json.load(open(f"{B}/data/t3/drugclip_pdb2uniprot.json"))
    e = lmdb.open(f"{B}/data/train_no_test_af/train.lmdb",
                  subdir=False, readonly=True, lock=False)
    A = set()
    with e.begin() as t:
        for _, v in t.cursor():
            pk = pickle.loads(v).get("pocket")
            if pk:
                A |= set(pdb2up.get(str(pk).split("_")[0].upper()[:4], []))
    e.close()
    C = {r["uniprot"] for r in csv.DictReader(
        open(f"{B}/results/export/T3_conplex_train_coverage.csv"))
        if float(r["best_identity_conplex_bindingdb"]) >= 0.95
        or r["in_dude_57"] == "1"}
    D = {l.strip() for l in open(f"{B}/data/sprint_train_uniprots.txt") if l.strip()}
    return {"A": A, "B": Bs, "C": C, "D": D}


def main():
    # ⚠️ The key must be (layer, target), not target alone: the 350 subset is
    # 328 entries / 293 unique uniprots, and 35 uniprots appear in more than
    # one layer. Filtering by uniprot alone would pull in "this target's
    # records in other layers" too — empirically, seen+unseen then reports
    # 417, 89 more than the 328 entries in the subset itself.
    # There's a second issue here: EF used to be keyed by uniprot, so the
    # same target appearing in two layers would get overwritten by whichever
    # layer was processed last, and which value survives depends on
    # iteration order. Keying by (layer, target) instead, with within-target
    # rank also computed inside (layer, target) — which is the more correct
    # convention anyway.
    keep = {(r["layer"], r["uniprot"]) for r in csv.DictReader(
        open(f"{B}/results/export/T3_vsds_matched.csv"))}
    SETS = load_sets()
    S = json.load(open(f"{B}/results/t3/summary.json"))

    EF = {}
    for m in S:
        for L in LAYERS:
            for r in S[m][L]["per_target"]:
                if (L, r["uniprot"]) in keep:
                    EF.setdefault((L, r["uniprot"]), {})[m] = r["ef1"]

    # Use only targets where all ten models have a result — otherwise rank isn't comparable
    full = sorted(t for t in EF if len(EF[t]) == len(S))
    print(f"十个模型都有结果的子集靶点：{len(full)} / {len(keep)}")

    RANK = {}
    for t in full:
        ms = sorted(EF[t])
        rk = rankdata([-EF[t][m] for m in ms], method="average")   # 1 = best
        RANK[t] = dict(zip(ms, rk, strict=True))

    rows = [["model", "train_set", "n_seen", "n_unseen", "P_seen_gt_unseen",
             "p_mwu", "rank_seen", "rank_unseen", "rank_gain"]]
    print("\n每个模型在「自己训练集见过的靶点」上是否更强（难度已由靶点内名次消掉）")
    print("%-24s %-3s %6s %7s %14s %9s %9s %9s %8s"
          % ("模型", "集", "seen", "unseen", "P(seen>unseen)", "p",
             "名次 seen", "名次 un", "名次差"))
    print("-" * 100)
    for m, tag in MODEL_SET.items():
        T = SETS[tag]
        es, eu, ks, ku = [], [], [], []
        for t in full:
            (es if t[1] in T else eu).append(EF[t][m])
            (ks if t[1] in T else ku).append(RANK[t][m])
        if len(es) < 5 or len(eu) < 5:
            print(f"{m}: 有一边样本太少（{len(es)}/{len(eu)}），跳过"); continue
        u = mannwhitneyu(es, eu, alternative="greater")
        P = u.statistic / (len(es) * len(eu))
        rs, ru = float(np.mean(ks)), float(np.mean(ku))
        print("%-24s %-3s %6d %7d %14.3f %9.5f %9.2f %9.2f %+8.2f"
              % (m, tag, len(es), len(eu), P, u.pvalue, rs, ru, ru - rs))
        rows.append([m, tag, len(es), len(eu), f"{P:.4f}", f"{u.pvalue:.6f}",
                     f"{rs:.3f}", f"{ru:.3f}", f"{ru - rs:.3f}"])
    print("-" * 100)
    print("\nP > 0.5 且 名次差 > 0 = 该模型在自己见过的靶点上确实更强（两个口径一致）。")
    print("名次是靶点内部十个模型的相对排位，靶点难不难对它没有影响。")
    print("注意这是**保守**估计：其余九个模型里有的也训练过同一批靶点。")

    out = f"{B}/results/export/T3_per_model_seen_effect.csv"
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\n写入 {out}")


if __name__ == "__main__":
    main()
