"""Per-model difficulty control for the seen/unseen comparison.

"Higher score on seen targets" has a trivial explanation: the targets a
model has seen are mostly well-studied ones, and well-studied targets have
more actives and a more concentrated chemical series, which makes them easy
regardless. To separate "this model was trained on it" from "this target is
easy for anyone", divide out difficulty:

    r_t = EF1%(model, t) / median(EF1%(other nine models, t))

Some of the other nine models were also trained on t, so this baseline is
conservative — it can only **underestimate** the true training effect, never
overestimate it. Only if the seen/unseen ratio of r is still significantly
greater than 1 does that indicate the effect comes from this model's own
training.
"""
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
    "conplex": "C", "sprint": "D",
}
LAYERS = ["L1", "L2", "L3", "L4"]


def load_sets():
    import pickle

    import lmdb
    lab = json.load(open(f"{B}/data/raw/figshare/train_label_blend_seq_full.json"))
    Bs = {a["uniprot"] for a in lab if a.get("uniprot")}
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
    keep = {r["uniprot"] for r in csv.DictReader(
        open(f"{B}/results/export/T3_vsds_matched.csv"))}
    SETS = load_sets()
    S = json.load(open(f"{B}/results/t3/summary.json"))

    EF = {}
    for m in S:
        for L in LAYERS:
            for r in S[m][L]["per_target"]:
                if r["uniprot"] in keep:
                    EF.setdefault(r["uniprot"], {})[m] = r["ef1"]
    ALL = sorted(EF)
    print(f"子集靶点 {len(ALL)}（十个模型不一定都有）")

    rows = [["model", "train_set", "n_seen", "n_unseen", "r_seen_med", "r_unseen_med",
             "fold_med", "r_seen_mean", "r_unseen_mean", "fold_mean", "p_mwu"]]
    print("\n难度归一后的 seen/unseen（r = EF1 ÷ 其余模型在该靶点的中位数）")
    print("%-24s %-3s %6s %7s %9s %9s %8s %8s %10s"
          % ("模型", "集", "seen", "unseen", "r seen", "r unseen",
             "倍数中位", "倍数均值", "p"))
    print("-" * 94)
    for m, tag in MODEL_SET.items():
        T = SETS[tag]
        rs, ru = [], []
        for t in ALL:
            if m not in EF[t]:
                continue
            others = [v for k, v in EF[t].items() if k != m]
            if len(others) < 5:
                continue
            base = float(np.median(others))
            if base <= 0.01:            # target where every model fails; ratio is meaningless
                continue
            (rs if t in T else ru).append(EF[t][m] / base)
        if len(rs) < 5 or len(ru) < 5:
            print(f"{m}: 有一边样本太少（{len(rs)}/{len(ru)}），跳过"); continue
        ms, mu = float(np.median(rs)), float(np.median(ru))
        # The median can be 0 (for ConPLex, more than half of the unseen targets
        # have EF1=0), so the fold ratio can't be computed; report a mean-based
        # version alongside it. The p-value is unaffected since it only looks at ranks.
        As, Au = float(np.mean(rs)), float(np.mean(ru))
        p = mannwhitneyu(rs, ru, alternative="greater").pvalue
        fm = ms / mu if mu > 0 else float("nan")
        fa = As / Au if Au > 0 else float("nan")
        print("%-24s %-3s %6d %7d %9.3f %9.3f %8s %8.2f %10.5f"
              % (m, tag, len(rs), len(ru), ms, mu,
                 ("%.2f" % fm) if fm == fm else "n/a", fa, p))
        rows.append([m, tag, len(rs), len(ru), f"{ms:.4f}", f"{mu:.4f}",
                     f"{fm:.3f}" if fm == fm else "nan",
                     f"{As:.4f}", f"{Au:.4f}", f"{fa:.3f}", f"{p:.6f}"])
    print("-" * 94)
    print("\n倍数接近 1 = 这个模型在自己见过的靶点上并不比别人更强，")
    print("             它的 seen/unseen 差距是靶点难度，不是训练记忆；")
    print("倍数显著 >1 = 确实存在该模型专属的训练集优势。")

    out = f"{B}/results/export/T3_per_model_layers_ctrl.csv"
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\n写入 {out}")


if __name__ == "__main__":
    main()
