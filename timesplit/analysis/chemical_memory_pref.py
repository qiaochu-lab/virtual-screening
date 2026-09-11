"""Chemical Memory Preference (CMP): how much more familiar are the
molecules a model retrieves than the candidate pool itself.

Definition
----------
For a target, let S be a molecule's maximum ECFP4 Tanimoto to the training
ligands:

    CMP = P(S >= 0.7 | ranked into top-1%) - P(S >= 0.7 | whole candidate pool)

Warning: this number **cannot be read as "preference" directly** (this
script's first-version docstring got that wrong)
----------------------------------------------------------------------------
Actives and decoys already have different novelty distributions: 53.9% of
L1's actives fall in the >=0.7 tier, versus only 8.7% of decoys. So an
**accurate** model gets a positive CMP purely from ranking the actives up,
even with zero preference for familiar chemistry. CMP conflates "accuracy"
with "how familiar the actives themselves are".

Decomposition
--------------
First compute the part accuracy alone explains. Let pi be the fraction of
actives in the model's top-1% (= precision@1%), a be the fraction of that
layer's actives that are familiar chemistry, and d be the same fraction
among decoys. Then **accuracy alone** should give:

    CMP_pred = [pi*a + (1-pi)*d] - fraction familiar in the pool

The real preference is the residual beyond that prediction:

    CMP_excess = CMP_obs - CMP_pred

CMP_excess ~= 0: the model is merely accurate, with no extra preference for
familiar chemistry.
CMP_excess > 0: at matched accuracy, it **still** prefers to retrieve
familiar molecules.

All three numbers are written out: `cmp_all` (raw observed), `cmp_pred`
(what accuracy explains), `cmp_excess` (the residual -- this is the real
"preference").
Also included: `cmp_actives` -- compared only within that target's own
actives (retrieved actives vs. all actives), which is unaffected by the
actives/decoys distribution difference by construction, giving an
independent corroboration of excess.

Warning: molecule order must be strictly validated
-----------------------------------------------------
The model reads an lmdb whose cursor order is lexicographic (0, 1, 10, 100,
...), which differs from the eval-set jsonl order, while the two have the
same length -- comparing only the length silently produces mismatches, and
this pitfall has bitten this project three times. `ligand_novelty.py`'s
section B only compared lengths; this script instead validates that "the
positions labeled 1 really are that target's actives", and skips and reports
whenever that check fails.
"""
import argparse
import collections
import csv
import json
import os
import pickle

import numpy as np

B = "/data/work/vs-benchmark"
THR = 0.70          # threshold for "familiar chemistry", consistent with the "very close" tier in SS3/SS4
FRAC = 0.01


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
                             "hypseek_rk", "litenclip", "drugclip",
                             "bindclip_randneg", "bindclip_hardneg", "conglude",
                             "sprint"])
    ap.add_argument("--layers", nargs="+", default=["L1", "L2", "L3", "L4"])
    ap.add_argument("--subset", default=f"{B}/results/export/T3_vsds_matched.csv")
    ap.add_argument("--out", default=f"{B}/results/export/T3_chemical_memory_pref.csv")
    args = ap.parse_args()

    keep = None
    if args.subset:
        keep = {(r["layer"], r["uniprot"]) for r in csv.DictReader(open(args.subset))}
        print(f"子集过滤：{len(keep)} 条")

    nov = json.load(open(args.novelty))
    recs = {L: [json.loads(x) for x in open(f"{args.eval_dir}/{L}.jsonl")]
            for L in args.layers}

    rows = [["model", "layer", "n_targets", "p_familiar_pool",
             "p_familiar_retrieved", "cmp_all", "cmp_pred", "cmp_excess",
             "cmp_actives", "precision_at_1pct"]]
    print("\n化学记忆偏好，S≥%.1f 算「熟悉」" % THR)
    print("  CMP      = P(熟 | top-1%) − P(熟 | 池)      观测值")
    print("  CMP_pred = 光靠「把活性排上去」就能得到的那部分")
    print("  CMP_exc  = 残差 —— 这个才是真正的偏好")
    print("=" * 104)
    print("%-24s %-4s %6s %8s %9s %9s %9s %9s %10s"
          % ("模型", "层", "靶点", "池内熟%", "捞回熟%", "CMP", "CMP_pred",
             "CMP_exc", "CMP(活性内)"))
    print("-" * 104)
    for m in args.models:
        for L in args.layers:
            d = f"{args.raw}/{m}/T3/{L}"
            if not os.path.isdir(d):
                continue
            pp, pr, pa, pd, n_bad = [], [], [], [], 0
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
                s = np.array([nov.get(x, -1.0) for x in order])
                have = s >= 0
                if have.sum() < 50:
                    continue
                pool = float((s[have] >= THR).mean())
                k = max(1, int(np.ceil(FRAC * len(y))))
                top = np.argsort(-p)[:k]
                th = [i for i in top if s[i] >= 0]
                if not th:
                    continue
                pp.append(pool)
                pr.append(float(np.mean([s[i] >= THR for i in th])))
                # This target's familiar fraction among actives / decoys
                # separately, and the fraction of actives within top-1%
                ai = [i for i in range(len(y)) if y[i] == 1 and s[i] >= 0]
                di = [i for i in range(len(y)) if y[i] == 0 and s[i] >= 0]
                if ai and di:
                    pd.append((float(np.mean([s[i] >= THR for i in ai])),
                               float(np.mean([s[i] >= THR for i in di])),
                               float(np.mean([y[i] == 1 for i in th])),
                               pool))
                ta = [i for i in top if y[i] == 1 and s[i] >= 0]
                aa = [i for i in range(len(y)) if y[i] == 1 and s[i] >= 0]
                if ta and aa:
                    pa.append(float(np.mean([s[i] >= THR for i in ta]))
                              - float(np.mean([s[i] >= THR for i in aa])))
            if len(pp) < 5:
                continue
            c_all = float(np.mean(pr)) - float(np.mean(pp))
            c_act = float(np.mean(pa)) if pa else float("nan")
            # Compute CMP_pred = [pi*a + (1-pi)*d] - pool-familiar fraction
            # per target, then average
            if pd:
                preds = [(pi * a + (1 - pi) * dd) - pl for a, dd, pi, pl in pd]
                c_pred = float(np.mean(preds))
                prec = float(np.mean([x[2] for x in pd]))
            else:
                c_pred, prec = float("nan"), float("nan")
            c_exc = c_all - c_pred
            print("%-24s %-4s %6d %7.1f%% %8.1f%% %+9.3f %+9.3f %+9.3f %+10.3f%s"
                  % (m, L, len(pp), 100 * np.mean(pp), 100 * np.mean(pr),
                     c_all, c_pred, c_exc, c_act,
                     f"  ⚠️{n_bad}" if n_bad else ""))
            rows.append([m, L, len(pp), f"{np.mean(pp):.4f}", f"{np.mean(pr):.4f}",
                         f"{c_all:.4f}", f"{c_pred:.4f}", f"{c_exc:.4f}",
                         f"{c_act:.4f}", f"{prec:.4f}"])
        print("-" * 104)
    print("\n读法：只看 CMP 会把「模型准」误读成「模型偏好熟悉化学」——")
    print("     活性本身就比诱饵熟（L1 活性 53.9% 在 ≥0.7 档，诱饵 8.7%），")
    print("     所以准的模型 CMP 自然为正。**CMP_exc 才是偏好。**")
    print("     CMP(活性内) 是独立旁证：只在该靶点的活性内部比，不受这个混杂影响。")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\n写入 {args.out}")


if __name__ == "__main__":
    main()
