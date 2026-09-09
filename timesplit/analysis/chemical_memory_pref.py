"""化学记忆偏好 CMP：模型捞回的分子，比候选池本身熟多少。

定义
----
对一个靶点，设 S 为分子对训练集配体的最大 ECFP4 Tanimoto：

    CMP = P(S ≥ 0.7 | 被排进 top-1%) − P(S ≥ 0.7 | 候选池全体)

⚠️ 但这个数**不能直接读成「偏好」**（本脚本第一版的 docstring 就读错了）
--------------------------------------------------------------------
活性和诱饵的新颖度分布本来就不同：L1 的活性有 53.9% 落在 ≥0.7 档，诱饵只有
8.7%。所以一个**准**的模型，光是因为把活性排上去，CMP 就会是正的，
哪怕它对熟悉化学毫无偏好。CMP 混了「准确率」和「活性本身有多熟」两件事。

分解
----
把准确率能解释的那部分先算出来。设模型 top-1% 里活性占 π（= precision@1%），
该层活性中熟悉化学占 a、诱饵中占 d，则**纯靠准确率**应当得到：

    CMP_pred = [π·a + (1−π)·d] − 池内熟悉比例

真正的偏好是超出这个预测的残差：

    CMP_excess = CMP_obs − CMP_pred

CMP_excess ≈ 0：模型只是准，对化学熟不熟没有额外偏好。
CMP_excess > 0：在同样的准确率下，它**还是**更愿意捞熟悉的分子。

三个数都写出来：`cmp_all`（原始观测）、`cmp_pred`（准确率能解释的）、
`cmp_excess`（残差，这个才是「偏好」）。
另附 `cmp_actives`——只在该靶点的活性内部比较（捞回的活性 vs 全部活性），
它天然不受活性/诱饵分布差异影响，是 excess 的一个独立旁证。

⚠️ 分子顺序必须硬校验
--------------------
模型读 lmdb，游标是字典序（0, 1, 10, 100, …），和评测集 jsonl 顺序不同，
而两者长度相同——只比长度会静默错配，这个坑在本项目里出现过三次。
`ligand_novelty.py` 的 B 段就只比了长度，本脚本改成验「标签为 1 的位置上
确实是该靶点的 active」，验不过就跳过并报出来。
"""
import argparse
import collections
import csv
import json
import os
import pickle

import numpy as np

B = "/data/work/vs-benchmark"
THR = 0.70          # 「熟悉化学」的门限，与 §3/§4 的「极近」档一致
FRAC = 0.01


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
                # 该靶点的活性 / 诱饵各自的熟悉比例，以及 top-1% 里活性占比
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
            # 逐靶点算 CMP_pred = [π·a + (1−π)·d] − 池内熟悉比例，再取均值
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
