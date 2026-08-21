"""T2 亲和力排序 —— 修正版，按分子身份对齐，而不是按下标顺序。

为什么要重写
------------
旧版 score_t2.py 这样取值：
    act = np.nonzero(lab == 1)[0]     # 模型顺序里 active 的下标（升序）
    sc  = s[act]
    ρ   = spearman(sc, pa)            # pa 来自评测集 jsonl 的 active 顺序
它默认「模型顺序里的第 k 个 active，就是评测集里的第 k 个 active」。
对 ConGLUDe / ConPLex 成立（它们直接遍历 jsonl）；
对 UniMol 系（DrugCLIP/BindCLIP/LigUnity/LiTENCLIP/HypSeek）**不成立**——
它们读 lmdb，而 lmdb 的 key 是字符串，遍历顺序是字典序
（0, 1, 10, 100, 1000, …），不是写入时的数值序。

后果：分数和亲和力被打乱配对，相关系数被摊平到零。
这正好解释了一件一直没想通的事——ConGLUDe（唯一走 jsonl 顺序的模型）
在 T3 上的 ρ 是所有模型里最高的（L1 +0.129），而七个结构模型全在 0 附近。
不是它更强，是只有它没被这个 bug 打乱。

修正做法
--------
按**分子身份**对齐：先还原模型看到的分子顺序（lmdb 游标序或 jsonl 序），
用 InChIKey 把每个 active 对上它自己的 pAffinity，再算相关。
InChIKey 由 SMILES 现算，不依赖任何顺序假设。

同时报旧口径的数，方便审计这次修正到底改了多少。
"""
import argparse
import json
import os
import pickle

import lmdb
import numpy as np
from rdkit import Chem, RDLogger
from scipy import stats

RDLogger.DisableLog("rdApp.*")
B = "/data/work/vs-benchmark"
MIN_ACT = 10


def model_smiles(up, L, n_pred, rec):
    """模型看到的分子顺序 -> [smiles]；两种布局都试，都对不上返回 None。"""
    jsonl = [m["smiles"] for m in rec["actives"]] + [m["smiles"] for m in rec["decoys"]]
    if len(jsonl) == n_pred:
        return jsonl
    p = f"{B}/data/T3_6A/{L}/{up}/{up}_lig.lmdb"
    if not os.path.exists(p):
        return None
    e = lmdb.open(p, subdir=False, readonly=True, lock=False)
    out = []
    with e.begin() as t:
        for _k, v in t.cursor():      # 游标序 = 模型看到的顺序
            out.append(pickle.loads(v)["smi"])
    e.close()
    return out if len(out) == n_pred else None


def ikey(s, cache):
    if s in cache:
        return cache[s]
    m = Chem.MolFromSmiles(s)
    try:
        k = Chem.MolToInchiKey(m) if m is not None else ""
    except Exception:
        k = ""
    cache[s] = k
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--layers", nargs="+", default=["L1", "L2", "L3", "L4"])
    ap.add_argument("--out", default=f"{B}/results/t3/summary_t2_v2.json")
    args = ap.parse_args()

    EV = {}
    for L in args.layers:
        p = f"{B}/data/t3/eval/{L}.jsonl"
        EV[L] = {json.loads(x)["uniprot"]: json.loads(x) for x in open(p)} \
            if os.path.exists(p) else {}

    cache, summary = {}, {}
    print("T2 亲和力排序：按分子身份对齐（修正）vs 按下标顺序（旧口径）")
    print("=" * 92)
    print("%-26s %-4s %7s %14s %14s %10s %9s" %
          ("模型", "层", "靶点", "修正 ρ", "旧口径 ρ", "修正 τ", "ρ>0 占比"))
    print("-" * 92)
    for m in args.models:
        summary[m] = {}
        for L in args.layers:
            d = f"{B}/results/t3_raw/{m}/T3/{L}"
            if not os.path.isdir(d):
                d = f"{B}/results/t3/{m}/{L}"
            if not os.path.isdir(d):
                continue
            new_r, old_r, new_t, ns, skip = [], [], [], [], 0
            for up in sorted(os.listdir(d)):
                rec = EV[L].get(up)
                if rec is None:
                    continue
                try:
                    s = np.load(f"{d}/{up}/saved_preds.npy").reshape(-1)
                    lab = np.load(f"{d}/{up}/saved_labels.npy")
                except Exception:
                    continue
                if len(s) != len(lab):
                    continue
                act_idx = np.nonzero(lab == 1)[0]
                if len(act_idx) < MIN_ACT:
                    continue
                smis = model_smiles(up, L, len(s), rec)
                if smis is None:
                    skip += 1
                    continue
                aff = {ikey(a["smiles"], cache): float(a["paff"]) for a in rec["actives"]}
                pairs = []
                for i in act_idx:
                    k = ikey(smis[i], cache)
                    if k in aff:
                        pairs.append((float(s[i]), aff[k]))
                if len(pairs) < MIN_ACT:
                    skip += 1
                    continue
                sc = np.array([x[0] for x in pairs])
                pa = np.array([x[1] for x in pairs])
                if np.std(sc) == 0 or np.std(pa) == 0:
                    continue
                r = stats.spearmanr(sc, pa).statistic
                t = stats.kendalltau(sc, pa).statistic
                if not np.isfinite(r):
                    continue
                new_r.append(r); new_t.append(t); ns.append(len(pairs))

                # 旧口径：模型 active 下标（升序）直接对评测集 active 顺序
                pa_old = [float(a["paff"]) for a in rec["actives"]]
                if len(pa_old) == len(act_idx):
                    ro = stats.spearmanr(s[act_idx], pa_old).statistic
                    if np.isfinite(ro):
                        old_r.append(ro)
            if not new_r:
                continue
            new_r = np.array(new_r)
            summary[m][L] = {"n_targets": len(new_r),
                             "spearman": float(new_r.mean()),
                             "spearman_sem": float(new_r.std(ddof=1) / np.sqrt(len(new_r))),
                             "kendall": float(np.mean(new_t)),
                             "frac_positive": float((new_r > 0).mean()),
                             "median_n_actives": int(np.median(ns)),
                             "spearman_old": float(np.mean(old_r)) if old_r else None}
            print("%-26s %-4s %7d %14s %14s %10.3f %8.0f%%" %
                  (m, L, len(new_r),
                   f"{new_r.mean():+.3f}±{new_r.std(ddof=1)/np.sqrt(len(new_r)):.3f}",
                   f"{np.mean(old_r):+.3f}" if old_r else "—",
                   float(np.mean(new_t)), (new_r > 0).mean() * 100))
    print("-" * 92)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=1)
    print(f"\n写入 {args.out}")
    print("旧口径对 ConGLUDe/ConPLex 是对的（它们本来就按 jsonl 顺序打分），"
          "两列应当基本一致；结构模型两列差多少，就是这个 bug 压低了多少。")


if __name__ == "__main__":
    main()
