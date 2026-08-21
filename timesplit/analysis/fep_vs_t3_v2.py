"""配对检验重做：同一批靶点上，FEP 数据的排序能力 vs T3 数据的排序能力。

为什么重做
----------
原版 fep_vs_t3_same_targets.py 的 T3 一侧写的是
    act = np.nonzero(l2 == 1)[0]; spearman(p2[act], pa)
把「模型顺序里的第 k 个 active」当成「评测集里的第 k 个 active」。
对 UniMol 系模型不成立——它们读 lmdb，顺序是游标序（字典序），不是写入序。
于是 T3 一列被压到零，得出「同一批靶点上 FEP 能排、T3 不能排」的结论。

这版按**分子身份**（InChIKey）对齐再算，重新检验那个结论还成不成立。
"""
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
_ik = {}


def ikey(s):
    if s not in _ik:
        m = Chem.MolFromSmiles(s)
        try:
            _ik[s] = Chem.MolToInchiKey(m) if m is not None else ""
        except Exception:
            _ik[s] = ""
    return _ik[s]


def model_smiles(up, L, n, rec):
    jl = [m["smiles"] for m in rec["actives"]] + [m["smiles"] for m in rec["decoys"]]
    if len(jl) == n:
        return jl
    p = f"{B}/data/T3_6A/{L}/{up}/{up}_lig.lmdb"
    if not os.path.exists(p):
        return None
    e = lmdb.open(p, subdir=False, readonly=True, lock=False)
    out = []
    with e.begin() as t:
        for _k, v in t.cursor():
            out.append(pickle.loads(v)["smi"])
    e.close()
    return out if len(out) == n else None


def t3_rho(m, up):
    """按分子身份对齐后的 T3 排序相关；找不到返回 (None, 0)。"""
    for L in ["L1", "L2", "L3", "L4"]:
        rec = EV[L].get(up)
        if rec is None:
            continue
        for d in (f"{B}/results/t3_raw/{m}/T3/{L}/{up}", f"{B}/results/t3/{m}/{L}/{up}"):
            if not os.path.isdir(d):
                continue
            try:
                p = np.load(f"{d}/saved_preds.npy").reshape(-1)
                lab = np.load(f"{d}/saved_labels.npy")
            except Exception:
                continue
            if len(p) != len(lab):
                continue
            smis = model_smiles(up, L, len(p), rec)
            if smis is None:
                continue
            aff = {ikey(a["smiles"]): float(a["paff"]) for a in rec["actives"]}
            pairs = [(float(p[i]), aff[ikey(smis[i])])
                     for i in np.nonzero(lab == 1)[0] if ikey(smis[i]) in aff]
            if len(pairs) < MIN_ACT:
                continue
            sc = np.array([x[0] for x in pairs]); pa = np.array([x[1] for x in pairs])
            if np.std(sc) == 0 or np.std(pa) == 0:
                continue
            r = stats.spearmanr(sc, pa).statistic
            if np.isfinite(r):
                return r, len(pairs)
    return None, 0


FEP = f"{B}/code/LigUnity/test_datasets/FEP"
fep_ups = {}
for e in json.load(open(f"{FEP}/fep_labels.json")):
    fep_ups[e["uniprot"]] = e["pockets"][0]

EV = {}
for L in ["L1", "L2", "L3", "L4"]:
    p = f"{B}/data/t3/eval/{L}.jsonl"
    EV[L] = {json.loads(x)["uniprot"]: json.loads(x) for x in open(p)} if os.path.exists(p) else {}

print("同一批靶点：FEP 数据 vs T3 数据（T3 一侧已按分子身份对齐）")
print("=" * 74)
for m in ["ligunity_pocket_ranking", "ligunity_protein_ranking"]:
    print(f"\n【{m}】")
    print("  %-9s %-9s %8s %10s %8s %10s" % ("靶点", "体系", "FEP n", "FEP ρ", "T3 n", "T3 ρ"))
    print("  " + "-" * 60)
    fv, tv = [], []
    for up, pk in sorted(fep_ups.items()):
        d = f"{B}/results/fep/{m}/FEP/{pk}"
        try:
            pr = np.load(f"{d}/saved_preds.npy"); yy = np.load(f"{d}/saved_labels.npy")
            fr, fn = stats.spearmanr(pr, yy).statistic, len(yy)
        except Exception:
            continue
        tr, tn = t3_rho(m, up)
        if tr is None or not np.isfinite(fr):
            continue
        fv.append(fr); tv.append(tr)
        print("  %-9s %-9s %8d %+10.3f %8d %+10.3f" % (up, pk, fn, fr, tn, tr))
    if fv:
        print("  " + "-" * 60)
        print("  %-19s %8s %+10.3f %8s %+10.3f" % ("均值", "", np.mean(fv), "", np.mean(tv)))
        try:
            print(f"  配对 Wilcoxon p = {stats.wilcoxon(fv, tv).pvalue:.4f}  (n={len(fv)})")
        except ValueError:
            pass
print("\n判读：若 T3 一列不再接近零、配对差异变小，"
      "原来「靶点相同、换一批配体就排不出」的说法要收回或重写。")
