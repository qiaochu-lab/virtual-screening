"""剔除「训练集里已有的 (靶点,分子) 对」之后重算 T3 主表。

为什么要做
----------
T3 只做了时间切分（取 2025+ 入库的记录），没做内容层面的差集。
check_pair_contamination.py 量出来：**L1 有 20.9% 的 pair 训练集里已经有了**
（L2 只有 0.01%，L3/L4 为 0）。也就是说 L1 这个对照层被系统性抬高，
主结论「L1→L4 衰减 64–77%」是个**上界**。

这里把污染的 active 从打分数组里直接删掉再重算，得到一个干净的下界。
不需要 GPU：所有模型的逐分子打分都已落盘，只是换一批下标重算指标。

分子顺序怎么对齐
----------------
各模型的输入构造方式不同，落盘顺序也就不同：
  · UniMol 系（DrugCLIP/BindCLIP/LigUnity/LiTENCLIP/HypSeek）
    读 data/T3_6A/{L}/{up}/{up}_lig.lmdb，缺构象的分子会被跳过 → 顺序是 lmdb 的
  · 其余（ConPLex/ConGLUDe/SPRINT）直接遍历评测集 jsonl → 顺序是 actives+decoys
脚本按长度自动判断用哪套顺序；两套都对不上就跳过这个靶点并计数，
**不猜**——猜错会把污染标到别的分子头上，比不做还糟。
"""
import argparse
import json
import os
import pickle
import sys

import lmdb
import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
B = "/data/work/vs-benchmark"
sys.path.insert(0, f"{B}/eval")
from metrics import bedroc, enrichment_factor, roc_auc   # noqa: E402

TD = f"{B}/code/LigUnity/test_datasets"
CACHE = f"{B}/data/t3/train_pairs.json"


def train_pairs():
    """训练集里的 (uniprot, inchikey) 对。算一次缓存下来，重跑就秒开。"""
    if os.path.exists(CACHE):
        d = json.load(open(CACHE))
        print(f"训练对（缓存）: {len(d):,}")
        return {tuple(x.split("\t")) for x in d}
    lab = json.load(open(f"{TD}/train_label_blend_seq_full.json"))
    pairs = set()
    for a in lab:
        up = a.get("uniprot")
        if not up:
            continue
        for l in a.get("ligands", []):
            smi = l.get("smi") if isinstance(l, dict) else None
            if not smi:
                continue
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            try:
                pairs.add((up, Chem.MolToInchiKey(m)))
            except Exception:
                pass
    json.dump(["\t".join(p) for p in sorted(pairs)], open(CACHE, "w"))
    print(f"训练对: {len(pairs):,}（已缓存）")
    return pairs


def lmdb_smis(path):
    if not os.path.exists(path):
        return None
    e = lmdb.open(path, subdir=False, readonly=True, lock=False)
    out = []
    with e.begin() as t:
        i = 0
        while True:
            raw = t.get(str(i).encode())
            if raw is None:
                break
            out.append(pickle.loads(raw)["smi"])
            i += 1
    e.close()
    return out


def contaminated_mask(up, L, n_pred, ikey_cache):
    """返回长度 n_pred 的布尔数组：True = 该分子与本靶点的组合训练集里已有。

    对不上顺序就返回 None，调用方跳过这个靶点。
    """
    rec = EVAL[L].get(up)
    if rec is None:
        return None
    jsonl_smis = [m["smiles"] for m in rec["actives"]] + [m["smiles"] for m in rec["decoys"]]
    n_act = len(rec["actives"])

    smis = None
    if len(jsonl_smis) == n_pred:
        smis, acts = jsonl_smis, set(range(n_act))
    else:
        ls = lmdb_smis(f"{B}/data/T3_6A/{L}/{up}/{up}_lig.lmdb")
        if ls is not None and len(ls) == n_pred:
            # lmdb 里 active 在前，个数从评测集的 active smiles 集合反推
            aset = {m["smiles"] for m in rec["actives"]}
            smis = ls
            acts = {i for i, s in enumerate(ls) if s in aset}
    if smis is None:
        return None

    mask = np.zeros(n_pred, dtype=bool)
    for i in acts:
        s = smis[i]
        ik = ikey_cache.get(s)
        if ik is None:
            m = Chem.MolFromSmiles(s)
            ik = Chem.MolToInchiKey(m) if m is not None else ""
            ikey_cache[s] = ik
        if ik and (up, ik) in TRAIN:
            mask[i] = True
    return mask


def metrics(p, y):
    return dict(ef1=enrichment_factor(p, y, 0.01),
                ef5=enrichment_factor(p, y, 0.05),
                bedroc=bedroc(p, y, 80.5),
                auroc=roc_auc(p, y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--layers", nargs="+", default=["L1", "L2", "L3", "L4"])
    ap.add_argument("--raw", default=f"{B}/results/t3_raw")
    args = ap.parse_args()

    ikey_cache = {}
    print("%-26s %-4s %6s %8s %8s %8s %8s %8s" %
          ("模型", "层", "靶点", "原EF1", "净EF1", "原AUROC", "净AUROC", "删掉分子"))
    print("-" * 92)
    summary = {}
    for m in args.models:
        for L in args.layers:
            # 两种落盘布局：UniMol 系是 t3_raw/<模型>/T3/<层>，
            # ConGLUDe/ConPLex 是 results/t3/<模型>/<层>
            d = f"{args.raw}/{m}/T3/{L}"
            if not os.path.isdir(d):
                d = f"{B}/results/t3/{m}/{L}"
            if not os.path.isdir(d):
                continue
            o, c, ndrop, nskip = [], [], 0, 0
            for up in sorted(os.listdir(d)):
                pp, lp = f"{d}/{up}/saved_preds.npy", f"{d}/{up}/saved_labels.npy"
                if not (os.path.exists(pp) and os.path.exists(lp)):
                    continue
                p, y = np.load(pp), np.load(lp)
                if p.ndim > 1:
                    p = p.reshape(-1)
                if len(p) != len(y) or y.sum() == 0:
                    continue
                o.append(metrics(p, y))
                mask = contaminated_mask(up, L, len(p), ikey_cache)
                if mask is None:
                    nskip += 1
                    c.append(o[-1])          # 对不上顺序：保守起见按原样计入
                    continue
                if not mask.any():
                    c.append(o[-1])
                    continue
                keep = ~mask
                ndrop += int(mask.sum())
                p2, y2 = p[keep], y[keep]
                c.append(metrics(p2, y2) if y2.sum() >= 5 else o[-1])
            if not o:
                continue
            f = lambda arr, k: float(np.mean([x[k] for x in arr]))
            summary[(m, L)] = (f(o, "ef1"), f(c, "ef1"))
            print("%-26s %-4s %6d %8.2f %8.2f %8.4f %8.4f %8d%s" %
                  (m, L, len(o), f(o, "ef1"), f(c, "ef1"),
                   f(o, "auroc"), f(c, "auroc"), ndrop,
                   f"  (顺序对不上 {nskip})" if nskip else ""))

    print("\n" + "=" * 92)
    print("L1→L4 衰减：污染剔除前 vs 剔除后")
    print("=" * 92)
    print("%-26s %14s %14s" % ("模型", "原始", "剔除污染后"))
    print("-" * 58)
    for m in args.models:
        a, b = summary.get((m, "L1")), summary.get((m, "L4"))
        if not a or not b:
            continue
        print("%-26s %13.1f%% %13.1f%%" %
              (m, (b[0] - a[0]) / a[0] * 100, (b[1] - a[1]) / a[1] * 100))
    print("\n注：L4 本来就没有污染，所以变化全部来自 L1 被拉低；"
          "剔除后的衰减是下界，原始值是上界。")


TRAIN = None
EVAL = None

if __name__ == "__main__":
    TRAIN = train_pairs()
    EVAL = {}
    for L in ["L1", "L2", "L3", "L4"]:
        p = f"{B}/data/t3/eval/{L}.jsonl"
        EVAL[L] = {}
        if os.path.exists(p):
            for line in open(p):
                r = json.loads(line)
                EVAL[L][r["uniprot"]] = r
    main()
