"""导出「剔除训练集已有 pair」之后的干净 T3 主表。

为什么值得做成正式产物
----------------------
T3 的分层只做了时间切分，L1 里 20.9% 的 (靶点,分子) 对训练集中已存在。
之前 score_t3_clean.py 把这件事当**一次性稳健性检验**跑过，
但主表报的仍是含污染的数字，读者只能在 LIMITATIONS 里看到一句「是上界」。
这里把干净版做成与主表并列的 CSV，让「衰减 64–81%」变成一个区间而非单点。

口径与限制
----------
· 只删「(靶点,分子) 对在训练集里已有」的 active，decoy 不动
· 因此删完后 active:decoy 比例会从 1:50 略微变化（active 变少），
  EF 的分母随之改变——这是与「重新构建评测集并重跑推理」的唯一差别。
  真正重建需要九个模型全部重跑推理，代价与收益不成比例，故采用删下标的方式，
  并在此写明差别。
· L3/L4 污染为 0，所以那两层的数字与主表完全相同，可用作自检。
"""
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
from metrics import bedroc, enrichment_factor, roc_auc  # noqa: E402

MODELS = ["drugclip", "bindclip_randneg", "bindclip_hardneg",
          "ligunity_pocket_ranking", "ligunity_protein_ranking",
          "litenclip", "hypseek_rk", "conglude", "conplex", "sprint"]


def train_pairs():
    p = f"{B}/data/t3/train_pairs.json"
    return {tuple(x.split("\t")) for x in json.load(open(p))}


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
        for _k, v in t.cursor():          # 游标序才是模型看到的顺序
            out.append(pickle.loads(v)["smi"])
    e.close()
    return out if len(out) == n else None


def main():
    TRAIN = train_pairs()
    cache = {}
    EV = {}
    for L in ["L1", "L2", "L3", "L4"]:
        p = f"{B}/data/t3/eval/{L}.jsonl"
        EV[L] = {json.loads(x)["uniprot"]: json.loads(x) for x in open(p)} if os.path.exists(p) else {}

    rows = ["model,layer,n_targets,n_actives_removed,EF1_raw,EF1_clean,BEDROC_raw,BEDROC_clean,AUROC_raw,AUROC_clean"]
    print("%-26s %-4s %7s %9s %16s %16s" % ("模型", "层", "靶点", "删掉", "EF1 原→净", "AUROC 原→净"))
    print("-" * 88)
    for m in MODELS:
        for L in ["L1", "L2", "L3", "L4"]:
            d = f"{B}/results/t3_raw/{m}/T3/{L}"
            if not os.path.isdir(d):
                d = f"{B}/results/t3/{m}/{L}"
            if not os.path.isdir(d):
                continue
            raw, clean, ndrop = [], [], 0
            for up in sorted(os.listdir(d)):
                try:
                    p = np.load(f"{d}/{up}/saved_preds.npy").reshape(-1)
                    y = np.load(f"{d}/{up}/saved_labels.npy")
                except Exception:
                    continue
                if len(p) != len(y) or y.sum() == 0:
                    continue
                r = dict(ef1=enrichment_factor(p, y, 0.01), bedroc=bedroc(p, y, 80.5),
                         auroc=roc_auc(p, y))
                raw.append(r)
                rec = EV[L].get(up)
                smis = model_smiles(up, L, len(p), rec) if rec else None
                if smis is None:
                    clean.append(r)          # 对不上顺序：按原样计入，不猜
                    continue
                mask = np.zeros(len(p), dtype=bool)
                for i in np.nonzero(y == 1)[0]:
                    s = smis[i]
                    if s not in cache:
                        mm = Chem.MolFromSmiles(s)
                        try:
                            cache[s] = Chem.MolToInchiKey(mm) if mm is not None else ""
                        except Exception:
                            cache[s] = ""
                    if cache[s] and (up, cache[s]) in TRAIN:
                        mask[i] = True
                if not mask.any():
                    clean.append(r)
                    continue
                ndrop += int(mask.sum())
                k = ~mask
                p2, y2 = p[k], y[k]
                clean.append(dict(ef1=enrichment_factor(p2, y2, 0.01),
                                  bedroc=bedroc(p2, y2, 80.5), auroc=roc_auc(p2, y2))
                             if y2.sum() >= 5 else r)
            if not raw:
                continue
            f = lambda a, k: float(np.mean([x[k] for x in a]))
            print("%-26s %-4s %7d %9d %16s %16s" %
                  (m, L, len(raw), ndrop,
                   f"{f(raw,'ef1'):.2f} → {f(clean,'ef1'):.2f}",
                   f"{f(raw,'auroc'):.4f} → {f(clean,'auroc'):.4f}"))
            rows.append(f"{m},{L},{len(raw)},{ndrop},{f(raw,'ef1'):.4f},{f(clean,'ef1'):.4f},"
                        f"{f(raw,'bedroc'):.4f},{f(clean,'bedroc'):.4f},"
                        f"{f(raw,'auroc'):.4f},{f(clean,'auroc'):.4f}")
    out = f"{B}/results/export/T3_main_clean.csv"
    open(out, "w").write("\n".join(rows) + "\n")
    print(f"\n写入 {out}")
    print("L3/L4 两列应完全相同（那两层污染为 0），可作自检")


if __name__ == "__main__":
    main()
