"""T2 的第三种解释：排序能力的零，是不是被标签噪声压出来的？

问题从哪来
----------
T2 现在有三套数据，结论对不上：
  · FEP 基准（同一化学系列）        ρ ≈ 0.40
  · CASF-2016（同靶点、跨骨架）     ρ = 0.42   ← 新测的
  · 自建 T3（跨库抽取）             ρ ≈ 0
原来的解释是「同系列内能排、跨系列归零」。**CASF 把这个解释推翻了**——
CASF 一个簇里五个配体骨架各不相同，照样能排到 0.42。

那 T3 的零还剩什么解释？最大嫌疑是**标签本身**：
T3 的 pAffinity 把 Ki / Kd / IC50 / EC50 跨实验室、跨测定格式混在一起，
而 FEP 和 CASF 的数值是同一批测定、口径一致的。
IC50 依赖底物浓度，和 Ki 之间差一个体系相关的常数——混在一起排序，
噪声可能直接盖掉信号。

怎么验（不用 GPU，打分都在盘上，只是换一批下标重算）
----------------------------------------------------
按标签洁净程度做三档，逐靶点算 Spearman 再对靶点平均：
  ①  全部 active                      —— 现状
  ②  只留该靶点占比最大的那种测定类型  —— 去掉类型混用
  ③  只留该靶点最大的**单个 assay_id** —— 同一次实验、同一个实验室，最干净
③ 是最接近 FEP/CASF 条件的一档。std_type 和 assay_id 只有 ChEMBL 那部分记录有，
所以 ②③ 只在 ChEMBL 来源的 active 上做。

怎么判读
--------
· ρ 随洁净度单调上升 → 「模型排不出强弱」要改写成「T3 的标签噪声掩盖了排序能力」
· ρ 一直是零         → 现有结论反而更硬，最后一个数据端解释也排除了
· ③ 的 n 会变小，Spearman 方差变大 —— 所以同时报每档的靶点数和配体数中位
"""
import argparse
import json
import os
import pickle
import sys
from collections import Counter, defaultdict

import lmdb
import numpy as np
from rdkit import Chem, RDLogger
from scipy import stats

RDLogger.DisableLog("rdApp.*")
B = "/data/work/vs-benchmark"
MIN_LIG = 5          # 少于 5 个配体的靶点算不出有意义的 Spearman


def chembl_index():
    """(uniprot, inchikey) -> [(std_type, assay_id)]，只有 ChEMBL 记录有这些字段。"""
    idx = defaultdict(list)
    p = f"{B}/data/t3/chembl37_2025plus.jsonl"
    for line in open(p):
        d = json.loads(line)
        m = Chem.MolFromSmiles(d["smiles"])
        if m is None:
            continue
        try:
            ik = Chem.MolToInchiKey(m)
        except Exception:
            continue
        idx[(d["uniprot"], ik)].append((d.get("std_type"), d.get("assay_id")))
    print(f"ChEMBL 记录索引: {len(idx):,} 个 (靶点,分子) 组合")
    return idx


def mol_order(up, L, n_pred, eval_rec):
    """还原模型看到的分子顺序 -> [(inchikey, paff or None)]，对不上返回 None。

    两种布局：UniMol 系读 lmdb（缺构象的分子被跳过），其余直接遍历 jsonl。
    对不上就返回 None，不猜——猜错会把配体和亲和力错配，比不做还糟。
    """
    acts = eval_rec["actives"]
    jsonl = [(m["inchikey"], m["paff"]) for m in acts] + \
            [(m["inchikey"], None) for m in eval_rec["decoys"]]
    if len(jsonl) == n_pred:
        return jsonl
    p = f"{B}/data/T3_6A/{L}/{up}/{up}_lig.lmdb"
    if not os.path.exists(p):
        return None
    # 必须按游标序读：key 是字符串，模型侧遍历得到的是字典序
    # （0, 1, 10, 100, ...），不是数值序。按数值下标读会整体错位。
    e = lmdb.open(p, subdir=False, readonly=True, lock=False)
    smis = []
    with e.begin() as t:
        for _k, v in t.cursor():
            smis.append(pickle.loads(v)["smi"])
    e.close()
    if len(smis) != n_pred:
        return None
    by_smi = {m["smiles"]: (m["inchikey"], m["paff"]) for m in acts}
    return [by_smi.get(s, (None, None)) for s in smis]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--layers", nargs="+", default=["L1", "L2", "L3", "L4"])
    args = ap.parse_args()

    CH = chembl_index()
    EV = {}
    for L in args.layers:
        p = f"{B}/data/t3/eval/{L}.jsonl"
        EV[L] = {json.loads(x)["uniprot"]: json.loads(x) for x in open(p)} \
            if os.path.exists(p) else {}

    print("\n%-26s %-4s %26s %26s %26s" %
          ("模型", "层", "① 全部 active", "② 单一测定类型", "③ 单个 assay"))
    print("%-26s %-4s %8s %8s %8s %8s %8s %8s %8s %8s %8s" %
          ("", "", "靶点", "配体中位", "ρ", "靶点", "配体中位", "ρ", "靶点", "配体中位", "ρ"))
    print("-" * 112)

    for m in args.models:
        for L in args.layers:
            d = f"{B}/results/t3_raw/{m}/T3/{L}"
            if not os.path.isdir(d):
                d = f"{B}/results/t3/{m}/{L}"
            if not os.path.isdir(d):
                continue
            res = {k: {"rho": [], "n": []} for k in "123"}
            for up in sorted(os.listdir(d)):
                try:
                    p = np.load(f"{d}/{up}/saved_preds.npy").reshape(-1)
                    y = np.load(f"{d}/{up}/saved_labels.npy")
                except Exception:
                    continue
                rec = EV.get(L, {}).get(up)
                if rec is None or len(p) != len(y):
                    continue
                order = mol_order(up, L, len(p), rec)
                if order is None:
                    continue

                # 所有有实测亲和力的 active
                items = [(i, ik, float(a)) for i, (ik, a) in enumerate(order)
                         if a is not None and ik]
                if len(items) < MIN_LIG:
                    continue

                def rho(sub):
                    if len(sub) < MIN_LIG:
                        return None
                    sc = np.array([p[i] for i, _, _ in sub])
                    aff = np.array([a for _, _, a in sub])
                    if np.std(sc) == 0 or np.std(aff) == 0:
                        return None
                    r = stats.spearmanr(sc, aff).statistic
                    return None if np.isnan(r) else r

                r1 = rho(items)
                if r1 is not None:
                    res["1"]["rho"].append(r1); res["1"]["n"].append(len(items))

                # ② 该靶点占比最大的测定类型
                types = Counter()
                for _, ik, _ in items:
                    for t, _a in CH.get((up, ik), []):
                        if t:
                            types[t] += 1
                if types:
                    top_t = types.most_common(1)[0][0]
                    sub = [x for x in items
                           if any(t == top_t for t, _ in CH.get((up, x[1]), []))]
                    r2 = rho(sub)
                    if r2 is not None:
                        res["2"]["rho"].append(r2); res["2"]["n"].append(len(sub))

                # ③ 该靶点最大的单个 assay
                assays = Counter()
                for _, ik, _ in items:
                    for _t, a in CH.get((up, ik), []):
                        if a:
                            assays[a] += 1
                if assays:
                    top_a = assays.most_common(1)[0][0]
                    sub = [x for x in items
                           if any(a == top_a for _, a in CH.get((up, x[1]), []))]
                    r3 = rho(sub)
                    if r3 is not None:
                        res["3"]["rho"].append(r3); res["3"]["n"].append(len(sub))

            if not res["1"]["rho"]:
                continue
            cells = []
            for k in "123":
                v = res[k]
                cells.append("%8d %8.0f %8.3f" % (len(v["rho"]), np.median(v["n"]),
                                                 np.mean(v["rho"])) if v["rho"]
                             else "%8s %8s %8s" % ("-", "-", "-"))
            print("%-26s %-4s %s" % (m, L, " ".join(cells)))

    print("-" * 112)
    print("\n判读：ρ 若随 ①→②→③ 单调上升，说明 T3 的零主要是标签噪声；"
          "若三档都接近零，则最后一个数据端解释也被排除。")
    print("注意 ③ 的配体数明显更少，Spearman 方差更大——看趋势，别看单个格子。")


if __name__ == "__main__":
    main()
