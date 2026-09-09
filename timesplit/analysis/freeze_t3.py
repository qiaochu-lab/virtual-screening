"""冻结 T3 的逐靶点逐分子中间表，让未来任何一次重新分层都只是重新汇总。

为什么要有这个
--------------
「这个靶点模型见没见过」不是客观事实，是随模型训练集而变的标签。所以 L1–L4
必须按模型重画，而且不止一次——本项目已经重画过三次（fall-through bug、
逐模型分层、结构半/亲和力半并集）。每次重画要多久，取决于保留了什么：

· 只留汇总表 → 得重跑模型，十个模型一天 GPU
· 留逐靶点逐分子的中间表 → 只是重新 group by，几分钟纯 CPU

三张表
------
1. `T3_molecules.csv.gz`  唯一分子：mol_id, inchikey, smiles, 两套新颖度
2. `T3_index.csv.gz`      逐靶点逐分子：层、靶点、**两种顺序下的位置**、
                          mol_id、标签、pAffinity
3. `T3_model_order.csv`   每个模型每个靶点用的是哪种顺序，以及是否通过硬校验

**第 2 张里的 jsonl_pos / lmdb_pos 是这套东西的核心。** 模型读 lmdb 时游标是
字典序（0, 1, 10, 100, …），与评测集 jsonl 的「活性+诱饵」顺序不同而长度相同；
只比长度会静默错配，本项目因此毁过一次 T2 的全部结论、并在另外两处重现
（PATCHES.md）。把两种顺序的位置一起存死，以后任何重算都不必再推导一次。
"""
import argparse, csv, gzip, json, os, pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

B = "/data/work/vs-benchmark"
LAYERS = ("L1", "L2", "L3", "L4")


def ikey(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return ""
    try:
        return Chem.MolToInchiKey(m)
    except Exception:
        return ""


def lmdb_order(up, L):
    p = f"{B}/data/T3_6A/{L}/{up}/{up}_lig.lmdb"
    if not os.path.exists(p):
        return None
    try:
        import lmdb
        e = lmdb.open(p, subdir=False, readonly=True, lock=False)
        out = []
        with e.begin() as t:
            for _k, v in t.cursor():
                out.append(pickle.loads(v)["smi"])
        e.close()
        return out
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=f"{B}/results/frozen")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--models", nargs="+", default=[
        "hypseek_rk", "ligunity_pocket_ranking", "ligunity_protein_ranking",
        "litenclip", "drugclip", "bindclip_randneg", "bindclip_hardneg",
        "conglude", "conplex", "sprint"])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    recs = {L: [json.loads(x) for x in open(f"{B}/data/t3/eval/{L}.jsonl")]
            for L in LAYERS}
    nov_p = json.load(open(f"{B}/data/t3/ligand_novelty.json"))
    nov_d = json.load(open(f"{B}/data/t3/ligand_novelty_drugclip.json"))

    smis = sorted({x["smiles"] for L in LAYERS for r in recs[L]
                   for g in ("actives", "decoys") for x in r[g]})
    print(f"唯一分子 {len(smis):,}，算 InChIKey…", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        keys = list(ex.map(ikey, smis, chunksize=500))
    mid = {s: i for i, s in enumerate(smis)}

    with gzip.open(f"{args.out_dir}/T3_molecules.csv.gz", "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mol_id", "inchikey", "smiles",
                    "novelty_pocketaffdb", "novelty_drugclip"])
        for i, s in enumerate(smis):
            w.writerow([i, keys[i], s,
                        f"{nov_p.get(s, -1):.4f}", f"{nov_d.get(s, -1):.4f}"])
    print(f"  写入 T3_molecules.csv.gz")

    # ---- 逐靶点逐分子，两种顺序的位置一起存 ----
    n_rows = 0
    n_nolmdb = 0
    with gzip.open(f"{args.out_dir}/T3_index.csv.gz", "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "uniprot", "jsonl_pos", "lmdb_pos",
                    "mol_id", "label", "paff"])
        for L in LAYERS:
            for r in recs[L]:
                up = r["uniprot"]
                jl = [(x["smiles"], 1, x.get("paff")) for x in r["actives"]] + \
                     [(x["smiles"], 0, None) for x in r["decoys"]]
                # ⚠️ 不要求 len(lmdb) == len(jsonl)。建 lmdb 时有分子会被丢掉
                # （RDKit 解析或构象生成失败），546 个靶点里 413 个的 jsonl 池子
                # 比 lmdb 多 1~9 个分子。**lmdb 才是模型实际打过分的那一批**，
                # 所以按 SMILES 对位；jsonl 里有而 lmdb 里没有的记 -1，
                # 这本身就是这张表要留的信息。
                lo = lmdb_order(up, L)
                lpos = {}
                if lo is not None:
                    for i, s in enumerate(lo):
                        lpos.setdefault(s, i)
                else:
                    n_nolmdb += 1
                for j, (s, lab, pa) in enumerate(jl):
                    w.writerow([L, up, j, lpos.get(s, -1), mid[s], lab,
                                "" if pa is None else f"{float(pa):.3f}"])
                    n_rows += 1
            print(f"  {L} 完成，累计 {n_rows:,} 行", flush=True)
    print(f"写入 T3_index.csv.gz（{n_rows:,} 行；{n_nolmdb} 个靶点没有可用 lmdb）")

    # ---- 每个模型每个靶点实际用的顺序 ----
    with open(f"{args.out_dir}/T3_model_order.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "layer", "uniprot", "n_molecules", "order_used"])
        for m in args.models:
            cnt = {"jsonl": 0, "lmdb": 0, "FAIL": 0}
            for L in LAYERS:
                d = f"{B}/results/t3_raw/{m}/T3/{L}"
                if not os.path.isdir(d):
                    d = f"{B}/results/t3/{m}/{L}"
                if not os.path.isdir(d):
                    continue
                for r in recs[L]:
                    up = r["uniprot"]
                    try:
                        y = np.load(f"{d}/{up}/saved_labels.npy")
                    except Exception:
                        continue
                    act = {x["smiles"] for x in r["actives"]}
                    jl = [x["smiles"] for x in r["actives"]] + \
                         [x["smiles"] for x in r["decoys"]]
                    used = "FAIL"
                    if len(jl) == len(y) and \
                       {jl[i] for i in range(len(y)) if y[i] == 1} == act:
                        used = "jsonl"
                    else:
                        lo = lmdb_order(up, L)
                        if lo is not None and len(lo) == len(y) and \
                           {lo[i] for i in range(len(y)) if y[i] == 1} == act:
                            used = "lmdb"
                    cnt[used] += 1
                    w.writerow([m, L, up, len(y), used])
            print(f"  {m}: jsonl {cnt['jsonl']} / lmdb {cnt['lmdb']} / "
                  f"验不过 {cnt['FAIL']}", flush=True)
    print(f"写入 T3_model_order.csv")


if __name__ == "__main__":
    main()
