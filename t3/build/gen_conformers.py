"""为 T3 全部分子生成 3D 构象，供 UniMol 系模型（DrugCLIP/BindCLIP/LigUnity）使用。

这三个模型的分子塔是 UniMol，吃的是 3D 坐标而不是 SMILES，
所以 T3 的 146,776 个唯一分子都要先算构象。这一步与层、与模型都无关，
算一次全局复用。

构象生成沿用 UniMol 预处理的常规做法：
    ETKDGv3 嵌入 → MMFF94 优化（失败则退回 UFF，再失败则用未优化的嵌入构象）
只保留 1 个构象：DrugCLIP 官方 DUD-E/DEKOIS 数据里每个分子也只有 1 个
（`coordinates` 列表长度为 1），保持一致。

产出 `data/t3/mols/conformers.lmdb`，key = InChIKey，value = {atoms, coordinates, smi}。
后续按靶点组装 `{target}_lig.lmdb` 时直接取用并补上 label。
"""
import argparse
import hashlib
import json
import os
import pickle
from multiprocessing import Pool

import lmdb
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
B = "/data/yicheng/xqc/vs-benchmark"


def embed(args):
    ik, smi = args
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return ik, None, "SMILES 解析失败"
        m = Chem.AddHs(m)
        ps = AllChem.ETKDGv3()
        ps.randomSeed = 42                 # 固定种子，保证可复现
        ps.useSmallRingTorsions = True
        if AllChem.EmbedMolecule(m, ps) != 0:
            ps.useRandomCoords = True      # 大环/柔性分子的常规退路
            if AllChem.EmbedMolecule(m, ps) != 0:
                return ik, None, "嵌入失败"
        try:
            if AllChem.MMFFHasAllMoleculeParams(m):
                AllChem.MMFFOptimizeMolecule(m, maxIters=500)
            else:
                AllChem.UFFOptimizeMolecule(m, maxIters=500)
        except Exception:
            pass                           # 优化失败就用未优化的嵌入构象
        m = Chem.RemoveHs(m)
        conf = m.GetConformer()
        coords = np.array([list(conf.GetAtomPosition(i))
                           for i in range(m.GetNumAtoms())], dtype=np.float32)
        atoms = [a.GetSymbol() for a in m.GetAtoms()]
        if len(atoms) == 0 or not np.isfinite(coords).all():
            return ik, None, "坐标异常"
        return ik, {"atoms": atoms, "coordinates": [coords], "smi": smi}, None
    except Exception as e:                 # noqa: BLE001 逐条容错，不中断整批
        return ik, None, f"{type(e).__name__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--out", default=f"{B}/data/t3/mols/conformers.lmdb")
    ap.add_argument("--chunk", type=int, default=200)
    args = ap.parse_args()

    # 汇总全局唯一分子
    seen = {}
    for L in ["L1", "L2", "L3", "L4"]:
        p = f"{B}/data/t3/layers/{L}.jsonl"
        if not os.path.exists(p):
            continue
        for line in open(p):
            d = json.loads(line)
            ik = d.get("inchikey")
            # 少数记录 InChIKey 为空：SMILES 里带 `->` 配位键或 `C:C` 这类
            # RDKit 解析不了的写法，建库时就没生成出来。空 key 会让 LMDB 直接
            # 报 MDB_BAD_VALSIZE，这里按 SMILES 兜底成一个稳定的 key。
            if not ik:
                ik = "NOKEY_" + hashlib.md5(d["smiles"].encode()).hexdigest()
            seen.setdefault(ik, d["smiles"])
    items = sorted(seen.items())
    print(f"唯一分子: {len(items):,}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if os.path.exists(args.out):
        os.remove(args.out)
    env = lmdb.open(args.out, subdir=False, map_size=1 << 40)

    from collections import Counter
    fails = Counter()
    n_ok = 0
    # 分批提交：单个大事务一旦中途出错，之前算的全部丢失（第一次就栽在这）
    txn = env.begin(write=True)
    with Pool(args.procs) as pool:
        for i, (ik, rec, err) in enumerate(
                pool.imap_unordered(embed, items, chunksize=args.chunk)):
            if rec is None:
                fails[err] += 1
            else:
                txn.put(ik.encode(), pickle.dumps(rec))
                n_ok += 1
            if (i + 1) % 10000 == 0:
                txn.commit()
                txn = env.begin(write=True)
                print(f"  {i+1:,}/{len(items):,}  成功 {n_ok:,}  失败 {sum(fails.values()):,}",
                      flush=True)
    txn.commit()
    env.close()

    print(f"\n成功 {n_ok:,} / {len(items):,}  ({n_ok/len(items)*100:.1f}%)")
    for k, v in fails.most_common():
        print(f"  {k}: {v:,}")
    print(f"已写入 {args.out}")


if __name__ == "__main__":
    main()
