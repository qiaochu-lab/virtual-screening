"""续跑构象生成：补齐 conformers.lmdb 里缺的分子，带**每分子超时**。

为什么需要
----------
第一版没有超时保护。RDKit 的 ETKDG 在个别病态分子（大环、大肽、
高度对称的笼状结构）上会无限自旋——实测有 3 个 worker 以 99% CPU
空转近 3 小时卡在同一批分子上，把整个任务拖住。

这里改成：每个分子起一个子进程算，超过 TIMEOUT 秒直接放弃并记下来。
放弃的分子会被排除出评测集（在 build_t3_unimol.py 里按缺构象处理），
数量很少且都是真正算不动的，不影响结论。

已落盘的 139,882 个不重算（第一版改成分批提交后保住了）。
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import pickle

import lmdb
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
B = "/data/work/vs-benchmark"


def _embed(smi, q):
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            q.put(("fail", "SMILES 解析失败")); return
        m = Chem.AddHs(m)
        ps = AllChem.ETKDGv3()
        ps.randomSeed = 42
        ps.useSmallRingTorsions = True
        if AllChem.EmbedMolecule(m, ps) != 0:
            ps.useRandomCoords = True
            if AllChem.EmbedMolecule(m, ps) != 0:
                q.put(("fail", "嵌入失败")); return
        try:
            if AllChem.MMFFHasAllMoleculeParams(m):
                AllChem.MMFFOptimizeMolecule(m, maxIters=500)
            else:
                AllChem.UFFOptimizeMolecule(m, maxIters=500)
        except Exception:
            pass
        m = Chem.RemoveHs(m)
        conf = m.GetConformer()
        coords = np.array([list(conf.GetAtomPosition(i))
                           for i in range(m.GetNumAtoms())], dtype=np.float32)
        atoms = [a.GetSymbol() for a in m.GetAtoms()]
        if not atoms or not np.isfinite(coords).all():
            q.put(("fail", "坐标异常")); return
        q.put(("ok", {"atoms": atoms, "coordinates": [coords], "smi": smi}))
    except Exception as e:                     # noqa: BLE001
        q.put(("fail", type(e).__name__))


def embed_with_timeout(smi, timeout):
    """每个分子起独立子进程，超时就杀 —— 这是唯一能中断 RDKit C++ 循环的办法。"""
    q = mp.Queue()
    p = mp.Process(target=_embed, args=(smi, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join(1)
        if p.is_alive():
            p.kill()
        return None, "超时"
    try:
        kind, val = q.get_nowait()
    except Exception:
        return None, "子进程无返回"
    return (val, None) if kind == "ok" else (None, val)


def worker(task):
    ik, smi, timeout = task
    rec, err = embed_with_timeout(smi, timeout)
    return ik, rec, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=45, help="单分子秒数上限")
    ap.add_argument("--out", default=f"{B}/data/t3/mols/conformers.lmdb")
    args = ap.parse_args()

    seen = {}
    for L in ["L1", "L2", "L3", "L4"]:
        p = f"{B}/data/t3/layers/{L}.jsonl"
        if not os.path.exists(p):
            continue
        for line in open(p):
            d = json.loads(line)
            ik = d.get("inchikey") or ("NOKEY_" + hashlib.md5(d["smiles"].encode()).hexdigest())
            seen.setdefault(ik, d["smiles"])

    env = lmdb.open(args.out, subdir=False, map_size=1 << 40)
    with env.begin() as t:
        have = {k.decode() for k, _ in t.cursor()}
    todo = [(ik, smi, args.timeout) for ik, smi in sorted(seen.items()) if ik not in have]
    print(f"总分子 {len(seen):,}；已有 {len(have):,}；待补 {len(todo):,}", flush=True)
    if not todo:
        env.close()
        print("无需续跑")
        return

    from collections import Counter
    from concurrent.futures import ProcessPoolExecutor
    fails = Counter()
    n_ok = 0
    txn = env.begin(write=True)
    # 必须用 ProcessPoolExecutor 而不是 mp.Pool：Pool 的 worker 是守护进程，
    # 不允许再起子进程，而「每分子一个可杀的子进程」正是超时控制的唯一可靠办法
    # （RDKit 在 C++ 里自旋时，signal.alarm 之类的手段打断不了）。
    with ProcessPoolExecutor(args.procs) as pool:
        for i, (ik, rec, err) in enumerate(pool.map(worker, todo, chunksize=1)):
            if rec is None:
                fails[err] += 1
            else:
                txn.put(ik.encode(), pickle.dumps(rec))
                n_ok += 1
            if (i + 1) % 500 == 0:
                txn.commit()
                txn = env.begin(write=True)
                print(f"  {i+1:,}/{len(todo):,}  成功 {n_ok:,}  失败 {sum(fails.values()):,}",
                      flush=True)
    txn.commit()
    with env.begin() as t:
        total = env.stat()["entries"]
    env.close()

    print(f"\n续跑成功 {n_ok:,} / {len(todo):,}")
    for k, v in fails.most_common():
        print(f"  {k}: {v:,}")
    print(f"构象库总数: {total:,} / {len(seen):,}  ({total/len(seen)*100:.1f}%)")
    print("已写入", args.out)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)   # fork 会继承 RDKit 状态，spawn 更干净
    main()
