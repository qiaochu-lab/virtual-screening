"""T6 的第二个物理方法：分子对接（smina）。

为什么必须有它
--------------
T6 现在的物理侧只有 Boltz-2 一家，所有结论都是「Boltz-2 如何」而不是
「物理方法如何」。而且 Boltz-2 每个复合物几十秒，撑不起两件事：
  · physics-only 的全量筛选（T3 每靶点上千个分子）
  · 深 shortlist 的 rerank —— recall@50 在 L4 只有 17.5%，要往深了排就得便宜
smina 每个分子几秒，正好补这两个位置。

盒子怎么定
----------
直接用我们已经提好的 6Å 口袋：取口袋所有原子坐标的包围盒，各方向留 4Å 余量。
这样**对接和检索模型看到的是同一个口袋**，比较才成立——
如果用 fpocket 之类另外找一遍口袋，比较的就变成「口袋定义」了。

受体准备
--------
口袋 lmdb 里只有原子类型和坐标，没有连接和电荷。smina 能直接读 PDB，
所以从口袋原子写出 PDB，再用 obabel 加氢、赋 Gasteiger 电荷转 pdbqt。
⚠️ 这是「口袋切片对接」，不是全蛋白对接：切片边缘的残基缺少邻接约束，
打分会有偏差。但对**同一批分子的相对排序**影响有限，而这正是我们要测的量。
"""
import argparse
import json
import os
import pickle
import subprocess

import lmdb
import numpy as np

B = "/data/work/vs"
DOCK = "/data/work/envs/dock/bin"
PAD = 4.0


def write_pocket_pdb(rec, path):
    """口袋原子 -> PDB。residue 信息已丢失，统一记为 POC/A/1，smina 只用坐标和元素。"""
    coords = np.asarray(rec["pocket_coordinates"], dtype=float)
    atoms = rec["pocket_atoms"]
    with open(path, "w") as f:
        for i, (a, c) in enumerate(zip(atoms, coords), 1):
            el = "".join(ch for ch in a if ch.isalpha())[:1] or "C"
            f.write(f"ATOM  {i:5d} {a[:4]:<4s} POC A   1    "
                    f"{c[0]:8.3f}{c[1]:8.3f}{c[2]:8.3f}  1.00  0.00          {el:>2s}\n")
        f.write("END\n")
    return coords


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default="L4")
    ap.add_argument("--targets", type=int, default=20)
    ap.add_argument("--topn", type=int, default=200,
                    help="每靶点取检索模型的 top-N 送对接（深于 Boltz-2 那轮的 50）")
    ap.add_argument("--model", default="ligunity_protein_ranking")
    ap.add_argument("--out", default=f"{B}/dock")
    args = ap.parse_args()

    pockets = {}
    for pref in ("pocket", "pdb_pocket"):
        p = f"{B}/data/t3/pockets/{pref}_6.0A.lmdb"
        if not os.path.exists(p):
            continue
        e = lmdb.open(p, subdir=False, readonly=True, lock=False)
        with e.begin() as t:
            for _k, v in t.cursor():
                d = pickle.loads(v)
                pockets[d["pocket"]] = d
        e.close()

    ev = {json.loads(x)["uniprot"]: json.loads(x)
          for x in open(f"{B}/data/t3/eval/{args.layer}.jsonl")}
    root = f"{B}/results/t3_raw/{args.model}/T3/{args.layer}"
    hq = set(json.load(open(f"{B}/data/t3/target_quality.json"))["high_quality"])

    picked, manifest = [], []
    for up in sorted(os.listdir(root)):
        if up not in pockets or up not in ev or up not in hq:
            continue
        try:
            pr = np.load(f"{root}/{up}/saved_preds.npy").reshape(-1)
            y = np.load(f"{root}/{up}/saved_labels.npy")
        except Exception:
            continue
        if len(pr) != len(y) or y.sum() < 5:
            continue
        picked.append((up, pr, y))
        if len(picked) >= args.targets:
            break
    print(f"选中靶点 {len(picked)}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    n_lig = 0
    for up, pr, y in picked:
        d = f"{args.out}/{up}"
        os.makedirs(d, exist_ok=True)
        coords = write_pocket_pdb(pockets[up], f"{d}/pocket.pdb")
        lo, hi = coords.min(0) - PAD, coords.max(0) + PAD
        center, size = (lo + hi) / 2, hi - lo
        # 受体 pdbqt
        subprocess.run([f"{DOCK}/obabel", f"{d}/pocket.pdb", "-O", f"{d}/pocket.pdbqt",
                        "-xr", "-p", "7.4"], capture_output=True)
        # 配体：按检索分数取 top-N，顺序保留以便和检索原序比较
        rec = ev[up]
        smis = [m["smiles"] for m in rec["actives"]] + [m["smiles"] for m in rec["decoys"]]
        if len(smis) != len(pr):
            e = lmdb.open(f"{B}/data/T3_6A/{args.layer}/{up}/{up}_lig.lmdb",
                          subdir=False, readonly=True, lock=False)
            smis = []
            with e.begin() as t:
                for _k, v in t.cursor():
                    smis.append(pickle.loads(v)["smi"])
            e.close()
        if len(smis) != len(pr):
            print(f"  {up} 顺序对不上，跳过")
            continue
        top = np.argsort(-pr)[:args.topn]
        rows = [{"idx": int(i), "rank": r, "smiles": smis[i], "label": int(y[i]),
                 "retrieval_score": float(pr[i])} for r, i in enumerate(top)]
        with open(f"{d}/ligands.smi", "w") as f:
            for j, x in enumerate(rows):
                f.write(f"{x['smiles']}\tlig{j}\n")
        json.dump({"uniprot": up, "center": center.tolist(), "size": size.tolist(),
                   "ligands": rows}, open(f"{d}/manifest.json", "w"))
        manifest.append(up)
        n_lig += len(rows)

    json.dump({"layer": args.layer, "model": args.model, "topn": args.topn,
               "targets": manifest}, open(f"{args.out}/manifest.json", "w"), indent=1)
    print(f"准备完成：{len(manifest)} 个靶点，{n_lig:,} 个配体待对接")
    print(f"盒子取自 6Å 口袋的包围盒 + {PAD}Å 余量——与检索模型同一个口袋")


if __name__ == "__main__":
    main()
