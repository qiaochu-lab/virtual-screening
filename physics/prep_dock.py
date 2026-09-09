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
        # 原子名和坐标长度不等必须炸——静默截断会写出一个残缺的口袋 PDB，
        # 而对接照样能跑完，只是盒子和受体都错了。
        for i, (a, c) in enumerate(zip(atoms, coords, strict=True), 1):
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
    ap.add_argument("--subset", default=None,
                    help="csv：只从这批靶点里选（例如 350 配额子集）")
    ap.add_argument("--seed", type=int, default=1,
                    help="靶点抽样种子。不给 --subset 时保持旧行为（字母序取前 N）")
    ap.add_argument("--max-box", type=float, default=0.0,
                    help="盒子体积上限 Å³，0 = 不限。"
                         "大盒子会撞 run_dock.sh 的 90 分钟上限、一个分数都出不来，"
                         "所以选靶点时就该排除，而不是跑完才发现")
    ap.add_argument("--report-boxes", action="store_true",
                    help="只打印候选靶点的盒子体积分布，不生成任何文件")
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

    keep = None
    if args.subset:
        import csv as _csv
        keep = {r["uniprot"] for r in _csv.DictReader(open(args.subset))
                if r.get("layer") == args.layer}
        print(f"限定在子集的 {args.layer}：{len(keep)} 个靶点")

    # 合格靶点先全收，再抽样。旧行为是 sorted() 取前 N —— 那是**字母序**，
    # 不是随机也不是按质量，等于按 UniProt 号选靶点。给了 --subset 就改成
    # 固定种子随机抽，避免这个偏差。
    cand = []
    for up in sorted(os.listdir(root)):
        if up not in pockets or up not in ev or up not in hq:
            continue
        if keep is not None and up not in keep:
            continue
        try:
            pr = np.load(f"{root}/{up}/saved_preds.npy").reshape(-1)
            y = np.load(f"{root}/{up}/saved_labels.npy")
        except Exception:
            continue
        if len(pr) != len(y) or y.sum() < 5:
            continue
        cand.append((up, pr, y))
    if keep is not None and len(cand) > args.targets:
        import random
        random.Random(args.seed).shuffle(cand)
    # 盒子体积在选靶点时就要看：它由口袋大小决定，和能不能跑完直接相关。
    # 旧版是选完才发现有的跑不动，白烧机时。
    vol = {}
    for up, _pr, _y in cand:
        c = np.asarray(pockets[up]["pocket_coordinates"], dtype=float)
        sz = (c.max(0) + PAD) - (c.min(0) - PAD)
        vol[up] = float(sz[0] * sz[1] * sz[2])
    if args.report_boxes:
        for up in sorted(vol, key=vol.get):
            print("  %-10s %9.0f Å³" % (up, vol[up]))
        v = sorted(vol.values())
        print(f"\n候选 {len(v)} 个：中位 {v[len(v)//2]:.0f}，"
              f"四分位 {v[len(v)//4]:.0f}–{v[3*len(v)//4]:.0f}，"
              f"范围 {v[0]:.0f}–{v[-1]:.0f}")
        for t in (25000, 30000, 35000):
            print(f"  ≤{t:,} Å³ 的有 {sum(1 for x in v if x <= t)} 个")
        return
    if args.max_box > 0:
        n0 = len(cand)
        cand = [x for x in cand if vol[x[0]] <= args.max_box]
        print(f"盒子体积 ≤{args.max_box:,.0f} Å³：{len(cand)}/{n0} 个候选通过")

    picked, manifest = cand[:args.targets], []
    print(f"合格靶点 {len(cand)}，选中 {len(picked)}"
          + (f"（seed={args.seed} 随机抽）" if keep is not None else "（字母序前 N）"),
          flush=True)

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
        # ⚠️ 分子顺序必须硬校验，只比长度不够：模型读 lmdb，游标是字典序
        # （0, 1, 10, 100, …），和 jsonl 顺序不同而长度相同，会静默错配。
        # 这个坑在本项目里出现过三次。校验方式是「标签为 1 的位置上确实是
        # 该靶点的 active」，两种顺序都试，都不过就跳过。
        act = {m["smiles"] for m in rec["actives"]}

        def ok(seq):
            if seq is None or len(seq) != len(pr):
                return None
            got = {seq[i] for i in range(len(seq)) if y[i] == 1}
            return seq if got == act else None

        smis = ok([m["smiles"] for m in rec["actives"]]
                  + [m["smiles"] for m in rec["decoys"]])
        if smis is None:
            try:
                e = lmdb.open(f"{B}/data/T3_6A/{args.layer}/{up}/{up}_lig.lmdb",
                              subdir=False, readonly=True, lock=False)
                cur = []
                with e.begin() as t:
                    for _k, v in t.cursor():
                        cur.append(pickle.loads(v)["smi"])
                e.close()
                smis = ok(cur)
            except Exception:
                smis = None
        if smis is None:
            print(f"  {up} 分子顺序校验不过，跳过")
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
