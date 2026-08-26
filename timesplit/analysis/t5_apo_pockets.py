"""T5 的第三项对照：apo（未结合）构象的口袋。

要测什么
--------
现有两项对照测的都是 holo 结构——口袋是被配体撑开过的构象。
真实虚筛拿到的往往是 apo：侧链没有为配体让位，口袋可能是塌的。
这是这类模型在实践中最可能吃亏、而我们一直没测的地方。

怎么保证只测「构象」这一个变量
------------------------------
apo 结构里没有配体，没法直接划口袋。做法是：
  1. 把 apo 按主链 CA 叠合到同一靶点的 holo 结构上
  2. 用 **holo 配体的坐标**在叠合后的 apo 里划 6Å 口袋
这样两边口袋的**位置完全一致**，差别只来自侧链构象——正是要测的东西。
如果换成「在 apo 里另找一个口袋」，测的就变成了口袋检测，不是构象敏感性。

叠合用共同残基编号的 CA 配对（同一 UniProt，编号体系一致），
Kabsch 求最优旋转平移。RMSD 太大（>5Å）说明不是同一构象态或编号对不上，
这类靶点直接跳过并记录，不硬叠。
"""
import argparse
import gzip
import json
import os
import pickle
import sys
import urllib.request
from collections import Counter, defaultdict

import lmdb
import numpy as np
from scipy.spatial import cKDTree

B = "/data/work/vs-benchmark"
sys.path.insert(0, B)
from extract_pocket_pdb import fetch, parse_cif   # noqa: E402  复用同一套解析

MAX_RMSD = 5.0


def ca_by_resid(prot):
    """{residue_id: CA 坐标}，用于叠合配对。

    parse_cif 返回的是列式字典（coord/atom_type/residue_id/...），不是原子列表。
    residue_id 已经是 "链+序号" 的组合，同一 UniProt 的不同条目编号体系一致，
    可直接作配对键。
    """
    out = {}
    for i, at in enumerate(prot["atom_type"]):
        if at == "CA":
            out[prot["residue_id"][i]] = prot["coord"][i]
    return out


def kabsch(P, Q):
    """求把 P 叠到 Q 的旋转平移（都是 N×3）。"""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=f"{B}/data/t3/apo_eval_targets.json")
    ap.add_argument("--out", default=f"{B}/data/t3/pockets/apo_pocket_6.0A.lmdb")
    ap.add_argument("--threshold", type=float, default=6.0)
    ap.add_argument("--cache", default=f"{B}/data/t3/cif_cache")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    apo_map = json.load(open(args.targets))
    # 已有的 holo 口袋，用来锚定「同一个位点」
    holo_pockets = {}
    _e = lmdb.open(f"{B}/data/t3/pockets/pdb_pocket_6.0A.lmdb", subdir=False,
                   readonly=True, lock=False)
    with _e.begin() as _t:
        for _k, _v in _t.cursor():
            _d = pickle.loads(_v)
            holo_pockets[_d["pocket"]] = _d
    _e.close()
    print(f"已有 holo 口袋 {len(holo_pockets):,}（用作位点锚点）")
    choice = json.load(open(f"{B}/data/t3/crystal_ligand_choice.json"))["choice"]
    ups = sorted(apo_map)[:args.limit] if args.limit else sorted(apo_map)
    print(f"待处理靶点 {len(ups)}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if os.path.exists(args.out):
        os.remove(args.out)
    env = lmdb.open(args.out, subdir=False, map_size=1 << 34)

    stat, manifest, n = Counter(), {}, 0
    with env.begin(write=True) as w:
        for i, up in enumerate(ups, 1):
            ch = choice.get(up)
            if not ch:
                stat["无 holo 参照"] += 1
                continue
            holo_path = fetch(ch["pdb_id"], args.cache)
            if not holo_path:
                stat["holo 下载失败"] += 1
                continue
            holo_prot, holo_ligs = parse_cif(holo_path, {ch["comp_id"]})
            if not holo_ligs:
                stat["holo 里找不到该配体"] += 1
                continue
            copies = next(iter(holo_ligs.values()))          # comp_id -> {拷贝: 坐标}
            # 必须选**与已有 holo 口袋同一处**的那个拷贝：
            # 同源多聚体里不同链的位点可能相距几十埃，随便取第一个拷贝
            # 会把 apo 口袋划到另一个亚基上（实测 17/52 个靶点质心偏了 5-62Å）。
            # 以 holo 口袋质心为锚，取最近的拷贝。
            ref = holo_pockets.get(up)
            if ref is None:
                stat["无 holo 口袋可对齐"] += 1
                continue
            ref_c = np.asarray(ref["pocket_coordinates"], dtype=float).mean(0)
            best = min(copies.values(),
                       key=lambda c: np.linalg.norm(np.asarray(c, dtype=float).mean(0) - ref_c))
            lig_coord = np.asarray(best, dtype=float)
            holo_ca = ca_by_resid(holo_prot)

            placed = False
            for apo_id in apo_map[up]:
                apo_path = fetch(apo_id, args.cache)
                if not apo_path:
                    continue
                apo_prot, _ = parse_cif(apo_path, set())
                apo_ca = ca_by_resid(apo_prot)
                common = [k for k in apo_ca if k in holo_ca]
                if len(common) < 30:            # 配对残基太少，叠合不可信
                    continue
                P = np.array([apo_ca[k] for k in common])
                Q = np.array([holo_ca[k] for k in common])
                R, t = kabsch(P, Q)
                rmsd = float(np.sqrt(((P @ R.T + t - Q) ** 2).sum(1).mean()))
                if rmsd > MAX_RMSD:
                    continue
                coords = apo_prot["coord"] @ R.T + t
                tree = cKDTree(coords)
                idx = sorted({j for j in
                              set().union(*[set(x) for x in
                                            tree.query_ball_point(lig_coord, args.threshold)])})
                if len(idx) < 20:
                    continue
                # 整残基入选，与 6Å 主口径一致
                keep_res = {apo_prot["residue_id"][j] for j in idx}
                sel = [j for j, rid in enumerate(apo_prot["residue_id"]) if rid in keep_res]
                rec = {"pocket": up,
                       "pocket_atoms": [apo_prot["atom_type"][j] for j in sel],
                       "pocket_coordinates": [coords[j] for j in sel],
                       "apo_pdb": apo_id, "holo_pdb": ch["pdb_id"],
                       "align_rmsd": rmsd, "n_aligned": len(common)}
                w.put(up.encode(), pickle.dumps(rec))
                manifest[up] = {"apo_pdb": apo_id, "holo_pdb": ch["pdb_id"],
                                "align_rmsd": round(rmsd, 2), "n_atoms": len(sel)}
                stat["成功"] += 1
                n += 1
                placed = True
                break
            if not placed:
                stat["无可用 apo（叠合失败或口袋太小）"] += 1
            if i % 20 == 0:
                print(f"  {i}/{len(ups)}  成功 {n}", flush=True)
    env.close()

    json.dump(manifest, open(f"{B}/data/t3/apo_pocket_manifest.json", "w"), indent=1)
    print(f"\n{dict(stat)}")
    if manifest:
        r = [v["align_rmsd"] for v in manifest.values()]
        a = [v["n_atoms"] for v in manifest.values()]
        print(f"叠合 RMSD 中位 {np.median(r):.2f}Å   口袋原子中位 {np.median(a):.0f}")
    print(f"写入 {args.out}")


if __name__ == "__main__":
    main()
