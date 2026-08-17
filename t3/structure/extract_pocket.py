"""从 Boltz-2 预测的复合物结构中提取口袋，写成各模型需要的 pocket.lmdb 格式。

提取逻辑严格照搬 DrugCLIP 官方 py_scripts/write_dude_multi.py 的 get_different_raid()：

    对每个蛋白原子 i、每个配体原子 j，若距离 < threshold，
    则把 i 所属的**整个残基**收入口袋（残基级，不是原子级）。

官方默认 threshold=6，导师亦确认 6 Å 优于 8 Å，故 6 Å 为主口径；
同时输出 5 Å 版本作对照（PocketAffDB 文件名暗示 5.0 Å），可并入 T5 结构鲁棒性。
"""
import argparse, os, pickle, sys
import lmdb
import numpy as np
from scipy.spatial import cKDTree


def read_complex_pdb(path):
    """读 Boltz-2 输出：链 A = 蛋白(ATOM)，链 B = 配体(HETATM, resname LIG)。"""
    prot = {"coord": [], "atom_type": [], "residue_id": [], "residue_type": []}
    lig = []
    with open(path) as f:
        for ln in f:
            rec = ln[:6]
            if rec not in ("ATOM  ", "HETATM"):
                continue
            chain = ln[21]
            try:
                xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
            except ValueError:
                continue
            if chain == "B":
                lig.append(xyz)
            elif chain == "A":
                prot["coord"].append(xyz)
                prot["atom_type"].append(ln[12:16].strip())
                prot["residue_id"].append(ln[21] + ln[22:27].strip())   # chain + resseq
                prot["residue_type"].append(ln[17:20].strip())
    prot["coord"] = np.asarray(prot["coord"], dtype=float)
    return prot, np.asarray(lig, dtype=float)


def extract_pocket(prot, lig_coord, threshold):
    """返回 (atom_types, coords, residue_types)。残基级选择，与官方实现一致。"""
    if len(prot["coord"]) == 0 or len(lig_coord) == 0:
        return None
    # 官方是 O(N*M) 双循环；这里用 KD-tree 等价加速
    tree = cKDTree(lig_coord)
    dmin, _ = tree.query(prot["coord"], k=1)
    near = dmin < threshold
    pocket_res = {prot["residue_id"][i] for i in np.nonzero(near)[0]}
    idx = [i for i, r in enumerate(prot["residue_id"]) if r in pocket_res]
    if not idx:
        return None
    return ([prot["atom_type"][i] for i in idx],
            [prot["coord"][i] for i in idx],
            [prot["residue_type"][i] for i in idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb_dirs", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[6.0, 5.0])
    args = ap.parse_args()

    pdbs = []
    for d in args.pdb_dirs:
        for root, _, files in os.walk(d):
            for fn in files:
                if fn.endswith("_model_0.pdb"):
                    pdbs.append(os.path.join(root, fn))
    pdbs.sort()
    print(f"找到 {len(pdbs)} 个复合物结构", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    envs = {}
    for t in args.thresholds:
        p = f"{args.out_dir}/pocket_{t:.1f}A.lmdb"
        if os.path.exists(p):
            os.remove(p)
        envs[t] = lmdb.open(p, subdir=False, map_size=1 << 34)

    stats = {t: [] for t in args.thresholds}
    n_ok = n_fail = 0
    for i, p in enumerate(pdbs):
        name = os.path.basename(p).replace("_model_0.pdb", "")
        prot, lig = read_complex_pdb(p)
        if len(lig) == 0:
            n_fail += 1
            print(f"  [skip] {name}: 未找到配体链", file=sys.stderr)
            continue
        ok = False
        for t in args.thresholds:
            r = extract_pocket(prot, lig, t)
            if r is None:
                continue
            atoms, coords, restypes = r
            rec = {
                "pocket": name,
                "pocket_index": 0,
                "pocket_atoms": atoms,
                "pocket_coordinates": coords,
                "pocket_residue_type": restypes,
                "threshold": t,
                "n_protein_atoms": int(len(prot["coord"])),
                "n_ligand_atoms": int(len(lig)),
            }
            with envs[t].begin(write=True) as txn:
                txn.put(str(i).encode(), pickle.dumps(rec))
            stats[t].append(len(atoms))
            ok = True
        n_ok += ok
        if (i + 1) % 50 == 0:
            print(f"  ...{i+1}/{len(pdbs)}", flush=True)

    for e in envs.values():
        e.close()

    print(f"\n成功 {n_ok} / 失败 {n_fail}")
    print("\n口袋规模（原子数）:")
    print("  %-10s %8s %8s %8s %8s" % ("阈值", "靶点数", "中位", "最小", "最大"))
    for t in args.thresholds:
        s = sorted(stats[t])
        if s:
            print("  %-10s %8d %8d %8d %8d"
                  % (f"{t:.1f} Å", len(s), s[len(s)//2], s[0], s[-1]))
    if len(args.thresholds) == 2:
        a, b = args.thresholds
        sa, sb = stats[a], stats[b]
        if sa and sb:
            print(f"\n  {a:.1f}Å 口袋平均比 {b:.1f}Å 大 "
                  f"{(np.mean(sa)/np.mean(sb)-1)*100:.1f}%")


if __name__ == "__main__":
    main()
