"""从 RCSB 实验结构中提取 T3 新靶点的口袋。

与 extract_pocket.py（Boltz-2 预测结构版）用**完全相同**的残基级选择逻辑，
即 DrugCLIP 官方 py_scripts/write_dude_multi.py 的 get_different_raid()：
蛋白原子只要有一个落在配体任意原子 threshold 内，其**整个残基**进入口袋。

相对预测结构版多出来的三件事
----------------------------
1. **链归属**。47.1% 的 PDB 条目含多个 UniProt（核糖体、蛋白酶体、呼吸链）。
   若用文件里所有链的蛋白原子，配体明明结合在 A 亚基上，会被当成 B 亚基的
   口袋。所以蛋白原子限制到该靶点自己的链；拿不到链归属时退回全部蛋白链。

2. **候选回退**。即使限制到自己的链，配体也可能压根不接触它——
   此时这个 (pdb, ligand) 对该靶点无效。按 rank_crystal_ligands.py 排好的
   候选顺序逐个试，取第一个能截出合格口袋的。

3. **实验结构的杂事**：多链、同一配体的多个拷贝、altloc、多 model。
   约定为：只取 model 1；altloc 只保留 '.'/'?'/'A'；同一 comp_id 的多个拷贝
   取与该靶点链接触原子最多的那个；丢弃氢原子（与 DUD-E 口袋一致）。
"""
import argparse
import gzip
import json
import os
import pickle
import sys
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import lmdb
import numpy as np
from scipy.spatial import cKDTree

AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL",
}
URL = "https://files.rcsb.org/download/{}.cif.gz"
MIN_POCKET_ATOMS = 40      # 低于此视为配体几乎不接触该链，判定候选无效


# ---------------------------------------------------------------- 下载

def fetch(pdb_id, cache_dir):
    dst = os.path.join(cache_dir, f"{pdb_id}.cif.gz")
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                URL.format(pdb_id), headers={"User-Agent": "vs-benchmark/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(dst, "wb") as f:
                f.write(r.read())
            return dst
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            if attempt == 2:
                print(f"  [下载失败] {pdb_id}: {e}", file=sys.stderr)
                if os.path.exists(dst):
                    os.remove(dst)
                return None
    return None


# ---------------------------------------------------------------- 解析

def parse_cif(path, lig_comps):
    """扫一遍 _atom_site 表，返回 (蛋白原子按链分组, 配体拷贝坐标)。

    lig_comps 是这个条目里所有待考察的配体 comp_id 集合，一次解析全取到，
    免得同一个 cif 为每个候选重复解析。
    """
    prot = {"coord": [], "atom_type": [], "residue_id": [],
            "residue_type": [], "chain": []}
    lig = {}                                   # comp_id -> {拷贝键: [坐标]}

    cols, in_loop, header = {}, False, False
    with gzip.open(path, "rt", errors="replace") as f:
        for ln in f:
            s = ln.strip()
            if s.startswith("_atom_site."):
                if not in_loop:
                    in_loop, header, cols = True, True, {}
                cols[s.split(".", 1)[1]] = len(cols)
                continue
            if not in_loop:
                continue
            if header:
                header = False
            if not s or s[0] in "#_" or s.startswith("loop_"):
                break
            p = s.split()
            if len(p) < len(cols):
                continue

            def g(k, default="."):
                i = cols.get(k)
                return p[i] if i is not None else default

            if g("pdbx_PDB_model_num", "1") != "1":
                continue
            if g("label_alt_id") not in (".", "?", "A"):
                continue
            if g("type_symbol") == "H":
                continue
            try:
                xyz = (float(g("Cartn_x")), float(g("Cartn_y")), float(g("Cartn_z")))
            except ValueError:
                continue

            comp = g("auth_comp_id") if "auth_comp_id" in cols else g("label_comp_id")
            chain = g("auth_asym_id") if "auth_asym_id" in cols else g("label_asym_id")
            seq = g("auth_seq_id") if "auth_seq_id" in cols else g("label_seq_id")

            if g("group_PDB") == "ATOM" and comp in AA:
                prot["coord"].append(xyz)
                prot["atom_type"].append(g("label_atom_id").strip('"'))
                prot["residue_id"].append(f"{chain}{seq}")
                prot["residue_type"].append(comp)
                prot["chain"].append(chain)
            elif comp in lig_comps:
                lig.setdefault(comp, {}).setdefault(f"{chain}{seq}", []).append(xyz)

    prot["coord"] = np.asarray(prot["coord"], dtype=float)
    prot["chain"] = np.asarray(prot["chain"], dtype=object)
    return prot, {c: {k: np.asarray(v, dtype=float) for k, v in d.items()}
                  for c, d in lig.items()}


def subset_chains(prot, chains):
    """把蛋白原子限制到指定链；chains 为空则原样返回。"""
    if not chains:
        return prot
    keep = np.isin(prot["chain"], list(chains))
    if not keep.any():
        return None
    idx = np.nonzero(keep)[0]
    return {"coord": prot["coord"][idx],
            "atom_type": [prot["atom_type"][i] for i in idx],
            "residue_id": [prot["residue_id"][i] for i in idx],
            "residue_type": [prot["residue_type"][i] for i in idx],
            "chain": prot["chain"][idx]}


def pick_copy(prot, copies, threshold=6.0):
    """同一 comp_id 的多个拷贝里，取与该靶点链接触原子最多的那个。"""
    if prot is None or len(prot["coord"]) == 0 or not copies:
        return None, None, 0
    tree = cKDTree(prot["coord"])
    best, best_c, best_n = None, None, -1
    for k, c in copies.items():
        if len(c) == 0:
            continue
        hit = tree.query_ball_point(c, threshold)
        n = len({i for sub in hit for i in sub})
        if n > best_n:
            best, best_c, best_n = k, c, n
    return best, best_c, max(best_n, 0)


def extract_pocket(prot, lig_coord, threshold):
    """与 extract_pocket.py 逐字相同的残基级选择。"""
    if prot is None or len(prot["coord"]) == 0 or len(lig_coord) == 0:
        return None
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


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[6.0, 5.0])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min_atoms", type=int, default=MIN_POCKET_ATOMS)
    args = ap.parse_args()

    cand_all = json.load(open(args.candidates))["candidates"]
    items = sorted(cand_all.items())
    print(f"待处理靶点: {len(items):,}", flush=True)

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    ids = sorted({c["pdb_id"] for _, cs in items for c in cs})
    print(f"涉及 mmCIF: {len(ids):,}，下载/校验 ...", flush=True)
    with ThreadPoolExecutor(args.workers) as ex:
        paths = list(ex.map(lambda i: fetch(i, args.cache_dir), ids))
    cache = {i: p for i, p in zip(ids, paths) if p}
    print(f"  拿到 {len(cache):,}/{len(ids):,}", flush=True)

    envs = {}
    for t in args.thresholds:
        p = f"{args.out_dir}/pdb_pocket_{t:.1f}A.lmdb"
        if os.path.exists(p):
            os.remove(p)
        envs[t] = lmdb.open(p, subdir=False, map_size=1 << 35)

    main_t = max(args.thresholds)          # 用最大阈值判定候选是否合格
    stats = {t: [] for t in args.thresholds}
    manifest, failed = {}, []
    rank_hist, reject = Counter(), Counter()
    parsed_cache = {}                      # 同一 cif 在一个靶点内复用

    for i, (up, cands) in enumerate(items):
        parsed_cache.clear()
        chosen = None
        for rank, c in enumerate(cands):
            path = cache.get(c["pdb_id"])
            if path is None:
                reject["cif 缺失"] += 1
                continue
            if c["pdb_id"] not in parsed_cache:
                comps = {x["comp_id"] for x in cands if x["pdb_id"] == c["pdb_id"]}
                try:
                    parsed_cache[c["pdb_id"]] = parse_cif(path, comps)
                except Exception as e:                       # noqa: BLE001 逐条容错
                    reject["解析失败"] += 1
                    parsed_cache[c["pdb_id"]] = None
                    print(f"  [解析失败] {c['pdb_id']}: {e}", file=sys.stderr)
            pr = parsed_cache[c["pdb_id"]]
            if pr is None:
                continue
            prot_all, ligs = pr

            sub = subset_chains(prot_all, c["target_chains"])
            used_chains = c["target_chains"]
            if sub is None:                 # 链归属与坐标对不上，退回全部蛋白链
                sub, used_chains = prot_all, []
            copies = ligs.get(c["comp_id"]) or {}
            if not copies:
                reject["结构里没有该配体"] += 1
                continue
            key, lig, n_contact = pick_copy(sub, copies)
            if lig is None:
                reject["配体不接触该靶点的链"] += 1
                continue
            r = extract_pocket(sub, lig, main_t)
            if r is None or len(r[0]) < args.min_atoms:
                reject["口袋过小(<%d原子)" % args.min_atoms] += 1
                continue
            chosen = (rank, c, sub, lig, key, used_chains)
            break

        if chosen is None:
            failed.append({"uniprot": up, "n_candidates": len(cands),
                           "reason": "所有候选都不合格"})
            continue

        rank, c, sub, lig, key, used_chains = chosen
        rank_hist[rank] += 1
        ok = False
        for t in args.thresholds:
            r = extract_pocket(sub, lig, t)
            if r is None:
                continue
            atoms, coords, restypes = r
            rec = {
                "pocket": up,
                "pocket_index": 0,
                "pocket_atoms": atoms,
                "pocket_coordinates": coords,
                "pocket_residue_type": restypes,
                "threshold": t,
                "source": "pdb",
                "pdb_id": c["pdb_id"],
                "ligand_comp_id": c["comp_id"],
                "ligand_copy": key,
                "ligand_tanimoto_to_t3": c["tanimoto"],
                "ligand_mw": c["mw"],
                "target_chains": used_chains,
                "candidate_rank": rank,
                "n_protein_atoms": int(len(sub["coord"])),
                "n_ligand_atoms": int(len(lig)),
            }
            with envs[t].begin(write=True) as txn:
                txn.put(str(i).encode(), pickle.dumps(rec))
            stats[t].append(len(atoms))
            ok = True
        if ok:
            manifest[up] = {"pdb_id": c["pdb_id"], "comp_id": c["comp_id"],
                            "copy": key, "tanimoto": c["tanimoto"],
                            "chains": used_chains, "rank": rank}
        else:
            failed.append({"uniprot": up, "n_candidates": len(cands),
                           "reason": "口袋为空"})
        if (i + 1) % 50 == 0:
            print(f"  ...{i+1}/{len(items)}  成功 {len(manifest)}", flush=True)

    for e in envs.values():
        e.close()

    json.dump({"manifest": manifest, "failed": failed,
               "rank_hist": dict(rank_hist), "reject": dict(reject)},
              open(f"{args.out_dir}/pdb_pocket_manifest.json", "w"), indent=1)

    print(f"\n成功 {len(manifest)} / 失败 {len(failed)}")
    print("\n用到第几个候选（0 = 首选就合格）:")
    for r in sorted(rank_hist):
        print(f"  第 {r+1} 个: {rank_hist[r]}")
    print("\n候选被否掉的原因:")
    for k, v in reject.most_common():
        print(f"  {k}: {v}")
    print("\n口袋规模（原子数）:")
    print("  %-10s %8s %8s %8s %8s" % ("阈值", "靶点数", "中位", "最小", "最大"))
    for t in args.thresholds:
        s = sorted(stats[t])
        if s:
            print("  %-10s %8d %8d %8d %8d"
                  % (f"{t:.1f} Å", len(s), s[len(s) // 2], s[0], s[-1]))


if __name__ == "__main__":
    main()
