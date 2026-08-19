"""把 T3 评测集组装成 UniMol 系模型（DrugCLIP/BindCLIP/LigUnity）吃的 lmdb。

产出目录结构照搬 DEKOIS 的约定，这样可以直接复用这三个模型现成的
`test_dekois_target()` 路径，只改数据路径，不动模型代码：

    data/T3/<layer>/<uniprot>/<uniprot>_lig.lmdb
                             /<uniprot>_pocket.lmdb

lig.lmdb 每条（与 DEKOIS 官方逐字段一致）：
    atoms        list[str]    元素符号
    coordinates  list[ndarray] 只放 1 个构象，与官方一致
    smi          str
    mol          rdkit Mol
    label        int          1=active, 0=decoy

pocket.lmdb 每条：
    pocket, pocket_index, pocket_atoms, pocket_coordinates

口袋来源优先级：PDB 实验结构 > Boltz-2 预测结构。
两者都按同一套残基级 6 Å 逻辑截出（见 extract_pocket*.py），
来源记进 manifest，供 T5 结构鲁棒性分层用。
"""
import argparse
import hashlib
import json
import os
import pickle

import lmdb
import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
B = "/data/work/vs-benchmark"


def load_pockets(threshold):
    """PDB 源优先，Boltz 源补位。返回 uniprot -> (记录, 来源)。"""
    out = {}
    for path, src in [(f"{B}/data/t3/pockets/pocket_{threshold:.1f}A.lmdb", "boltz2_pred"),
                      (f"{B}/data/t3/pockets/pdb_pocket_{threshold:.1f}A.lmdb", "pdb_holo")]:
        if not os.path.exists(path):
            continue
        e = lmdb.open(path, subdir=False, readonly=True, lock=False)
        with e.begin() as t:
            for _, v in t.cursor():
                d = pickle.loads(v)
                out[d["pocket"]] = (d, src)          # PDB 源后加载，自然覆盖 Boltz 源
        e.close()
    return out


def mol_key(m):
    ik = m.get("inchikey")
    return ik if ik else "NOKEY_" + hashlib.md5(m["smiles"].encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="+", default=["L3", "L4", "L1", "L2"])
    ap.add_argument("--threshold", type=float, default=6.0)
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--force", action="store_true", help="已存在也重建")
    args = ap.parse_args()
    out_root = args.out_root or f"{B}/data/T3_{args.threshold:.0f}A"

    # 幂等：数据已建好就跳过（我们会提前手动跑一次，队列里还会再调）
    if os.path.exists(f"{out_root}/manifest.json") and not args.force:
        import json as _j
        m = _j.load(open(f"{out_root}/manifest.json"))
        n = sum(v.get("targets", 0) for v in m.values())
        print(f"{out_root} 已存在（{n} 个靶点），跳过。要重建加 --force")
        return

    pockets = load_pockets(args.threshold)
    from collections import Counter
    print(f"口袋库 {len(pockets):,}  {dict(Counter(s for _, s in pockets.values()))}", flush=True)

    conf_env = lmdb.open(f"{B}/data/t3/mols/conformers.lmdb",
                         subdir=False, readonly=True, lock=False)
    with conf_env.begin() as t:
        n_conf = conf_env.stat()["entries"]
    print(f"构象库 {n_conf:,}", flush=True)

    manifest = {}
    for L in args.layers:
        p = f"{B}/data/t3/eval/{L}.jsonl"
        if not os.path.exists(p):
            print(f"[{L}] 评测集不存在，跳过", flush=True)
            continue

        n_ok, skip_pocket, skip_few = 0, 0, 0
        miss_conf_total = 0
        info = {}
        with conf_env.begin() as ct:
            for line in open(p):
                r = json.loads(line)
                up = r["uniprot"]
                if up not in pockets:
                    skip_pocket += 1
                    continue
                prec, psrc = pockets[up]

                mols, labels, miss = [], [], 0
                for kind, lab in (("actives", 1), ("decoys", 0)):
                    for m in r[kind]:
                        raw = ct.get(mol_key(m).encode())
                        if raw is None:
                            miss += 1
                            continue
                        c = pickle.loads(raw)
                        rd = Chem.MolFromSmiles(c["smi"])
                        if rd is None:
                            miss += 1
                            continue
                        mols.append({"atoms": c["atoms"], "coordinates": c["coordinates"],
                                     "smi": c["smi"], "mol": rd, "label": lab})
                        labels.append(lab)
                miss_conf_total += miss
                # 构象缺失会同时削减 active 和 decoy，比例会漂；
                # active 少于 10 个就不再入选，与评测集的门槛保持一致
                if sum(labels) < 10 or len(labels) - sum(labels) < 10:
                    skip_few += 1
                    continue

                d = f"{out_root}/{L}/{up}"
                os.makedirs(d, exist_ok=True)
                lp = f"{d}/{up}_lig.lmdb"
                pp = f"{d}/{up}_pocket.lmdb"
                for x in (lp, pp):
                    if os.path.exists(x):
                        os.remove(x)

                le = lmdb.open(lp, subdir=False, map_size=1 << 33)
                with le.begin(write=True) as wt:
                    for i, rec in enumerate(mols):
                        wt.put(str(i).encode(), pickle.dumps(rec))
                le.close()

                pe = lmdb.open(pp, subdir=False, map_size=1 << 30)
                with pe.begin(write=True) as wt:
                    wt.put(b"0", pickle.dumps({
                        "pocket": up, "pocket_index": 0,
                        "pocket_atoms": prec["pocket_atoms"],
                        "pocket_coordinates": prec["pocket_coordinates"],
                    }))
                pe.close()

                info[up] = {"n_mols": len(mols), "n_actives": int(sum(labels)),
                            "pocket_source": psrc,
                            "pocket_atoms": len(prec["pocket_atoms"])}
                n_ok += 1

        manifest[L] = {"targets": n_ok, "skipped_no_pocket": skip_pocket,
                       "skipped_too_few": skip_few,
                       "missing_conformers": miss_conf_total, "per_target": info}
        srcs = Counter(v["pocket_source"] for v in info.values())
        print(f"[{L}] 组装 {n_ok:4d} 个靶点  {dict(srcs)}   "
              f"无口袋跳过 {skip_pocket:4d}  数量不足跳过 {skip_few:3d}  "
              f"缺构象分子 {miss_conf_total:,}", flush=True)

    conf_env.close()
    os.makedirs(out_root, exist_ok=True)
    json.dump(manifest, open(f"{out_root}/manifest.json", "w"), indent=1)
    print(f"\n已写入 {out_root}/")


if __name__ == "__main__":
    main()
