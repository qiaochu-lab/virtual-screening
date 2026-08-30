"""把待对接配体写成带 3D 坐标的 SDF —— 复用建 T3 输入时生成的构象。

为什么不用 obabel 现生成
------------------------
smina 读不了 SMILES（内部报 tree.h 错误），需要 3D 结构。
而 data/T3_6A/<层>/<靶点>/<靶点>_lig.lmdb 里已经存着每个分子的 3D 构象——
**模型看到的就是这一份**。复用它有两个好处：省掉 4,000 次 ETKDG，
以及保证对接和检索模型吃的是同一个构象，比较时少一个混杂变量。

lmdb 里是「原子类型 + 坐标 + smi + rdkit mol」，直接用 mol 对象写 SDF 即可。
"""
import json
import os
import pickle
import sys

import lmdb
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
B = "/data/work/vs"


def main(layer="L4"):
    man = json.load(open(f"{B}/dock/manifest.json"))
    n_ok = n_gen = n_fail = 0
    for up in man["targets"]:
        d = f"{B}/dock/{up}"
        out = f"{d}/ligands.sdf"
        if os.path.exists(out) and os.path.getsize(out) > 100:
            continue
        m = json.load(open(f"{d}/manifest.json"))
        p = f"{B}/data/T3_6A/{layer}/{up}/{up}_lig.lmdb"
        cache = {}
        if os.path.exists(p):
            e = lmdb.open(p, subdir=False, readonly=True, lock=False)
            with e.begin() as t:
                for _k, v in t.cursor():            # 游标序 = 模型看到的顺序
                    r = pickle.loads(v)
                    # 存的 mol 只有 2D 图，3D 坐标在 coordinates 里（UniMol 存多个构象）
                    cache[r["smi"]] = (r.get("mol"), r.get("atoms"), r.get("coordinates"))
            e.close()
        w = Chem.SDWriter(out)
        for j, lig in enumerate(m["ligands"]):
            hit = cache.get(lig["smiles"])
            mol = None
            if hit is not None:
                m0, atoms, coords = hit
                # 把存好的构象装回 2D 图：原子数对得上才用，否则宁可重算
                if m0 is not None and atoms and coords is not None and len(coords):
                    try:
                        mh = Chem.RemoveHs(Chem.Mol(m0))
                        if mh.GetNumAtoms() == len(atoms):
                            import numpy as _np
                            from rdkit.Geometry import Point3D
                            c0 = _np.asarray(coords[0], dtype=float)
                            conf = Chem.Conformer(mh.GetNumAtoms())
                            for ai in range(mh.GetNumAtoms()):
                                conf.SetAtomPosition(ai, Point3D(*map(float, c0[ai])))
                            mh.RemoveAllConformers(); mh.AddConformer(conf, assignId=True)
                            mol = mh
                    except Exception:
                        mol = None
            if mol is None:                          # 缓存对不上就现生成
                mol = Chem.MolFromSmiles(lig["smiles"])
                if mol is None:
                    n_fail += 1
                    continue
                mol = Chem.AddHs(mol)
                if AllChem.EmbedMolecule(mol, randomSeed=1) != 0:
                    n_fail += 1
                    continue
                AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
                n_gen += 1
            if mol.GetNumConformers() == 0:
                n_fail += 1
                continue
            mol.SetProp("_Name", f"lig{j}")
            w.write(mol)
            n_ok += 1
        w.close()
    print(f"写出 3D 配体 {n_ok:,}（缓存命中 {n_ok - n_gen:,}，现生成 {n_gen}，失败 {n_fail}）")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "L4")
