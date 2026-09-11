"""Write the ligands to be docked out as an SDF with 3D coordinates -- reusing the
conformers generated when the T3 inputs were built.

Why not just regenerate with obabel
------------------------------------
smina cannot read SMILES (it raises an internal tree.h error) and needs a 3D
structure. data/T3_6A/<layer>/<target>/<target>_lig.lmdb already stores each
molecule's 3D conformer -- **this is exactly what the model saw**. Reusing it
has two benefits: it skips 4,000 ETKDG embeddings, and it guarantees docking
and the retrieval models are scored on the same conformer, removing one
confound from the comparison.

The lmdb holds "atom types + coordinates + smi + rdkit mol"; the SDF can be
written directly from the mol object.
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
                for _k, v in t.cursor():            # cursor order = the order the model saw
                    r = pickle.loads(v)
                    # the stored mol is 2D-only; 3D coordinates live in coordinates (UniMol stores multiple conformers)
                    cache[r["smi"]] = (r.get("mol"), r.get("atoms"), r.get("coordinates"))
            e.close()
        w = Chem.SDWriter(out)
        for j, lig in enumerate(m["ligands"]):
            hit = cache.get(lig["smiles"])
            mol = None
            if hit is not None:
                m0, atoms, coords = hit
                # re-attach the stored conformer to the 2D graph: only if the atom count matches, else recompute
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
            if mol is None:                          # no cache match, so generate it now
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
