"""Sanity check before the apo control: how different are these apo pockets
from holo, really?

Why this had to be done
------------
The apo group's EF1 went up, not down. Two possibilities:
  (a) the model really is insensitive to pocket conformation — a valuable
      finding
  (b) the apo structures we picked are nearly identical to holo — in which
      case this test is vacuous
A global alignment RMSD with a median of 1.21Å already hints at (b), but
global RMSD is dominated by the backbone, which **masks the side chains**.
What actually needs checking is at the pocket:
  * whether the two sides' pockets cover the same residues
  * the **side-chain heavy-atom** RMSD of the matched residues (induced fit
    shows up mainly in the side chains)
If the side-chain RMSD is also small, the conclusion has to be written as
"insensitive on apo structures whose conformation doesn't differ much", not
"insensitive to apo in general".
"""
import json
import pickle

import lmdb
import numpy as np

B = "/data/work/vs-benchmark"
BACKBONE = {"N", "CA", "C", "O"}


def load(path):
    e = lmdb.open(path, subdir=False, readonly=True, lock=False)
    out = {}
    with e.begin() as t:
        for _k, v in t.cursor():
            d = pickle.loads(v)
            out[d["pocket"]] = d
    e.close()
    return out


def main():
    apo = load(f"{B}/data/t3/pockets/apo_pocket_6.0A.lmdb")
    holo = load(f"{B}/data/t3/pockets/pdb_pocket_6.0A.lmdb")
    man = json.load(open(f"{B}/data/t3/apo_pocket_manifest.json"))
    common = sorted(set(apo) & set(holo))
    print(f"两边都有口袋的靶点: {len(common)}")

    n_apo, n_holo, jac, sc_rmsd, glob = [], [], [], [], []
    for up in common:
        a, h = apo[up], holo[up]
        n_apo.append(len(a["pocket_atoms"]))
        n_holo.append(len(h["pocket_atoms"]))
        glob.append(man.get(up, {}).get("align_rmsd", np.nan))
        # side-chain heavy atoms among the pocket atoms (backbone's four atoms excluded)
        ac = np.asarray(a["pocket_coordinates"], dtype=float)
        hc = np.asarray(h["pocket_coordinates"], dtype=float)
        asel = [i for i, t in enumerate(a["pocket_atoms"]) if t not in BACKBONE]
        hsel = [i for i, t in enumerate(h["pocket_atoms"]) if t not in BACKBONE]
        if not asel or not hsel:
            continue
        # No atom-level correspondence exists, so use the median nearest-neighbour distance as a proxy for side-chain deviation
        from scipy.spatial import cKDTree
        d, _ = cKDTree(hc[hsel]).query(ac[asel])
        sc_rmsd.append(float(np.median(d)))
        jac.append(min(len(asel), len(hsel)) / max(len(asel), len(hsel)))

    print(f"口袋原子数  apo 中位 {np.median(n_apo):.0f}   holo 中位 {np.median(n_holo):.0f}")
    print(f"侧链原子数之比（小/大）中位 {np.median(jac):.2f}")
    print(f"全局叠合 RMSD 中位 {np.nanmedian(glob):.2f} Å")
    print(f"**口袋侧链原子到最近对应原子的距离** 中位 {np.median(sc_rmsd):.2f} Å")
    q = np.percentile(sc_rmsd, [25, 50, 75, 90])
    print(f"   分位 25/50/75/90: {q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f} / {q[3]:.2f} Å")
    big = sum(1 for x in sc_rmsd if x > 1.0)
    print(f"   侧链偏离 >1Å 的靶点: {big}/{len(sc_rmsd)}（{big/len(sc_rmsd)*100:.0f}%）")
    print("\n判读：若绝大多数靶点侧链偏离都在 1Å 以内，"
          "说明我们拿到的 apo 与 holo 构象接近，\n"
          "      结论只能写成「构象差异不大时不敏感」，不能推广到塌陷口袋。")


if __name__ == "__main__":
    main()
