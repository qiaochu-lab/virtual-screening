"""T5's third control: apo (unbound) conformation pockets.

What this is testing
------------
The existing two controls both test holo structures — pockets in a
conformation that's been opened up by the ligand. Real screening campaigns
often only have an apo structure: side chains haven't made room for a
ligand, and the pocket may be collapsed. This is the place these models are
most likely to suffer in practice, and one we had never tested.

How to make sure only "conformation" varies
------------
An apo structure has no ligand, so a pocket can't be carved out directly.
The approach:
  1. Align the apo structure to the holo structure of the same target by
     backbone CA
  2. Carve a 6Å pocket in the aligned apo structure using the **holo
     ligand's coordinates**
This keeps the pocket **position identical** on both sides, so the only
difference comes from side-chain conformation — exactly what we want to
measure. Finding "a different pocket in the apo structure" instead would
turn this into a pocket-detection test, not a conformational-sensitivity
test.

Alignment uses CA pairs matched by shared residue numbering (same UniProt,
consistent numbering scheme), with Kabsch to find the optimal rotation and
translation. An RMSD that's too large (>5Å) means it isn't the same
conformational state, or the numbering doesn't match — such targets are
simply skipped and logged rather than force-aligned.
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
from extract_pocket_pdb import fetch, parse_cif   # noqa: E402  reuse the same parsing

MAX_RMSD = 5.0


def ca_by_resid(prot):
    """{residue_id: CA coordinates}, used for alignment pairing.

    parse_cif returns a columnar dict (coord/atom_type/residue_id/...), not a
    list of atoms. residue_id is already a "chain+number" composite, and the
    numbering scheme is consistent across different entries of the same
    UniProt, so it can be used directly as the pairing key.
    """
    out = {}
    for i, at in enumerate(prot["atom_type"]):
        if at == "CA":
            out[prot["residue_id"][i]] = prot["coord"][i]
    return out


def kabsch(P, Q):
    """Find the rotation and translation that aligns P onto Q (both N×3)."""
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
    # Existing holo pockets, used to anchor "the same site"
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
            copies = next(iter(holo_ligs.values()))          # comp_id -> {copy: coordinates}
            # Must pick the copy that is **at the same site as the existing holo
            # pocket**: in a homo-oligomer, sites on different chains can be tens
            # of angstroms apart, and just taking the first copy would carve the
            # apo pocket on the wrong subunit (empirically, 17/52 targets had
            # centroids off by 5-62A). Anchor on the holo pocket's centroid and
            # take the nearest copy.
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
                if len(common) < 30:            # too few paired residues, alignment isn't trustworthy
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
                # Whole residues included, consistent with the main 6Å convention
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
