"""Generate SaProt structure-aware sequences for T3's targets (needed by SPRINT's protein tower).

A SaProt sequence encodes each residue as two characters, "amino acid +
foldseek 3Di structure token", e.g. MdEvKp.... This is exactly how
SPRINT's released checkpoint was trained; without the structure token it
falls back to a mask, which effectively throws away half the input, so we
compute it properly from the available structures.

Structure sourcing matches run_t3_conglude.py exactly (PDB experimental
structures first, Boltz-2 predictions filling the gaps), so SPRINT and
ConGLUDe see the same structural conditioning and any difference between
them comes only from the models themselves.

pLDDT masking
-------------
SaProt's convention is to mask out the structure token in low-confidence
regions (pLDDT < 70). But pLDDT only exists for predicted structures:
  - Boltz-2 predicted structure -> the B-factor column holds pLDDT, masking on
  - PDB experimental structure  -> the B-factor column holds the real B-factor,
                                    masking **must be turned off**, otherwise
                                    B-factors get mistaken for pLDDT and a large
                                    chunk gets masked by accident
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

B = "/data/work/vs-benchmark"
FOLDSEEK = f"{B}/tools/foldseek/bin/foldseek"
sys.path.insert(0, f"{B}/code/panspecies-dti")
# get_struc_seq lives in utils/structure_to_saprot.py, not a package module, so its path must be added separately
sys.path.insert(0, f"{B}/code/panspecies-dti/utils")


def build_boltz_index():
    idx = {}
    for d in ["boltz_batch_out", "boltz_retry_out", "boltz_gap_out", "boltz_r2_out"]:
        p = f"{B}/{d}"
        if not os.path.isdir(p):
            continue
        for root, _, files in os.walk(p):
            for fn in files:
                if fn.endswith("_model_0.pdb"):
                    idx.setdefault(fn.replace("_model_0.pdb", ""), os.path.join(root, fn))
    return idx


def one(task):
    up, path, is_pred, pid = task
    from structure_to_saprot import get_struc_seq
    try:
        d = get_struc_seq(FOLDSEEK, path, chains=None,
                          process_id=pid, plddt_mask=is_pred)
        if not d:
            return up, None, "foldseek 无输出"
        # get_struc_seq returns {chain: (aa_seq, struc_seq, combined)}; take the longest chain
        best = max(d.values(), key=lambda v: len(v[0]))
        combined = best[2]
        if not combined:
            return up, None, "结构序列为空"
        return up, {"saprot": combined, "len": len(best[0]),
                    "source": "boltz2_pred" if is_pred else "pdb_holo"}, None
    except Exception as e:                      # noqa: BLE001 tolerate failures per-item
        return up, None, f"{type(e).__name__}: {e}"[:120]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--pdb_dir", default=f"{B}/data/t3/pdb_for_struct")
    ap.add_argument("--out", default=f"{B}/data/t3/saprot_seqs.json")
    args = ap.parse_args()

    # Reuse the PDB files already materialized for the ConGLUDe run (both experimental and predicted structures are in there)
    cands = {}
    for L in ["L1", "L2", "L3", "L4"]:
        d = f"{B}/code/conglude/data/datasets/predict_datasets/t3_{L}/raw/pdb_files"
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.endswith(".pdb"):
                cands.setdefault(fn[:-4], os.path.join(d, fn))

    boltz = build_boltz_index()
    for up, p in boltz.items():
        cands.setdefault(up, p)

    # Determine the source: a file under a boltz output directory is a predicted structure
    tasks = []
    for i, (up, p) in enumerate(sorted(cands.items())):
        is_pred = "/boltz_" in p or (up in boltz and boltz[up] == p)
        tasks.append((up, p, is_pred, i % max(1, args.procs)))
    print(f"待处理结构: {len(tasks):,}", flush=True)

    from collections import Counter
    out, fails = {}, Counter()
    with Pool(args.procs) as pool:
        for i, (up, rec, err) in enumerate(pool.imap_unordered(one, tasks, chunksize=8)):
            if rec is None:
                fails[err.split(":")[0]] += 1
            else:
                out[up] = rec
            if (i + 1) % 200 == 0:
                print(f"  {i+1:,}/{len(tasks):,}  成功 {len(out):,}", flush=True)

    json.dump(out, open(args.out, "w"))
    print(f"\n成功 {len(out):,} / {len(tasks):,}")
    print(f"  {dict(Counter(v['source'] for v in out.values()))}")
    for k, v in fails.most_common(6):
        print(f"  失败 {k}: {v}")
    print(f"已写入 {args.out}")


if __name__ == "__main__":
    main()
