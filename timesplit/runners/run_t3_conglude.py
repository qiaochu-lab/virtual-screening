"""Run ConGLUDe on the T3 evaluation set, producing the raw scores the unified evaluation layer needs.

ConGLUDe's interface
---------------------
predict.py consumes `info/protein_ids.txt` + `info/smiles.txt` and emits
a `vs_predictions.npy` similarity matrix.

Warning: matrix orientation. The official README says "rows correspond
to protein names", which **doesn't match the code**. predict.py actually
computes `vs_preds = encoded_ligands_b @ protein_embeddings.t()`, so it's
really **ligand (rows) x protein (columns)**:
    row order matches `processed/ligand_embeddings/index2smiles.json`
    column order matches `embeddings/protein_names.txt`
Following the README leads straight to an IndexError (we hit this once),
so a shape assertion is added below to prevent it from happening again.

Where the structures come from
---------------------------------
ConGLUDe only recognizes `{protein_id}.pdb`, not mmCIF. So:
  - targets with a PDB experimental structure available in the legacy
    PDB format -> use RCSB's .pdb
  - everything else (oversized structures with no legacy format, or
    targets with no PDB entry at all) -> use the Boltz-2 predicted .pdb
Both sources are written to disk keyed by **UniProt accession** as the
protein_id, keeping the target convention consistent with the other
models; the manifest also records which structure type each target used
(experimental holo vs. predicted), which is itself a stratification
variable needed for the T5 structure-robustness analysis.

One layer at a time: ConGLUDe computes a single big matrix of every
target x every molecule for that layer, but each target's molecule set
differs, so after splitting by target only that target's own columns
are taken.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

B = "/data/work/vs-benchmark"
CG = f"{B}/code/conglude"
PY = "/data/work/envs/conglude/bin/python"
PDB_URL = "https://files.rcsb.org/download/{}.pdb"


def fetch_pdb(pdb_id, dst):
    """Download the legacy PDB format; oversized structures have no such format, returns False for the caller to fall back on."""
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return True
    try:
        req = urllib.request.Request(PDB_URL.format(pdb_id),
                                     headers={"User-Agent": "vs-benchmark/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dst, "wb") as f:
            f.write(r.read())
        return True
    except (urllib.error.URLError, OSError, TimeoutError):
        if os.path.exists(dst):
            os.remove(dst)
        return False


def build_boltz_index():
    """Scan out all Boltz-2 structures in one pass, avoiding an os.walk per target."""
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


def prepare(layer, recs, ds_dir, boltz_idx, pdb_choice, workers):
    info, pdbdir = f"{ds_dir}/info", f"{ds_dir}/raw/pdb_files"
    os.makedirs(info, exist_ok=True)
    os.makedirs(pdbdir, exist_ok=True)

    # First download experimental structures concurrently; fall back to the predicted structure for anything unavailable
    want_rcsb = [(r["uniprot"], pdb_choice[r["uniprot"]]) for r in recs
                 if r["uniprot"] in pdb_choice]
    with ThreadPoolExecutor(workers) as ex:
        got = list(ex.map(lambda t: fetch_pdb(t[1], f"{pdbdir}/{t[0]}.pdb"), want_rcsb))
    rcsb_ok = {u for (u, _), g in zip(want_rcsb, got) if g}

    used, missing = {}, []
    for r in recs:
        up = r["uniprot"]
        if up in rcsb_ok:
            used[up] = {"kind": "pdb_holo", "pdb_id": pdb_choice[up]}
            continue
        src = boltz_idx.get(up)
        if src:
            shutil.copyfile(src, f"{pdbdir}/{up}.pdb")
            used[up] = {"kind": "boltz2_pred"}
        else:
            missing.append(up)
            if os.path.exists(f"{pdbdir}/{up}.pdb"):
                os.remove(f"{pdbdir}/{up}.pdb")

    ups = [r["uniprot"] for r in recs if r["uniprot"] in used]
    open(f"{info}/protein_ids.txt", "w").write("\n".join(ups) + "\n")

    # Unique molecules across the whole layer; each target's labels are backfilled by SMILES after scoring
    smi_set, per_target = {}, {}
    for r in recs:
        if r["uniprot"] not in used:
            continue
        lab = {}
        for kind, v in (("actives", 1), ("decoys", 0)):
            for m in r[kind]:
                smi_set.setdefault(m["smiles"], None)
                lab[m["smiles"]] = v
        per_target[r["uniprot"]] = lab
    smiles = list(smi_set)
    open(f"{info}/smiles.txt", "w").write("\n".join(smiles) + "\n")

    from collections import Counter
    print(f"[{layer}] 靶点 {len(ups)}（{dict(Counter(v['kind'] for v in used.values()))}）"
          f"，缺结构 {len(missing)}；唯一分子 {len(smiles):,}", flush=True)
    return ups, per_target, used, missing


def run_and_collect(layer, ds_dir, ds_rel, out_dir, per_target, gpu):
    res_root = f"{ds_dir}/_results"
    if os.path.exists(res_root):
        shutil.rmtree(res_root)
    env = dict(os.environ,
               LD_LIBRARY_PATH="/data/work/envs/conglude/lib",
               CUDA_VISIBLE_DEVICES=str(gpu))
    # Must pass a relative path of the form ./data/...: for any path not
    # starting with data/, predict.py silently substitutes
    # <dataset_dir>/ConGLUDe/data as the data root instead
    p = subprocess.run([PY, "predict.py", "--dataset_dir", ds_rel,
                        "--results_dir", res_root, "--num_workers", "8", "--overwrite"],
                       cwd=CG, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        print(p.stdout[-4000:], file=sys.stderr)
        raise SystemExit(f"[{layer}] ConGLUDe 失败 (returncode={p.returncode})")

    vs = np.load(f"{res_root}/predictions/vs_predictions.npy")
    names = [x.strip() for x in open(f"{res_root}/embeddings/protein_names.txt") if x.strip()]
    i2s = json.load(open(f"{ds_dir}/processed/ligand_embeddings/index2smiles.json"))
    lig_row = {i2s[k]: int(k) for k in i2s}      # smiles -> row index
    print(f"[{layer}] VS 矩阵 {vs.shape}，蛋白 {len(names)}，分子 {len(lig_row):,}", flush=True)
    # Note: the matrix is **ligand x protein** (predict.py computes
    # vs_preds = ligands @ proteins.T). ConGLUDe's README says "rows
    # correspond to protein names", which doesn't match the code -- the
    # code is authoritative here.
    if vs.shape != (len(lig_row), len(names)):
        raise SystemExit(f"[{layer}] VS 矩阵形状 {vs.shape} 与 "
                         f"(分子 {len(lig_row)}, 蛋白 {len(names)}) 不符，拒绝继续")

    n_ok = 0
    for pcol, up in enumerate(names):
        lab = per_target.get(up)
        if not lab:
            continue
        s, l = [], []
        for smi, y in lab.items():
            r = lig_row.get(smi)
            if r is None:
                continue
            s.append(float(vs[r, pcol]))
            l.append(y)
        if not s or sum(l) in (0, len(l)):
            continue
        d = f"{out_dir}/{layer}/{up}"
        os.makedirs(d, exist_ok=True)
        np.save(f"{d}/saved_preds.npy", np.asarray(s, dtype=np.float32))
        np.save(f"{d}/saved_labels.npy", np.asarray(l, dtype=np.int8))
        n_ok += 1
    print(f"[{layer}] 落盘 {n_ok} 个靶点", flush=True)
    return n_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="+", default=["L3", "L4", "L1", "L2"])
    ap.add_argument("--out_dir", default=f"{B}/results/t3/conglude")
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    man = json.load(open(f"{B}/data/t3/pockets/pdb_pocket_manifest.json"))["manifest"]
    pdb_choice = {u: v["pdb_id"] for u, v in man.items()}
    boltz_idx = build_boltz_index()
    print(f"可用：实验结构候选 {len(pdb_choice):,}，Boltz-2 结构 {len(boltz_idx):,}", flush=True)

    report = {}
    for L in args.layers:
        p = f"{B}/data/t3/eval/{L}.jsonl"
        if not os.path.exists(p):
            print(f"[{L}] 评测集不存在，跳过", flush=True)
            continue
        recs = [json.loads(l) for l in open(p)]
        if args.limit:
            recs = recs[:args.limit]
        ds_rel = f"./data/datasets/predict_datasets/t3_{L}"
        ds = f"{CG}/data/datasets/predict_datasets/t3_{L}"
        ups, per_target, used, missing = prepare(L, recs, ds, boltz_idx, pdb_choice, args.workers)
        if not ups:
            continue
        n = run_and_collect(L, ds, ds_rel, args.out_dir, per_target, args.gpu)
        report[L] = {"n_scored": n, "structure_source": used, "missing": missing}

    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(report, open(f"{args.out_dir}/structure_manifest.json", "w"), indent=1)


if __name__ == "__main__":
    main()
