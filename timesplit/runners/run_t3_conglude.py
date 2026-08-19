"""在 T3 评测集上跑 ConGLUDe，输出统一评测层要的原始分数。

ConGLUDe 的接口
---------------
predict.py 吃 `info/protein_ids.txt` + `info/smiles.txt`，
吐 `vs_predictions.npy` 相似度矩阵。

⚠️ 矩阵方向：官方 README 写「rows 对应蛋白名」，**与代码不符**。
predict.py 里是 `vs_preds = encoded_ligands_b @ protein_embeddings.t()`，
所以实际是 **配体（行） × 蛋白（列）**：
    行序对应 `processed/ligand_embeddings/index2smiles.json`
    列序对应 `embeddings/protein_names.txt`
按 README 写会直接 IndexError（我们就撞了一次），下面加了形状断言防止再错。

结构从哪来
----------
ConGLUDe 只认 `{protein_id}.pdb`，不认 mmCIF。所以：
  - 有 PDB 实验结构且能拿到传统 PDB 格式的靶点 → 用 RCSB 的 .pdb
  - 其余（超大结构没有传统格式的、本来就无 PDB 的）→ 用 Boltz-2 预测的 .pdb
两种来源都以 **UniProt 号**作为 protein_id 落盘，保证与其它模型的靶点口径一致，
同时在 manifest 里记下每个靶点用的是哪一种结构（holo 实验 vs 预测），
这本身是 T5 结构鲁棒性要用的分层变量。

一次跑一层：ConGLUDe 会把该层所有靶点 × 所有分子算成一个大矩阵，
而各靶点的分子集合不同，所以按靶点切分后只取自己那些列。
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
    """下载传统 PDB 格式；超大结构没有该格式，返回 False 由调用方回退。"""
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
    """一次性扫出所有 Boltz-2 结构，避免每个靶点走一遍 os.walk。"""
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

    # 先并发下载实验结构，拿不到的再回退预测结构
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

    # 全层唯一分子；每个靶点的标签在打分后按 SMILES 回填
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
    # 必须传 ./data/... 形式的相对路径：predict.py 对不以 data/ 开头的路径
    # 会自作主张改用 <dataset_dir>/ConGLUDe/data 作为数据根目录
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
    lig_row = {i2s[k]: int(k) for k in i2s}      # smiles -> 行号
    print(f"[{layer}] VS 矩阵 {vs.shape}，蛋白 {len(names)}，分子 {len(lig_row):,}", flush=True)
    # 注意：矩阵是 **配体 × 蛋白**（predict.py 里 vs_preds = ligands @ proteins.T）。
    # ConGLUDe 的 README 写的是「rows 对应蛋白名」，与代码不符——以代码为准。
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
