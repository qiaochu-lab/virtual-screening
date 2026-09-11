"""Run ConPLex on the T3 evaluation set, producing the raw scores the unified evaluation layer needs.

Output format matches the DrugCLIP/BindCLIP/LigUnity patch:
    <out_dir>/<layer>/<uniprot>/saved_preds.npy    one score per molecule (higher = more likely active)
    <out_dir>/<layer>/<uniprot>/saved_labels.npy   1=active, 0=decoy
This lets docs/eval/metrics.py compute EF/AUROC/BEDROC uniformly with no changes needed.

Two quirks of ConPLex's interface (both bit us):
  - the input TSV is `protein_id  mol_id  sequence  SMILES` (no header)
  - **the output column order is reversed from the input**: `mol_id  protein_id  score`
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

B = "/data/work/vs-benchmark"
ENV = "/data/work/envs/conplex/bin/conplex-dti"
CKPT = f"{B}/ckpt/conplex/BindingDB_ExperimentalValidModel.pt"


def run_layer(layer, out_dir, work_dir, seqs, max_len, limit=None):
    recs = [json.loads(l) for l in open(f"{B}/data/t3/eval/{layer}.jsonl")]
    if limit:
        recs = recs[:limit]

    usable, skipped = [], {"无序列": 0, "序列过长": 0}
    for r in recs:
        s = (seqs.get(r["uniprot"]) or {}).get("seq")
        if not s:
            skipped["无序列"] += 1
        elif len(s) > max_len:
            skipped["序列过长"] += 1
        else:
            usable.append(r)
    print(f"[{layer}] 靶点 {len(recs)} → 可评 {len(usable)}；跳过 {skipped}", flush=True)
    if not usable:
        return

    # Write one big TSV in a single pass: ConPLex caches protein/molecule features by unique value, so batching is far faster than going target by target
    os.makedirs(work_dir, exist_ok=True)
    tsv = f"{work_dir}/{layer}_pairs.tsv"
    index = []                      # (uniprot, mol_id, label), one-to-one with the TSV rows
    with open(tsv, "w") as f:
        for r in usable:
            up = r["uniprot"]
            seq = seqs[up]["seq"]
            for kind, lab in (("actives", 1), ("decoys", 0)):
                for i, m in enumerate(r[kind]):
                    mid = f"{up}_{kind[0]}{i}"
                    f.write(f"{up}\t{mid}\t{seq}\t{m['smiles']}\n")
                    index.append((up, mid, lab))
    print(f"[{layer}] 待打分对: {len(index):,}  -> {tsv}", flush=True)

    out_tsv = f"{work_dir}/{layer}_out.tsv"
    env = dict(os.environ, HF_ENDPOINT="https://hf-mirror.com", HF_HOME=f"{B}/hf_cache")
    p = subprocess.run([ENV, "predict", "--data-file", tsv,
                        "--model-path", CKPT, "--outfile", out_tsv],
                       cwd=work_dir, env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0 or not os.path.exists(out_tsv):
        print(p.stdout[-3000:], file=sys.stderr)
        raise SystemExit(f"[{layer}] ConPLex 失败 (returncode={p.returncode})")

    # Output column order is mol_id -> protein_id -> score
    score = {}
    with open(out_tsv) as f:
        for line in f:
            p_ = line.rstrip("\n").split("\t")
            if len(p_) >= 3:
                score[p_[0]] = float(p_[2])

    per = {}
    for up, mid, lab in index:
        if mid not in score:
            continue
        per.setdefault(up, ([], []))
        per[up][0].append(score[mid])
        per[up][1].append(lab)

    n_ok = 0
    for up, (s, l) in per.items():
        if sum(l) == 0 or sum(l) == len(l):      # AUROC is undefined when every label is the same
            continue
        d = f"{out_dir}/{layer}/{up}"
        os.makedirs(d, exist_ok=True)
        np.save(f"{d}/saved_preds.npy", np.asarray(s, dtype=np.float32))
        np.save(f"{d}/saved_labels.npy", np.asarray(l, dtype=np.int8))
        n_ok += 1
    miss = len(index) - len(score)
    print(f"[{layer}] 落盘 {n_ok} 个靶点" + (f"；有 {miss:,} 对没拿到分数" if miss else ""),
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="+", default=["L4", "L3", "L2", "L1"])
    ap.add_argument("--out_dir", default=f"{B}/results/t3/conplex")
    ap.add_argument("--work_dir", default=f"{B}/tmp/conplex_t3")
    ap.add_argument("--max_len", type=int, default=2000,
                    help="ProtBert 上下文限制，超长序列跳过并记录")
    ap.add_argument("--limit", type=int, default=None, help="每层只跑前 N 个靶点（调试用）")
    args = ap.parse_args()

    seqs = json.load(open(f"{B}/data/t3/sequences.json"))
    for L in args.layers:
        if not os.path.exists(f"{B}/data/t3/eval/{L}.jsonl"):
            print(f"[{L}] 评测集尚未生成，跳过", flush=True)
            continue
        run_layer(L, args.out_dir, args.work_dir, seqs, args.max_len, args.limit)


if __name__ == "__main__":
    main()
