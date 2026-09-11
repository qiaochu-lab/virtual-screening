"""Run ConPLex on CASF-2016 too, bringing the model count for T2's third
dataset from 4 up to 5.

Why this one can run and the others can't
----------------------
CASF only needs "protein + ligand -> one score", and `casf_label_seq.json`
carries, for each of its 285 complexes, a uniprot, full sequence, ligand
SMILES, and measured act. ConPLex takes exactly sequence + SMILES as input,
so no extra preparation is needed.
ConGLUDe needs .pdb files (downloadable by PDB ID for all 285, a moderate
amount of work);
SPRINT needs foldseek run on these structures to produce 3Di;
DrugCLIP/BindCLIP's repos have no CASF branch at all and would need porting.
None of these three is a quick add.

Two conventions kept consistent with score_casf.py
---------------------------------
* scoring power: correlation computed jointly over all 285 complexes
  (cross-target, tests absolute affinity)
* ranking power: grouped by uniprot (68 proteins), within-target Spearman
  computed then averaged
"""
import json
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

B = "/data/work/vs-benchmark"
ENV = "/data/work/envs/conplex/bin/conplex-dti"
CKPT = f"{B}/ckpt/conplex/BindingDB_ExperimentalValidModel.pt"
LAB = f"{B}/code/LigUnity/test_datasets/casf_label_seq.json"
WORK = f"{B}/tmp/conplex_casf"
MAX_LEN = 2000          # ProtBert context limit, consistent with T3/T1


def main():
    recs = json.load(open(LAB))
    os.makedirs(WORK, exist_ok=True)
    tsv = f"{WORK}/casf_pairs.tsv"
    index, skipped = [], 0
    with open(tsv, "w") as f:
        for e in recs:
            seq = e["sequence"]
            if len(seq) > MAX_LEN:
                skipped += 1
                continue
            pdb = e["pockets"][0]
            for lig in e["ligands"]:
                mid = f"{pdb}"
                f.write(f"{pdb}\t{mid}\t{seq}\t{lig['smi']}\n")
                index.append((pdb, e["uniprot"], mid, float(lig["act"])))
    print(f"待打分 {len(index)} 条（跳过序列过长 {skipped} 个复合物）", flush=True)

    out_tsv = f"{WORK}/casf_out.tsv"
    env = dict(os.environ, HF_ENDPOINT="https://hf-mirror.com", HF_HOME=f"{B}/hf_cache")
    p = subprocess.run([ENV, "predict", "--data-file", tsv, "--model-path", CKPT,
                        "--outfile", out_tsv], cwd=WORK, env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0 or not os.path.exists(out_tsv):
        print(p.stdout[-2500:], file=sys.stderr)
        raise SystemExit(f"ConPLex 失败 (returncode={p.returncode})")

    # Output column order is molecule ID -> protein ID -> score (same pitfall as on the T3 side)
    score = {}
    for line in open(out_tsv):
        c = line.rstrip("\n").split("\t")
        if len(c) >= 3:
            score[c[0]] = float(c[2])

    pairs = [(score[mid], act, up) for pdb, up, mid, act in index if mid in score]
    if not pairs:
        raise SystemExit("一条分数都没对上")
    s = np.array([x[0] for x in pairs]); a = np.array([x[1] for x in pairs])
    print(f"\nCASF-2016 · ConPLex（n={len(pairs)}）")
    print(f"  打分能力 Spearman {stats.spearmanr(s, a).statistic:.3f}   "
          f"Pearson {stats.pearsonr(s, a).statistic:.3f}")

    per = defaultdict(list)
    for sc, act, up in pairs:
        per[up].append((sc, act))
    rhos = []
    for v in per.values():
        if len(v) >= 3 and np.std([x[0] for x in v]) > 0:
            r = stats.spearmanr([x[0] for x in v], [x[1] for x in v]).statistic
            if np.isfinite(r):
                rhos.append(r)
    print(f"  靶点内排序 Spearman {np.mean(rhos):.3f}（{len(rhos)} 个靶点）")
    json.dump({"n": len(pairs),
               "scoring_spearman": float(stats.spearmanr(s, a).statistic),
               "scoring_pearson": float(stats.pearsonr(s, a).statistic),
               "ranking_spearman": float(np.mean(rhos)), "n_targets": len(rhos)},
              open(f"{B}/results/casf_conplex.json", "w"), indent=1)


if __name__ == "__main__":
    main()
