"""让 ConPLex 也跑 CASF-2016，把 T2 第三套数据的模型数从 4 个补到 5 个。

为什么它能跑而别的不能
----------------------
CASF 只需要「蛋白 + 配体 → 一个分数」，而 `casf_label_seq.json` 里
285 个复合物各自带 uniprot、完整序列、配体 SMILES 和实测 act。
ConPLex 恰好只吃序列 + SMILES，所以不需要任何额外准备。
ConGLUDe 要 .pdb（可按 PDB ID 下 285 个，工作量中等）；
SPRINT 要在这些结构上跑 foldseek 出 3Di；
DrugCLIP/BindCLIP 的仓库里根本没有 CASF 分支，要移植。这三类都不是"顺手"。

两个口径与 score_casf.py 保持一致
---------------------------------
· scoring power  285 个复合物一起算相关（跨靶点，考绝对亲和力）
· ranking power  按 uniprot 分组（68 个蛋白）算靶点内 Spearman 再平均
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
MAX_LEN = 2000          # ProtBert 上下文限制，与 T3/T1 那边一致


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

    # 输出列序是 分子ID -> 蛋白ID -> 分数（与 T3 那边同一个坑）
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
