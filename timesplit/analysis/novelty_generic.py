"""按任意一份训练配体 SMILES 清单，算 T3 分子的最大 Tanimoto 新颖度。

口径与 ligand_novelty.py / novelty_drugclip.py 完全一致：
ECFP4（Morgan r=2, fpSize=2048），对训练配体取最大 Tanimoto，
这样几套缓存可以并排读。

为什么每个模型要一套：三套训练集的配体空间差两个数量级
（DrugCLIP 13,590 / PocketAffDB 428,767 / SPRINT 1,390,031），
用别人的训练集给一个模型划新颖度档，档位标签是没有意义的。
"""
import argparse, json, os
from concurrent.futures import ProcessPoolExecutor

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")

B = "/data/work/vs-benchmark"
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
TRAIN_FPS = None


def fp(smi):
    m = Chem.MolFromSmiles(smi)
    return GEN.GetFingerprint(m) if m else None


def _init(fps):
    global TRAIN_FPS
    TRAIN_FPS = fps


def max_sim(smi):
    f = fp(smi)
    if f is None:
        return -1.0
    return max(DataStructs.BulkTanimotoSimilarity(f, TRAIN_FPS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-smiles", required=True, help="一行一个 SMILES")
    ap.add_argument("--eval-dir", default=f"{B}/data/t3/eval")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tsmi = sorted({l.strip() for l in open(args.train_smiles) if l.strip()})
    print(f"训练配体 {len(tsmi):,}，建指纹…", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        tfps = [f for f in ex.map(fp, tsmi, chunksize=500) if f is not None]
    print(f"  有效指纹 {len(tfps):,}", flush=True)

    mols = set()
    for L in ("L1", "L2", "L3", "L4"):
        for line in open(f"{args.eval_dir}/{L}.jsonl"):
            r = json.loads(line)
            for g in ("actives", "decoys"):
                for x in r[g]:
                    mols.add(x["smiles"])
    smis = sorted(mols)
    print(f"T3 唯一分子 {len(smis):,}，算最大 Tanimoto…", flush=True)

    nov = {}
    with ProcessPoolExecutor(args.workers, initializer=_init,
                             initargs=(tfps,)) as ex:
        for i, (s, v) in enumerate(zip(smis, ex.map(max_sim, smis, chunksize=100),
                                       strict=True)):
            nov[s] = v
            if (i + 1) % 10000 == 0:
                print(f"  {i+1:,}/{len(smis):,}", flush=True)
    json.dump(nov, open(args.out, "w"))
    print(f"写入 {args.out}")

    import numpy as np
    v = np.array([x for x in nov.values() if x >= 0])
    print("中位 %.3f | ≥0.7 %.1f%% | ≥0.5 %.1f%% | <0.35 %.1f%%" %
          (np.median(v), 100*(v >= 0.7).mean(), 100*(v >= 0.5).mean(),
           100*(v < 0.35).mean()))


if __name__ == "__main__":
    main()
