"""对 A 组(DrugCLIP/BindCLIP)自己的训练配体算 T3 分子的新颖度。

为什么需要
----------
现有的 `data/t3/ligand_novelty.json` 是对**亲和力半**的 428,767 个配体算的
(`train_label_blend_seq_full.json`)。但 DrugCLIP / BindCLIP ×2 训练在
`train_no_test_af` 上,配体是那 66,164 条 pair 里的分子——去重后 13,590 个,
是另一批,而且小 31.6 倍。

所以凡是按新颖度分档的分析(T2 的 t2_novelty_tiers、T3 的 novelty_tiered_ef、
纯配体学习基线、2×2),对 A 组三个模型都只是近似。这个脚本补上 A 组自己的缓存。

口径与 ligand_novelty.py 完全一致:ECFP4(Morgan r=2, fpSize=2048)、
对训练配体取最大 Tanimoto,这样两份缓存可以并排读。
"""
import argparse, json, os, pickle
from concurrent.futures import ProcessPoolExecutor

import lmdb
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
    ap.add_argument("--train-lmdb", default=f"{B}/data/train_no_test_af/train.lmdb")
    ap.add_argument("--eval-dir", default=f"{B}/data/t3/eval")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=f"{B}/data/t3/ligand_novelty_drugclip.json")
    args = ap.parse_args()

    e = lmdb.open(args.train_lmdb, subdir=False, readonly=True, lock=False,
                  readahead=False)
    tsmi = set()
    with e.begin() as t:
        for _k, v in t.cursor():
            d = pickle.loads(v)
            if d.get("smi"):
                tsmi.add(d["smi"])
    e.close()
    tsmi = sorted(tsmi)
    print(f"A 组训练配体去重 {len(tsmi):,},建指纹…", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        tfps = [f for f in ex.map(fp, tsmi, chunksize=200) if f is not None]
    print(f"  有效指纹 {len(tfps):,}", flush=True)

    mols = set()
    for L in ("L1", "L2", "L3", "L4"):
        for line in open(f"{args.eval_dir}/{L}.jsonl"):
            r = json.loads(line)
            for g in ("actives", "decoys"):
                for x in r[g]:
                    mols.add(x["smiles"])
    smis = sorted(mols)
    print(f"T3 唯一分子 {len(smis):,},算最大 Tanimoto…", flush=True)

    nov = {}
    with ProcessPoolExecutor(args.workers, initializer=_init,
                             initargs=(tfps,)) as ex:
        for i, (s, v) in enumerate(zip(smis, ex.map(max_sim, smis, chunksize=200))):
            nov[s] = v
            if (i + 1) % 20000 == 0:
                print(f"  {i+1:,}/{len(smis):,}", flush=True)
    json.dump(nov, open(args.out, "w"))
    print(f"写入 {args.out}")

    import numpy as np
    v = np.array([x for x in nov.values() if x >= 0])
    old = json.load(open(f"{B}/data/t3/ligand_novelty.json"))
    o = np.array([x for x in old.values() if x >= 0])
    print("\n两份缓存对比(全部 T3 分子的最大 Tanimoto 分布)")
    print("%-22s %8s %8s %8s %8s" % ("训练配体来源", "n", "中位", "≥0.7 占比", "<0.35 占比"))
    for name, a in (("A 组 13,590", v), ("亲和力半 428,767", o)):
        print("%-22s %8d %8.3f %8.1f%% %9.1f%%" %
              (name, len(a), np.median(a), 100*(a >= 0.7).mean(), 100*(a < 0.35).mean()))


if __name__ == "__main__":
    main()
