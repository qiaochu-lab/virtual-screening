"""Input preparation for T6's cascade rerank: retrieval model's top-N -> Boltz-2 rescores each one.

What this answers
-------------------
So far T6 has only shown that "Boltz-2 ranks well" (rho=0.615 on the 16 FEP
systems vs. retrieval's 0.28-0.40). But a real virtual-screening pipeline
cares about a different question: **is coarse retrieval followed by physics
reranking better than retrieval alone?** This is the only experiment that
yields a methodological recommendation rather than just another evaluation
number.

Why designed this way
-----------------------
. Targets are drawn from L4 (targets that only appeared after training) --
  that is exactly where reranking would add value, since it is where the
  model is least familiar
. Structure quality restricted to grade A/B: if the pocket cannot be
  trusted, physics reranking necessarily fails, and the measurement is no
  longer about rerank itself
. Classes kept as diverse as possible: kinases/GPCRs/epigenetic targets
  etc., to avoid a conclusion that only holds for one target class
. Each target takes the retrieval model's top-N (default 50) -- exactly the
  scale a real pipeline would send for detailed scoring, and the top-N
  usually contains some genuine actives, which is what gives reranking
  something to compare against

MSA reuse
-----------
By default Boltz-2 requests an alignment from the MSA server for every
single record. With 50 ligands on the same target sharing an identical
protein sequence, repeating that 50 times is both slow and likely to get
rate-limited.
MSAs were already generated per UniProt (msa/<uniprot>_0.csv) when the T3
structures were run; here they are referenced directly in the yaml, without
issuing a single new server request.

Warning: mind rerank's evaluation scope: "retrieval's original order vs.
Boltz reranking" can only be compared **within the top-N subset itself**,
not directly against the full-pool EF -- the active fraction within the
subset has already been raised by coarse retrieval.
"""
import argparse
import json
import os
import pickle
from collections import Counter, defaultdict

import lmdb
import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
B = "/data/work/vs-benchmark"
MAX_ATOMS, MAX_LEN = 128, 1170


def msa_index():
    """uniprot -> path to an existing MSA csv (generated earlier when the T3 structures were run)."""
    out = {}
    for d in ["boltz_batch_out", "boltz_retry_out", "boltz_gap_out", "boltz_r2_out"]:
        root = f"{B}/{d}"
        if not os.path.isdir(root):
            continue
        for dp, _, fn in os.walk(root):
            if not dp.endswith("msa"):
                continue
            for f in fn:
                if f.endswith("_0.csv"):
                    out.setdefault(f[:-6], os.path.join(dp, f))
    return out


def lig_order(up, L, n_pred, rec):
    """The molecule order the model saw -> [smiles]; returns None on a mismatch.

    Labels **do not** come from here -- they come from the model's own
    saved_labels.npy, which is in the same order as the scores. An earlier
    version string-matched against the eval set's active SMILES, and
    inconsistent canonicalization misclassified most actives as decoys
    (269 true actives in the top-50, of which matching only recognised 36).
    """
    jsonl = [m["smiles"] for m in rec["actives"]] + \
            [m["smiles"] for m in rec["decoys"]]
    if len(jsonl) == n_pred:
        return jsonl
    p = f"{B}/data/T3_6A/{L}/{up}/{up}_lig.lmdb"
    if not os.path.exists(p):
        return None
    # must read in cursor order: the key is a string, and iterating on the model side gives
    # lexicographic order (0, 1, 10, 100, ...), not numeric order. Reading by numeric index would misalign everything.
    e = lmdb.open(p, subdir=False, readonly=True, lock=False)
    smis = []
    with e.begin() as t:
        for _k, v in t.cursor():
            smis.append(pickle.loads(v)["smi"])
    e.close()
    if len(smis) != n_pred:
        return None
    return smis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ligunity_protein_ranking",
                    help="拿哪个检索模型的 top-N 做粗筛（默认 L4 上 EF1% 最高的那个）")
    ap.add_argument("--layer", default="L4")
    ap.add_argument("--topn", type=int, default=50)
    ap.add_argument("--targets", type=int, default=20)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--out", default=f"{B}/boltz_rerank")
    args = ap.parse_args()

    seqs = {k: v["seq"] for k, v in json.load(open(f"{B}/data/t3/sequences.json")).items()}
    # the truncation table only records beg/end (1-based, closed interval); the sequence is sliced out here by coordinate
    tp = f"{B}/data/t3/domain_truncation.json"
    trunc = {}
    if os.path.exists(tp):
        for k, v in json.load(open(tp)).get("truncation", {}).items():
            if k in seqs:
                trunc[k] = seqs[k][v["beg"] - 1:v["end"]]
    hq = set(json.load(open(f"{B}/data/t3/target_quality.json"))["high_quality"])
    cls = json.load(open(f"{B}/data/t3/target_class.json"))["class"]
    msas = msa_index()
    print(f"可复用 MSA: {len(msas):,}   高质量靶点: {len(hq):,}")

    ev = {json.loads(x)["uniprot"]: json.loads(x)
          for x in open(f"{B}/data/t3/eval/{args.layer}.jsonl")}
    root = f"{B}/results/t3_raw/{args.model}/T3/{args.layer}"

    # candidates: high-quality structure + has an MSA + has a sequence + enough actives
    cand = []
    for up in sorted(os.listdir(root)):
        if up not in hq or up not in msas or up not in seqs or up not in ev:
            continue
        try:
            p = np.load(f"{root}/{up}/saved_preds.npy").reshape(-1)
            y = np.load(f"{root}/{up}/saved_labels.npy")
        except Exception:
            continue
        if len(p) != len(y) or y.sum() < 15:
            continue
        order = lig_order(up, args.layer, len(p), ev[up])
        if order is None:
            continue
        seq = trunc.get(up, seqs[up])
        if len(seq) > MAX_LEN:
            continue
        cand.append((up, p, y, order, seq))
    print(f"候选靶点: {len(cand)}")

    # round-robin selection by class, to keep classes diverse
    by_cls = defaultdict(list)
    for c in cand:
        by_cls[cls.get(c[0], "其他/未分类")].append(c)
    picked, i = [], 0
    while len(picked) < args.targets and any(by_cls.values()):
        for k in sorted(by_cls):
            if by_cls[k] and len(picked) < args.targets:
                picked.append(by_cls[k].pop(0))
        i += 1
        if i > 50:
            break
    print("选中类别分布:", dict(Counter(cls.get(c[0], "其他/未分类") for c in picked)))

    for s in range(args.shards):
        os.makedirs(f"{args.out}/shard_{s}", exist_ok=True)
    manifest, n, skipped = [], 0, Counter()
    for up, p, y, order, seq in picked:
        top = np.argsort(-p)[:args.topn]
        for rank, idx in enumerate(top):
            smi, is_act = order[idx], int(y[idx])   # label comes only from saved_labels
            m = Chem.MolFromSmiles(smi)
            if m is None:
                skipped["SMILES 解析失败"] += 1; continue
            if m.GetNumAtoms() > MAX_ATOMS:
                skipped[f">{MAX_ATOMS} 原子"] += 1; continue
            name = f"{up}__{rank:03d}"
            yml = ("version: 1\nsequences:\n  - protein:\n      id: A\n"
                   f"      sequence: {seq}\n      msa: {msas[up]}\n"
                   f"  - ligand:\n      id: B\n      smiles: '{smi}'\n"
                   "properties:\n  - affinity:\n      binder: B\n")
            open(f"{args.out}/shard_{n % args.shards}/{name}.yaml", "w").write(yml)
            manifest.append({"name": name, "uniprot": up, "rank": int(rank),
                             "pred": float(p[idx]), "label": int(is_act), "smi": smi})
            n += 1

    json.dump({"model": args.model, "layer": args.layer, "topn": args.topn,
               "targets": [c[0] for c in picked], "entries": manifest},
              open(f"{B}/data/t3/rerank_manifest.json", "w"), indent=1)
    print(f"\n生成 {n:,} 个输入（{len(picked)} 个靶点 × top-{args.topn}）")
    if skipped:
        print("跳过:", dict(skipped))
    act = sum(e["label"] for e in manifest)
    print(f"其中真 active {act} 个（占 {act/max(n,1)*100:.1f}%）"
          f"—— 粗筛已经把 active 比例从约 2% 抬到这里，rerank 就是在这个子集里比")
    for s in range(args.shards):
        print(f"  shard_{s}: {len(os.listdir(f'{args.out}/shard_{s}')):,}")


if __name__ == "__main__":
    main()
