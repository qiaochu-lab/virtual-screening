"""Pre-warm MSAs for the L1/L2 rerank targets -- no GPU needed.

Why this step is done separately
-------------------------------------
T6 currently has only a negative result on L4, which cannot distinguish
"the physics-rerank idea does not work" from "the idea works, but fails
under novel targets + predicted structures + cross-series ligands". Adding
L1/L2 settles this. And L1's recall@50 is 64.1% (L4 is only 17.5%), so the
shortlist there actually holds most of the actives, which is what makes the
measurement meaningful.

The blocker: the existing 934 MSAs were all produced incidentally while
running T3's structure prediction, and structure prediction is only needed
for **novel targets that have no experimental structure** -- i.e. L3/L4.
L1/L2 are old targets that already have crystal structures, so no MSA was
ever generated for them.
Without an MSA, Boltz-2 can only fall back to single-sequence mode, which
sharply degrades structure quality -- and then the measurement would no
longer be about rerank itself.

This calls boltz's internal compute_msa directly (the same function predict
uses), issuing only the MSA server request without any structure prediction,
so it is **pure CPU, no GPU used** -- it can be done ahead of time while
others have the GPUs fully occupied.

Output: {out}/{uniprot}_0.csv; the rerank yaml's msa: field then points
straight to it, and the real run issues zero server requests.
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

B = "/data/work/vs-benchmark"
sys.path.insert(0, "/data/work/envs/boltz2/lib/python3.11/site-packages")
from boltz.main import compute_msa  # noqa: E402

MSA_URL = "https://api.colabfold.com"


def pick_targets(layer, topn, min_hits, max_hits, model, seqs, need):
    """Pick targets with a sparse shortlist, using exactly the same convention as the L4 version, to keep it comparable."""
    ev = {json.loads(x)["uniprot"]: json.loads(x)
          for x in open(f"{B}/data/t3/eval/{layer}.jsonl")}
    root = f"{B}/results/t3_raw/{model}/T3/{layer}"
    out = []
    for up in sorted(os.listdir(root)):
        if up not in ev or up not in seqs:
            continue
        try:
            p = np.load(f"{root}/{up}/saved_preds.npy").reshape(-1)
            y = np.load(f"{root}/{up}/saved_labels.npy")
        except Exception:
            continue
        if len(p) != len(y) or y.sum() < 5:
            continue
        hits = int(y[np.argsort(-p)[:topn]].sum())
        if hits < min_hits or hits > max_hits:
            continue
        if len(seqs[up]) > 1170:
            continue
        out.append((up, hits))
        if len(out) >= need:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="+", default=["L1", "L2"])
    ap.add_argument("--per-layer", type=int, default=15)
    ap.add_argument("--topn", type=int, default=50)
    ap.add_argument("--min-hits", type=int, default=1)
    ap.add_argument("--max-hits", type=int, default=6)
    ap.add_argument("--model", default="ligunity_protein_ranking")
    ap.add_argument("--out", default=f"{B}/data/t3/msa_l1l2")
    args = ap.parse_args()

    seqs = {k: v["seq"] for k, v in json.load(open(f"{B}/data/t3/sequences.json")).items()}
    os.makedirs(args.out, exist_ok=True)

    picked = {}
    for L in args.layers:
        got = pick_targets(L, args.topn, args.min_hits, args.max_hits,
                           args.model, seqs, args.per_layer)
        picked[L] = got
        print(f"[{L}] 选中 {len(got)} 个靶点（top-{args.topn} 里 active 数 "
              f"{args.min_hits}–{args.max_hits}）: "
              + ", ".join(f"{u}({h})" for u, h in got[:8])
              + (" ..." if len(got) > 8 else ""), flush=True)

    todo = [(L, u) for L in args.layers for u, _ in picked[L]]
    print(f"\n共 {len(todo)} 个靶点要生成 MSA（纯 CPU，只发服务器请求）\n", flush=True)

    ok, fail = 0, []
    for i, (L, up) in enumerate(todo, 1):
        dest = f"{args.out}/{up}_0.csv"
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            ok += 1
            continue
        try:
            # msa_dir must be a Path: compute_msa builds paths internally with /
            compute_msa(
                data={up: seqs[up]},
                target_id=up,
                msa_dir=Path(args.out),
                msa_server_url=MSA_URL,
                msa_pairing_strategy="greedy")
            ok += 1
            print(f"  [{i}/{len(todo)}] {L} {up} ✓", flush=True)
        except Exception as e:
            fail.append((up, str(e)[:80]))
            print(f"  [{i}/{len(todo)}] {L} {up} ✗ {str(e)[:80]}", flush=True)
        time.sleep(1)          # don't hammer the MSA server too fast

    json.dump({L: [{"uniprot": u, "hits": h} for u, h in picked[L]] for L in picked},
              open(f"{B}/data/t3/rerank_l1l2_targets.json", "w"), indent=1)
    print(f"\nMSA 完成 {ok}/{len(todo)}，失败 {len(fail)}")
    for u, e in fail[:5]:
        print("   ", u, e)


if __name__ == "__main__":
    main()
