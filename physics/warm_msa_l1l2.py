"""为 L1/L2 的 rerank 靶点预热 MSA —— 不需要 GPU。

为什么要单独做这一步
--------------------
T6 现在只有 L4 的负面结果，分不清是「物理重排这个思路不行」还是
「思路可行，但在新靶点 + 预测结构 + 跨系列配体下失效」。补 L1/L2 就能判定。
而 L1 的 recall@50 是 64.1%（L4 只有 17.5%），shortlist 在那里才装得下
大部分 active，测出来才有意义。

卡点是：现有的 934 份 MSA 全是跑 T3 结构预测时顺带产的，
而只有**没有实验结构的新靶点**才需要预测结构——也就是 L3/L4。
L1/L2 是旧靶点，本来就有晶体结构，从没生成过 MSA。
Boltz-2 没有 MSA 只能走单序列模式，结构质量大幅下降，那测的就不是 rerank 本身了。

这里直接调 boltz 内部的 compute_msa（就是 predict 时用的同一个函数），
只发 MSA 服务器请求、不做结构预测，所以**纯 CPU、不占卡**，
可以在别人占满 GPU 的时候先把这步做完。

产出：{out}/{uniprot}_0.csv，之后 rerank 的 yaml 直接 msa: 指过去，
正式跑的时候一次服务器请求都不发。
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
    """挑 shortlist 稀疏的靶点，口径与 L4 那一版完全一致，保证可比。"""
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
            # msa_dir 必须是 Path：compute_msa 内部用 / 拼路径
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
        time.sleep(1)          # 别把 MSA 服务器打太急

    json.dump({L: [{"uniprot": u, "hits": h} for u, h in picked[L]] for L in picked},
              open(f"{B}/data/t3/rerank_l1l2_targets.json", "w"), indent=1)
    print(f"\nMSA 完成 {ok}/{len(todo)}，失败 {len(fail)}")
    for u, e in fail[:5]:
        print("   ", u, e)


if __name__ == "__main__":
    main()
