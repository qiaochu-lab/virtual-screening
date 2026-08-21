"""把三个标准基准转成 T3 评测集同样的格式，好让另外三个模型直接跑 T1。

为什么要转
----------
ConGLUDe / ConPLex / SPRINT 不吃 UniMol 那套口袋 lmdb——
它们要的是序列、`.pdb` 结构、SaProt 3Di 序列。
但我们已经有三个能跑 T3 的 runner，而 T3 评测集的格式是
    {"uniprot": ..., "actives": [{"smiles":...}], "decoys": [{"smiles":...}]}
只要把 DUD-E / DEKOIS / LIT-PCBA 也写成这个格式，三个 runner 换个
`--eval` 路径就能跑 T1，不用各写一遍。

靶点身份从哪来
--------------
test_datasets 里的 dude.json / dekois.json / PCBA.json 是
[UniProt, PDB, 名称] 三元组，102 / 81 / 15 个，正好给出：
  · UniProt → 序列（ConPLex 用）
  · PDB     → 结构（ConGLUDe 用 .pdb，SPRINT 用它做 3Di）
DUD-E 每个靶点目录里本来就带 receptor.pdb，省一次下载。

⚠️ LIT-PCBA 单个靶点最多 36 万个分子，全量跑 ConPLex/SPRINT 代价很大。
默认按 --max-decoys 抽样（对 active 不抽），并把抽样比例记进产出文件，
算 EF 时按实际比例算，不能直接和全量的数字比。
"""
import argparse
import json
import os
import pickle
import random

import lmdb

B = "/data/work/vs-benchmark"
TD = f"{B}/code/LigUnity/test_datasets"


def read_lmdb(path):
    """按游标序读——和模型侧看到的顺序一致（key 是字符串，字典序）。"""
    e = lmdb.open(path, subdir=False, readonly=True, lock=False)
    out = []
    with e.begin() as t:
        for _k, v in t.cursor():
            d = pickle.loads(v)
            smi = d.get("smi")
            if smi is None:
                continue
            # LIT-PCBA 有些 smi 后面跟了个 ID，用空格分开
            out.append((smi.split()[0], int(d.get("label", 0))))
    e.close()
    return out


def targets(bench):
    j = {"DUDE": "dude.json", "DEKOIS": "dekois.json", "PCBA": "PCBA.json"}[bench]
    return json.load(open(f"{TD}/{j}"))


def lig_path(bench, name):
    if bench == "DUDE":
        return f"{TD}/DUD-E/{name.lower()}/mols.lmdb"
    if bench == "DEKOIS":
        return f"{TD}/DEKOIS_2.0x/{name}/{name}_lig.lmdb"
    return f"{TD}/lit_pcba/{name}/mols.lmdb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["DUDE", "DEKOIS", "PCBA"])
    ap.add_argument("--max-decoys", type=int, default=20000,
                    help="每个靶点最多保留多少 decoy，0=不限")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    out = args.out or f"{B}/data/t1/{args.bench}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    rows, miss, meta = [], [], []
    for up, pdb, name in targets(args.bench):
        # DEKOIS 的目录名是小写，DUD-E 用第三列的小写，LIT-PCBA 用原名
        cands = [name, name.lower()]
        p = None
        for c in cands:
            q = lig_path(args.bench, c)
            if os.path.exists(q):
                p = q
                break
        if p is None:
            miss.append((up, pdb, name, "找不到配体 lmdb"))
            continue
        mols = read_lmdb(p)
        act = [{"smiles": s} for s, y in mols if y == 1]
        dec = [{"smiles": s} for s, y in mols if y == 0]
        if len(act) < 5 or len(dec) < 10:
            miss.append((up, pdb, name, f"active {len(act)} decoy {len(dec)} 太少"))
            continue
        n_dec_all = len(dec)
        if args.max_decoys and len(dec) > args.max_decoys:
            dec = rng.sample(dec, args.max_decoys)
        rows.append({"uniprot": up, "pdb": pdb, "name": name,
                     "n_actives": len(act), "n_decoys": len(dec),
                     "n_decoys_full": n_dec_all,
                     "decoy_sampling": len(dec) / n_dec_all,
                     "actives": act, "decoys": dec})
        meta.append((name, len(act), n_dec_all, len(dec)))

    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"{args.bench}: 写出 {len(rows)} 个靶点 -> {out}")
    sampled = [m for m in meta if m[3] < m[2]]
    print(f"  active 合计 {sum(m[1] for m in meta):,}   "
          f"decoy 合计 {sum(m[3] for m in meta):,}（原始 {sum(m[2] for m in meta):,}）")
    if sampled:
        print(f"  ⚠️ {len(sampled)} 个靶点的 decoy 被抽样，"
              f"最低保留比例 {min(m[3]/m[2] for m in sampled):.1%}"
              "——EF 要按实际比例算，不能直接和全量数字比")
    for u, p, n, why in miss:
        print(f"  跳过 {n}: {why}")


if __name__ == "__main__":
    main()
