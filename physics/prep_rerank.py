"""T6 串联 rerank 的输入准备：检索模型 top-N → Boltz-2 逐个重打分。

要回答什么
----------
T6 到目前为止只证明了「Boltz-2 排序强」（FEP 16 体系 ρ=0.615 vs 检索 0.28–0.40）。
但真实虚筛流程关心的是另一件事：**先用检索粗筛、再用物理精排，比单用检索强吗？**
这是唯一能给出方法学建议、而不只是评测数字的实验。

为什么这样设计
--------------
· 靶点选 L4（训练后才出现的新靶点）——rerank 的价值就在模型最不熟的地方
· 只要结构质量 A/B 级：口袋不可信的话，物理重排必然崩，那测的就不是 rerank 本身
· 类别尽量分散：激酶/GPCR/表观等，避免结论只在一类靶点上成立
· 每靶点取检索模型的 top-N（默认 50）——正是真实流程会送去精算的规模，
  而且 top-N 里通常有若干真 active，重排才有得比

MSA 复用
--------
Boltz-2 默认每条记录都去 MSA 服务器要一次比对。同一个靶点 50 个配体，
蛋白序列完全一样，重复 50 次既慢又容易被限流。
之前跑 T3 结构时已经为每个 UniProt 生成过 MSA（msa/<uniprot>_0.csv），
这里直接在 yaml 里引用，一次服务器请求都不发。

⚠️ 注意 rerank 的评价口径：只能在 **top-N 这个子集内部**比较
「检索原序 vs Boltz 重排」，不能拿来和全库 EF 直接比——
子集里的 active 比例已经被粗筛抬高了。
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
    """uniprot -> 已有的 MSA csv 路径（之前跑 T3 结构时生成的）。"""
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
    """模型看到的分子顺序 -> [smiles]；对不上返回 None。

    标签**不**从这里出——用模型自己的 saved_labels.npy，它与打分同序。
    早先版本拿评测集的 active SMILES 做字符串匹配，规范化不一致导致
    大部分 active 被误判成 decoy（top-50 里真值 269 个，匹配只认出 36 个）。
    """
    jsonl = [m["smiles"] for m in rec["actives"]] + \
            [m["smiles"] for m in rec["decoys"]]
    if len(jsonl) == n_pred:
        return jsonl
    p = f"{B}/data/T3_6A/{L}/{up}/{up}_lig.lmdb"
    if not os.path.exists(p):
        return None
    # 必须按游标序读：key 是字符串，模型侧遍历得到的是字典序
    # （0, 1, 10, 100, ...），不是数值序。按数值下标读会整体错位。
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
    # 截断表只记 beg/end（1-based，闭区间），序列在这里按坐标切出来
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

    # 候选：高质量结构 + 有 MSA + 有序列 + active 够多
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

    # 按类别轮转挑选，保证类别分散
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
            smi, is_act = order[idx], int(y[idx])   # 标签只认 saved_labels
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
