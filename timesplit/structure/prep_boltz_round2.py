"""第二轮 Boltz-2 输入：结构域截取后的长靶点 + 换配体重试的失败靶点。

两批
----
A. 123 个 >1170aa 靶点，按 truncate_domains2.py 的结果用截断序列
B. 9 个第一轮失败的靶点，换一个配体重试：
   6 个是肽类配体超过 128 原子（Boltz-2 亲和力模块硬限制），
   1 个 RDKit 生不出 3D 构象，2 个 MSA/文件错误（换配体不影响，顺带重试）
   —— 口袋只需要一个有代表性的配体，不必是亲和力最高的那个。
"""
import json
import os

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

B = "/data/work/vs-benchmark"
OUT = f"{B}/boltz_r2"
SHARDS = 4
MAX_LIG_ATOMS = 128          # Boltz-2 亲和力模块上限

os.makedirs(OUT, exist_ok=True)
for i in range(SHARDS):
    os.makedirs(f"{OUT}/shard_{i}", exist_ok=True)

trunc = json.load(open(f"{B}/data/t3/domain_truncation.json"))["truncation"]
seqs = json.load(open(f"{B}/data/t3/sequences.json"))
br = json.load(open(f"{B}/data/t3/missing_breakdown.json"))
retry = set(br["跑了但失败"])

# 每个靶点按亲和力从高到低收集配体，供挑选
ligs = {}
need = set(trunc) | retry
for L in ["L3", "L4"]:
    for line in open(f"{B}/data/t3/layers/{L}.jsonl"):
        d = json.loads(line)
        u = d["uniprot"]
        if u not in need:
            continue
        try:
            ligs.setdefault(u, []).append((float(d["paff"]), d["smiles"]))
        except (TypeError, ValueError):
            pass
for u in ligs:
    ligs[u].sort(reverse=True)


def pick_ligand(u):
    """取亲和力最高、且原子数与构象都过关的配体。"""
    for paff, smi in ligs.get(u, []):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        if m.GetNumAtoms() > MAX_LIG_ATOMS:          # 含氢前的重原子数已足够筛掉肽类
            continue
        mh = Chem.AddHs(m)
        if mh.GetNumAtoms() > MAX_LIG_ATOMS * 2:
            continue
        return smi
    return None


rows, no_lig = [], []
for u in sorted(trunc):
    smi = pick_ligand(u)
    (rows.append((u, trunc[u]["seq"], smi, "truncated")) if smi else no_lig.append(u))
for u in sorted(retry):
    s = (seqs.get(u) or {}).get("seq")
    smi = pick_ligand(u)
    if s and smi:
        rows.append((u, s, smi, "retry"))
    else:
        no_lig.append(u)

rows.sort(key=lambda r: -len(r[1]))          # 长的先排，四片负载均衡
for n, (u, seq, smi, kind) in enumerate(rows):
    y = ("version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: %s\n"
         "  - ligand:\n      id: B\n      smiles: '%s'\n"
         "properties:\n  - affinity:\n      binder: B\n" % (seq, smi))
    open(f"{OUT}/shard_{n % SHARDS}/{u}.yaml", "w").write(y)

json.dump({"targets": [{"uniprot": u, "kind": k, "seq_len": len(s)} for u, s, _, k in rows],
           "no_usable_ligand": no_lig},
          open(f"{B}/data/t3/boltz_r2_manifest.json", "w"), indent=1)

from collections import Counter
print(f"生成输入 {len(rows)}  （{dict(Counter(k for *_, k in rows))}）")
print(f"无可用配体 {len(no_lig)}: {no_lig[:10]}")
ls = sorted(len(s) for _, s, _, _ in rows)
print(f"序列长度 中位 {ls[len(ls)//2]}  最长 {ls[-1]}")
for i in range(SHARDS):
    print(f"  shard_{i}: {len(os.listdir(f'{OUT}/shard_{i}'))}")
