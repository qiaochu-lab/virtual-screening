"""为「有 PDB 条目但截不出可用口袋」的靶点补 Boltz-2 输入。

这批靶点的来历：它们在 PDB 里确实有结构，但所有候选共晶配体都不接触
该靶点自己的链——多半是核糖体、蛋白酶体、呼吸链这类大复合物里的亚基，
配体结合在别的亚基上。这种情况下 PDB 结构给不出该靶点的口袋，只能预测。

沿用 prep_boltz.py 的约定：每个靶点取亲和力最高的配体做代表，
序列 >1170aa 的单列（Boltz-2 在本机实测的上限，且多为多聚蛋白，
按 hivpr 的教训应先按结构域截取）。
"""
import json
import os

B = "/data/yicheng/xqc/vs-benchmark"
OUT = f"{B}/boltz_gap"
LIMIT = 1170
SHARDS = 4                      # 最多用 4 张卡

os.makedirs(OUT, exist_ok=True)

need = set(json.load(open(f"{B}/data/t3/need_boltz2.json")))
seqs = json.load(open(f"{B}/data/t3/sequences.json"))
print(f"待补靶点: {len(need):,}")

best = {}
for L in ["L3", "L4"]:
    for line in open(f"{B}/data/t3/layers/{L}.jsonl"):
        d = json.loads(line)
        u = d["uniprot"]
        if u not in need:
            continue
        try:
            paff = float(d["paff"])
        except (TypeError, ValueError):
            continue
        if u not in best or paff > best[u][0]:
            best[u] = (paff, d["smiles"])

ok, too_long, no_seq, no_lig = [], [], [], []
for u in sorted(need):
    s = (seqs.get(u) or {}).get("seq")
    if not s:
        no_seq.append(u)
        continue
    if u not in best:
        no_lig.append(u)
        continue
    if len(s) > LIMIT:
        too_long.append((u, len(s)))
        continue
    ok.append((u, s, best[u][1]))

# 按序列长度轮转分片，让各卡负载均衡（Boltz-2 耗时随长度陡增）
ok.sort(key=lambda x: -len(x[1]))
for i in range(SHARDS):
    os.makedirs(f"{OUT}/shard_{i}", exist_ok=True)
for n, (u, seq, smi) in enumerate(ok):
    y = ("version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: %s\n"
         "  - ligand:\n      id: B\n      smiles: '%s'\n"
         "properties:\n  - affinity:\n      binder: B\n" % (seq, smi))
    open(f"{OUT}/shard_{n % SHARDS}/{u}.yaml", "w").write(y)

json.dump({"ok": [u for u, _, _ in ok],
           "too_long": [{"uniprot": u, "length": l} for u, l in too_long],
           "no_seq": no_seq, "no_ligand": no_lig},
          open(f"{B}/data/t3/boltz_gap_manifest.json", "w"), indent=1)

print(f"  生成输入   : {len(ok):,}  （分 {SHARDS} 片）")
print(f"  序列过长   : {len(too_long):,}  (>{LIMIT}aa，需先按结构域截取)")
print(f"  无序列     : {len(no_seq):,}")
print(f"  无代表配体 : {len(no_lig):,}")
for i in range(SHARDS):
    print(f"    shard_{i}: {len(os.listdir(f'{OUT}/shard_{i}')):,}")
