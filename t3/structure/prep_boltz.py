"""为 494 个待预测靶点生成 Boltz-2 输入。

每个靶点取**亲和力最高**的配体作为代表——预测出的复合物用于后续按任意阈值
截取口袋（口袋阈值未定不影响这一步，结构本身与阈值无关）。

长度 >1170aa 的单独列出：Boltz-2 在本机实测的上限，且这类多为多聚蛋白，
按之前 hivpr 的教训应先按结构域截取，不在本批处理。
"""
import json, os, time, urllib.parse, urllib.request
from collections import defaultdict

B = "/data/work/vs-benchmark"
OUT = f"{B}/boltz_batch"
os.makedirs(OUT, exist_ok=True)

cov = json.load(open(f"{B}/data/t3/new_target_structure_coverage.json"))
need = sorted(set(cov["no_pdb"]))
print(f"待预测靶点: {len(need):,}", flush=True)

# 每个靶点取亲和力最高的配体
best = {}
for L in ["L3", "L4"]:
    for line in open(f"{B}/data/t3/layers/{L}.jsonl"):
        d = json.loads(line)
        u = d["uniprot"]
        if u in need and (u not in best or d["paff"] > best[u]["paff"]):
            best[u] = d
print(f"有代表配体的: {len(best):,}", flush=True)

# 批量取序列
seqs = {}
BATCH = 80
for i in range(0, len(need), BATCH):
    chunk = need[i:i + BATCH]
    q = " OR ".join("accession:%s" % a for a in chunk)
    url = ("https://rest.uniprot.org/uniprotkb/stream?format=tsv&fields=accession,sequence&query=%s"
           % urllib.parse.quote(q))
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                txt = r.read().decode()
            break
        except Exception:
            txt = ""
            time.sleep(3)
    for ln in txt.split("\n")[1:]:
        p = ln.split("\t")
        if len(p) >= 2 and p[1].strip():
            seqs[p[0]] = p[1].strip()
    print(f"  ...取序列 {min(i+BATCH, len(need)):,}/{len(need):,}", flush=True)

print(f"拿到序列: {len(seqs):,}", flush=True)

LIMIT = 1170
ok, too_long, no_seq, no_lig = [], [], [], []
for u in need:
    s = seqs.get(u)
    d = best.get(u)
    if not s:
        no_seq.append(u); continue
    if not d:
        no_lig.append(u); continue
    if len(s) > LIMIT:
        too_long.append((u, len(s))); continue
    y = ("version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: %s\n"
         "  - ligand:\n      id: B\n      smiles: '%s'\n"
         "properties:\n  - affinity:\n      binder: B\n" % (s, d["smiles"]))
    open(f"{OUT}/{u}.yaml", "w").write(y)
    ok.append((u, len(s)))

print()
print("=" * 58)
print(f"可直接预测       : {len(ok):,}")
print(f"超过 1170aa 上限 : {len(too_long):,}   ← 需先按结构域截取")
print(f"取不到序列       : {len(no_seq):,}")
print(f"无代表配体       : {len(no_lig):,}")
print("=" * 58)
if ok:
    L = sorted(x[1] for x in ok)
    print(f"\n待预测长度: min={L[0]} 中位={L[len(L)//2]} max={L[-1]}")
    est = sum(3.5 if x < 600 else (4.5 if x < 900 else 8.5) for x in L)
    print(f"估算: 单卡 {est/60:.1f} 小时，8 卡 {est/60/8:.1f} 小时")
json.dump({"pending_too_long": too_long, "no_seq": no_seq, "no_lig": no_lig},
          open(f"{B}/data/t3/boltz_batch_skipped.json", "w"), indent=1)
print(f"\nYAML 已写入 {OUT}/")
