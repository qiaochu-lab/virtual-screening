"""ConPLex 训练集对 T3 靶点的覆盖率。

之前 per_model_audit.py 把 ConPLex 归到 "unavailable"，因为官方只给序列不给
accession。现在用 mmseqs 反查：把 ConPLex 的 BindingDB train 序列当库，
T3 靶点序列当查询，双向覆盖 ≥50%、E ≤1e-3，按同一性分档。
DUD-E 那 57 个靶点（对比学习用过的）也一并算进「见过」。
"""
import json, os, subprocess, sys
B = "/data/work/vs-benchmark"
MM = "/data/work/mmseqs/bin/mmseqs"
W = f"{B}/tmp_conplex_t3"
os.makedirs(W, exist_ok=True)

seqs = json.load(open(f"{B}/data/t3/sequences.json"))
print(f"T3 靶点序列 {len(seqs)}")
with open(f"{W}/t3.fasta", "w") as f:
    for up, s in sorted(seqs.items()):
        s = s["seq"] if isinstance(s, dict) else s
        if isinstance(s, str) and s:
            f.write(f">{up}\n{s}\n")

def run(*a):
    subprocess.run([MM, *a], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

run("createdb", f"{W}/t3.fasta", f"{W}/qdb")
run("createdb", f"{B}/conplex_bdb_train.fasta", f"{W}/tdb")
run("search", f"{W}/qdb", f"{W}/tdb", f"{W}/res", f"{W}/tmp", "-s", "7.5",
    "--max-seqs", "2000", "-e", "1e-3")
run("convertalis", f"{W}/qdb", f"{W}/tdb", f"{W}/res", f"{W}/hits.tsv",
    "--format-output", "query,target,fident,qcov,tcov,evalue")

best = {}
for line in open(f"{W}/hits.tsv"):
    q, t, fid, qcov, tcov, ev = line.rstrip("\n").split("\t")
    if float(qcov) < 0.5 or float(tcov) < 0.5:
        continue
    best[q] = max(best.get(q, 0.0), float(fid))

dude_up = set()
for line in open(f"{B}/data/t1/DUDE.jsonl"):
    dude_up.add(json.loads(line)["uniprot"])

rows = ["uniprot,best_identity_conplex_bindingdb,in_dude_57"]
allcsv = set()
for p in (f"{B}/dude_cross_full.csv", f"{B}/dude_within_full.csv"):
    for line in open(p):
        q = line.strip().split(",")
        if len(q) >= 2:
            allcsv.add(q[0].lower())
dude57 = set()
for line in open(f"{B}/data/t1/DUDE.jsonl"):
    r = json.loads(line)
    if r["name"].lower() in allcsv:
        dude57.add(r["uniprot"])

for up in sorted(seqs):
    rows.append("%s,%.4f,%d" % (up, best.get(up, 0.0), int(up in dude57)))
open(f"{B}/results/T3_conplex_train_coverage.csv", "w").write("\n".join(rows) + "\n")

print()
for thr in (0.95, 0.9, 0.7, 0.4):
    n = sum(1 for v in best.values() if v >= thr)
    print(f"  与 ConPLex BindingDB 训练序列 ≥{thr:.0%}：{n}/{len(seqs)} ({n/len(seqs):.1%})")
print(f"  同时也是 ConPLex 用过的 DUD-E 靶点：{len(dude57 & set(seqs))}")
print("\n写出 results/T3_conplex_train_coverage.csv")
