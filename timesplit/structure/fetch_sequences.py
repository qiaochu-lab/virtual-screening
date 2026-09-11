"""Fetch UniProt sequences for all of T3's targets.

Sequence-based models (ConPLex, SPRINT) don't need a pocket, but do need
the full sequence; previously only the batch of targets routed through
Boltz-2 had sequences in their yaml, not the 972 targets that have a PDB
structure.
"""
import json
import os
import time
import urllib.parse
import urllib.request

B = "/data/work/vs-benchmark"
OUT = f"{B}/data/t3/sequences.json"
BATCH = 100


def main():
    ups = set()
    for L in ["L1", "L2", "L3", "L4"]:
        p = f"{B}/data/t3/layers/{L}.jsonl"
        if not os.path.exists(p):
            continue
        for line in open(p):
            ups.add(json.loads(line)["uniprot"])
    ups = sorted(ups)
    print(f"T3 靶点总数: {len(ups):,}", flush=True)

    seqs = {}
    for i in range(0, len(ups), BATCH):
        chunk = ups[i:i + BATCH]
        q = " OR ".join(f"accession:{a}" for a in chunk)
        url = ("https://rest.uniprot.org/uniprotkb/stream?format=tsv"
               "&fields=accession,sequence,length,protein_name&query=" + urllib.parse.quote(q))
        txt = ""
        for _ in range(3):
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    txt = r.read().decode()
                break
            except Exception:
                time.sleep(3)
        for ln in txt.split("\n")[1:]:
            p = ln.split("\t")
            if len(p) >= 3 and p[1].strip():
                seqs[p[0]] = {"seq": p[1].strip(), "length": int(p[2]),
                              "name": p[3] if len(p) > 3 else ""}
        if (i // BATCH) % 5 == 0:
            print(f"  {min(i + BATCH, len(ups)):,}/{len(ups):,}  已拿到 {len(seqs):,}", flush=True)

    json.dump(seqs, open(OUT, "w"))
    miss = [u for u in ups if u not in seqs]
    lens = sorted(v["length"] for v in seqs.values())
    print(f"\n拿到序列: {len(seqs):,}/{len(ups):,}  缺失 {len(miss)}")
    if miss[:10]:
        print(f"  缺失例: {miss[:10]}")
    print(f"序列长度: 中位 {lens[len(lens)//2]}  最短 {lens[0]}  最长 {lens[-1]}")
    print(f"  >2000 残基的: {sum(1 for x in lens if x > 2000)}  （ESM 类模型需截断或分段）")
    print(f"已写入 {OUT}")


if __name__ == "__main__":
    main()
