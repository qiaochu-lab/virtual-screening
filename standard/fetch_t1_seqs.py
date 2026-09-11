"""Fill in sequences for T1's three benchmarks' targets (149 UniProt IDs,
26 of which T3 doesn't have).

ConPLex consumes only sequences; SPRINT needs a sequence as the basis for
its 3Di computed on the structure. Fetched directly from UniProt's REST API
as fasta and stored in a separate file, leaving T3's sequences.json untouched.
"""
import json
import os
import time
import urllib.error
import urllib.request

B = "/data/work/vs-benchmark"
OUT = f"{B}/data/t1/sequences.json"


def fetch(up, retry=3):
    url = f"https://rest.uniprot.org/uniprotkb/{up}.fasta"
    for i in range(retry):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                txt = r.read().decode()
            lines = [l.strip() for l in txt.splitlines() if l and not l.startswith(">")]
            return "".join(lines)
        except Exception as e:
            if i == retry - 1:
                print(f"  {up} 失败: {e}")
                return None
            time.sleep(2 * (i + 1))


def main():
    have = json.load(open(f"{B}/data/t3/sequences.json"))
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    ups = set()
    for b in ["DUDE", "DEKOIS", "PCBA"]:
        p = f"{B}/data/t1/{b}.jsonl"
        for line in open(p):
            ups.add(json.loads(line)["uniprot"])

    for up in sorted(ups):
        if up in out:
            continue
        if up in have:                       # reuse directly if T3 has already fetched it
            out[up] = have[up]["seq"] if isinstance(have[up], dict) else have[up]
            continue
        s = fetch(up)
        if s:
            out[up] = s
            print(f"  {up} {len(s)} aa")
        time.sleep(0.2)                      # don't hammer UniProt too fast

    json.dump(out, open(OUT, "w"))
    miss = sorted(ups - set(out))
    print(f"\n序列 {len(out)}/{len(ups)} 已就绪 -> {OUT}")
    if miss:
        print(f"缺 {len(miss)}: {miss}")


if __name__ == "__main__":
    main()
