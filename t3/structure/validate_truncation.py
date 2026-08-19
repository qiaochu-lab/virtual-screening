"""独立校验结构域截取：截出的片段是否包含该蛋白已注释的结合/活性位点。

这个校验对两种依据都有效，且用的是截取时**没有参与决策**的信息：
  - pdb_construct 依据完全没看注释
  - uniprot_domain 依据虽用了位点数排序，但只在候选结构域之间比较，
    并不保证选中的那个真的含位点（无注释时是按长度选的）
所以「截出片段是否含位点」是一个独立的对错信号。
"""
import json
import re
import time
import urllib.parse
import urllib.request

B = "/data/work/vs-benchmark"
T = json.load(open(f"{B}/data/t3/domain_truncation.json"))["truncation"]
accs = sorted(T)


def fetch_sites(accs):
    out = {}
    for i in range(0, len(accs), 60):
        chunk = accs[i:i + 60]
        q = " OR ".join(f"accession:{a}" for a in chunk)
        url = ("https://rest.uniprot.org/uniprotkb/stream?format=tsv"
               "&fields=accession,ft_binding,ft_act_site,ft_site&query="
               + urllib.parse.quote(q))
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
            if len(p) < 2:
                continue
            blob = ";".join(p[1:])
            out[p[0]] = sorted({int(a) for a in
                                re.findall(r"(?:BINDING|ACT_SITE|SITE)\s+(\d+)", blob)})
    return out


sites = fetch_sites(accs)

n_with, n_hit, n_partial = 0, 0, 0
bad, short = [], []
for u, d in T.items():
    s = sites.get(u) or []
    if d["length"] < 120:
        short.append((u, d["length"], d["source"]))
    if not s:
        continue
    n_with += 1
    inside = [x for x in s if d["beg"] <= x <= d["end"]]
    if len(inside) == len(s):
        n_hit += 1
    elif inside:
        n_partial += 1
    else:
        bad.append((u, d["source"], d["beg"], d["end"], s[:6]))

print("=" * 64)
print(f"截取靶点 {len(T)}，其中有位点注释可校验的: {n_with}")
print(f"  全部位点都落在截出片段内 : {n_hit}  ({n_hit/max(1,n_with)*100:.1f}%)")
print(f"  部分落在片段内           : {n_partial}")
print(f"  ⚠️ 一个都没落进去        : {len(bad)}")
print("=" * 64)
if bad:
    print("\n没命中的（截错了，需回退整条或换结构域）:")
    for u, src, b, e, s in bad[:15]:
        print(f"  {u:10s} 依据={src:16s} 截取 {b}-{e}   位点在 {s}")
print(f"\n截出片段 <120aa 的: {len(short)}（可能是锌指/小模块，不是可成药口袋）")
for u, l, src in sorted(short, key=lambda x: x[1])[:10]:
    print(f"  {u:10s} {l:4d}aa  依据={src}")

json.dump({"bad": [u for u, *_ in bad], "short": [u for u, *_ in short]},
          open(f"{B}/data/t3/truncation_flags.json", "w"), indent=1)
