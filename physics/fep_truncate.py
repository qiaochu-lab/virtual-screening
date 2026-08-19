"""对超过 Boltz-2 长度上限的 FEP 体系做结构域截取。

只有 cmet(1390aa) 和 tyk2(1187aa) 超过 1170 上限，共 40 个配体。
两者都是激酶，真正的药物靶点是激酶结构域——用 UniProt 的
Protein kinase 结构域注释定位，比按长度乱截可靠。
（这套思路与 T3 的 truncate_domains2.py 一致：按结合位点/结构域定位，
不按构建体长度。）
"""
import json, re, time, urllib.parse, urllib.request
B = "/data/work/vs-benchmark"
PAD, LIMIT = 30, 1170

need = {"P08581": "cmet", "P29597": "tyk2"}   # c-Met, TYK2

q = " OR ".join(f"accession:{a}" for a in need)
url = ("https://rest.uniprot.org/uniprotkb/stream?format=tsv"
       "&fields=accession,ft_domain,ft_binding,ft_act_site&query=" + urllib.parse.quote(q))
txt = ""
for _ in range(3):
    try:
        with urllib.request.urlopen(url, timeout=90) as r: txt = r.read().decode(); break
    except Exception: time.sleep(3)

labels = json.load(open(f"{B}/code/LigUnity/test_datasets/FEP/fep_labels.json"))
seqs = {e["uniprot"]: e["sequence"] for e in labels}

out = {}
for ln in txt.split("\n")[1:]:
    p = ln.split("\t")
    if len(p) < 2 or p[0] not in need: continue
    acc = p[0]
    doms = [(int(a), int(b), True) for a, b in re.findall(r"DOMAIN\s+(\d+)\.\.(\d+)", p[1])]
    sites = [int(x) for x in re.findall(r"(?:BINDING|ACT_SITE)\s+(\d+)", ";".join(p[2:]))]
    # 找 Protein kinase 结构域（名字里含 kinase 的那条）
    kin = [(int(a), int(b)) for a, b in
           re.findall(r"DOMAIN\s+(\d+)\.\.(\d+);\s*/note=\"Protein kinase", p[1])]
    cand = kin or doms
    if not cand:
        print(f"  {acc}: 无结构域注释"); continue
    # 选覆盖最多结合位点的；没位点就取最长
    def score(d): return (sum(1 for s in sites if d[0] <= s <= d[1]), d[1]-d[0])
    beg, end = max(((c[0], c[1]) for c in cand), key=score)
    L = len(seqs[acc])
    beg, end = max(1, beg-PAD), min(L, end+PAD)
    if end-beg+1 > LIMIT:
        mid = (beg+end)//2; beg = max(1, mid-LIMIT//2); end = min(L, beg+LIMIT-1)
    ncov = sum(1 for s in sites if beg <= s <= end)
    out[acc] = {"pocket": need[acc], "beg": beg, "end": end,
                "len": end-beg+1, "full": L,
                "sites_total": len(sites), "sites_covered": ncov,
                "seq": seqs[acc][beg-1:end],
                "from_kinase_domain": bool(kin)}
    print(f"  {acc} ({need[acc]}): {L} → {end-beg+1} aa  [{beg}-{end}]  "
          f"结合位点覆盖 {ncov}/{len(sites)}  {'激酶域' if kin else '最长域'}")

json.dump(out, open(f"{B}/data/t3/fep_truncation.json", "w"), indent=1)
print(f"\n已写入 {B}/data/t3/fep_truncation.json")
