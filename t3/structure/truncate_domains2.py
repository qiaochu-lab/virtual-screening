"""为 >1170aa 的 T3 靶点按结构域截取（v2：以结合位点为准）。

v1 为什么不行
-------------
v1 的规则是「取所有 PDB 条目里覆盖最广的那段对齐区间」。对多结构域大蛋白，
不同 PDB 条目覆盖的是**不同结构域**，取最广的与「配体结合在哪」毫无关系。
校验打脸得很干净：只有 41.8% 的截取片段完整包含 UniProt 注释的结合/活性位点，
26 个（33%）一个位点都没落进去。典型如 P23468，v1 截了 577-946，
位点却在 1181-1844。

v2 的规则
---------
**结合/活性位点注释是配体结合位置的直接证据**，所以拿它当第一判据，
而不是当事后校验：

  1. 收集全部候选区间 —— 每个 PDB 条目的构建体范围 + 每条 UniProt 结构域注释
  2. 打分排序：覆盖的位点数（主）→ 是否为实验构建体（次）→ 长度（再次）
  3. 没有任何位点注释的蛋白：退回「最长的 PDB 构建体 → 最长的结构域」，
     并标记为低置信，单独统计

位点密集区若没有任何候选区间覆盖，就直接以位点簇为中心开窗，
保证截出的片段一定含结合位点。

窗口两侧留 30 残基余量，最短 150aa（太短的多是锌指之类的小模块，
不是可成药口袋），最长夹到 1170aa。
"""
import json
import re
import time
import urllib.parse
import urllib.request
from collections import Counter

B = "/data/yicheng/xqc/vs-benchmark"
LIMIT, PAD, MIN_LEN = 1170, 30, 150
OUT = f"{B}/data/t3/domain_truncation.json"

GQL = """
query($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    polymer_entities {
      entity_poly { rcsb_entity_polymer_type }
      rcsb_polymer_entity_align {
        reference_database_accession
        reference_database_name
        aligned_regions { entity_beg_seq_id ref_beg_seq_id length }
      }
    }
  }
}
"""


def gql(ids):
    body = json.dumps({"query": GQL, "variables": {"ids": ids}}).encode()
    req = urllib.request.Request("https://data.rcsb.org/graphql", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def uniprot_bulk(accs):
    """一次取全：结构域注释 + 结合/活性/一般位点。"""
    out = {}
    for i in range(0, len(accs), 60):
        chunk = accs[i:i + 60]
        q = " OR ".join(f"accession:{a}" for a in chunk)
        url = ("https://rest.uniprot.org/uniprotkb/stream?format=tsv"
               "&fields=accession,ft_domain,ft_binding,ft_act_site,ft_site&query="
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
            doms = [(int(a), int(b)) for a, b in
                    re.findall(r"DOMAIN\s+(\d+)\.\.(\d+)", p[1])]
            blob = ";".join(p[2:])
            sites = sorted({int(a) for a in
                            re.findall(r"(?:BINDING|ACT_SITE|SITE)\s+(\d+)", blob)})
            out[p[0]] = {"domains": doms, "sites": sites}
        print(f"  UniProt {min(i + 60, len(accs))}/{len(accs)}", flush=True)
    return out


def clamp(beg, end, L):
    """加余量、保证最短长度、夹到上限。"""
    beg, end = max(1, beg - PAD), min(L, end + PAD)
    if end - beg + 1 < MIN_LEN:                     # 太短则以中心扩展
        mid = (beg + end) // 2
        beg = max(1, mid - MIN_LEN // 2)
        end = min(L, beg + MIN_LEN - 1)
        beg = max(1, end - MIN_LEN + 1)
    if end - beg + 1 > LIMIT:                       # 太长则以中心收缩
        mid = (beg + end) // 2
        beg = max(1, mid - LIMIT // 2)
        end = min(L, beg + LIMIT - 1)
        beg = max(1, end - LIMIT + 1)
    return beg, end


def main():
    targets = json.load(open(f"{B}/data/t3/missing_breakdown.json"))["序列>1170aa"]
    seqs = json.load(open(f"{B}/data/t3/sequences.json"))
    up2pdb = json.load(open(f"{B}/data/t3/pdb_meta.json"))["up2pdb"]
    print(f"待截取靶点: {len(targets)}", flush=True)

    # ---------- 候选区间来源 1：每个 PDB 条目的构建体范围 ----------
    pdb_ids = sorted({p for u in targets for p in (up2pdb.get(u) or [])[:12]})
    print(f"涉及 PDB 条目: {len(pdb_ids)}", flush=True)
    cand = {u: [] for u in targets}          # uniprot -> [(beg, end, 来源)]
    tset = set(targets)
    for i in range(0, len(pdb_ids), 50):
        d = None
        for _ in range(3):
            try:
                d = gql(pdb_ids[i:i + 50])
                break
            except Exception:
                time.sleep(4)
        if not d or "data" not in d:
            continue
        for e in (d["data"].get("entries") or []):
            for pe in (e.get("polymer_entities") or []):
                if (pe.get("entity_poly") or {}).get("rcsb_entity_polymer_type") != "Protein":
                    continue
                for al in (pe.get("rcsb_polymer_entity_align") or []):
                    if al.get("reference_database_name") != "UniProt":
                        continue
                    acc, regs = al.get("reference_database_accession"), al.get("aligned_regions") or []
                    if acc not in tset or not regs:
                        continue
                    beg = min(r["ref_beg_seq_id"] for r in regs)
                    end = max(r["ref_beg_seq_id"] + r["length"] - 1 for r in regs)
                    cand[acc].append((beg, end, "pdb_construct"))
        if (i // 50) % 5 == 0:
            print(f"  RCSB {min(i + 50, len(pdb_ids))}/{len(pdb_ids)}", flush=True)

    # ---------- 候选区间来源 2 + 位点：UniProt ----------
    feats = uniprot_bulk(targets)
    for u in targets:
        for d in (feats.get(u) or {}).get("domains") or []:
            cand[u].append((d[0], d[1], "uniprot_domain"))

    # ---------- 选区间 ----------
    result, unresolved = {}, []
    for u in targets:
        L = seqs[u]["length"]
        sites = (feats.get(u) or {}).get("sites") or []
        cs = cand[u]

        if sites:
            if cs:
                def score(c):
                    n = sum(1 for s in sites if c[0] <= s <= c[1])
                    return (n, 1 if c[2] == "pdb_construct" else 0, c[1] - c[0])
                best = max(cs, key=score)
                n_cov = sum(1 for s in sites if best[0] <= s <= best[1])
            else:
                best, n_cov = None, 0
            if n_cov == 0:
                # 没有任何候选区间盖到位点 —— 直接以位点簇开窗
                lo, hi = min(sites), max(sites)
                beg, end = clamp(lo, hi, L)
                src, n_cov = "site_window", sum(1 for s in sites if beg <= s <= end)
            else:
                beg, end = clamp(best[0], best[1], L)
                src = best[2]
                n_cov = sum(1 for s in sites if beg <= s <= end)
            conf = "high" if n_cov == len(sites) else "partial"
        else:
            if not cs:
                unresolved.append(u)
                continue
            best = max(cs, key=lambda c: (1 if c[2] == "pdb_construct" else 0, c[1] - c[0]))
            beg, end = clamp(best[0], best[1], L)
            src, n_cov, conf = best[2], 0, "no_site_annotation"

        result[u] = {"beg": beg, "end": end, "length": end - beg + 1,
                     "full_length": L, "source": src, "confidence": conf,
                     "n_sites_total": len(sites), "n_sites_covered": n_cov,
                     "seq": seqs[u]["seq"][beg - 1:end]}

    json.dump({"truncation": result, "unresolved": sorted(unresolved)},
              open(OUT, "w"), indent=1)

    print()
    print("=" * 64)
    print(f"成功截取 {len(result)} / {len(targets)}   无法判定 {len(unresolved)}")
    print("\n置信度:")
    for k, v in Counter(r["confidence"] for r in result.values()).most_common():
        print(f"  {k:20s} {v}")
    print("\n依据:")
    for k, v in Counter(r["source"] for r in result.values()).most_common():
        print(f"  {k:20s} {v}")
    have = [r for r in result.values() if r["n_sites_total"]]
    if have:
        full = sum(1 for r in have if r["n_sites_covered"] == r["n_sites_total"])
        none = sum(1 for r in have if r["n_sites_covered"] == 0)
        print(f"\n位点覆盖（有注释的 {len(have)} 个）:")
        print(f"  全覆盖 {full} ({full/len(have)*100:.1f}%)   部分 {len(have)-full-none}   零覆盖 {none}")
    ls = sorted(r["length"] for r in result.values())
    fs = sorted(r["full_length"] for r in result.values())
    print(f"\n截取后长度: 中位 {ls[len(ls)//2]}  最短 {ls[0]}  最长 {ls[-1]}")
    print(f"原始中位 {fs[len(fs)//2]}，平均截掉 {(1 - sum(ls)/sum(fs))*100:.1f}%")
    print(f"\n已写入 {OUT}")


if __name__ == "__main__":
    main()
