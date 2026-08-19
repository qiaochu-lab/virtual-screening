"""给 T3 评测集靶点标注蛋白类别（v3：ChEMBL 官方分类树，可用版）。

两版失败教训
------------
v1 手写 UniProt 关键词规则 → L4 有 21% 落进「未分类」，抽查发现里面全是
   明摆着的 GPCR（多巴胺 D2/D4、内皮素受体 A/B、PAR-4），只因规则写的是
   "g-protein coupled receptor" 而漏了 "G-protein-coupled" 这种带连字符的写法。
   手写词表天然会漏，漏多少还无法自查。
v2 改用 ChEMBL，但查错了端点：`target.json` 的 target_components 里
   **不含** protein_classifications，结果 868 个靶点全部返回空。

v3 的正确路径：
   1. 一次性拉下整棵分类树（905 个节点），按 parent_id 重建 id → 完整路径
   2. `target_component.json?accession=<UniProt>` 拿该蛋白的 protein_class_id
   3. 用路径匹配折成与 DUD-E 可比的粒度

用 ChEMBL 分类而不是自己定规则，是因为它是虚筛领域做靶点分类的事实标准，
层级由 ChEMBL 维护，不依赖我拍脑袋列词表。
"""
import argparse
import json
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

B = "/data/work/vs-benchmark"
OUT = f"{B}/data/t3/target_class.json"
ROOT = "https://www.ebi.ac.uk/chembl/api/data"


def get(url):
    for _ in range(4):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.load(r)
        except Exception:
            time.sleep(2)
    return None


def load_tree():
    """拉全部 905 个节点，按 parent_id 重建 id -> ['Enzyme','Kinase',...]。"""
    nodes, offset = {}, 0
    while True:
        d = get(f"{ROOT}/protein_classification.json?limit=1000&offset={offset}")
        if not d:
            break
        for p in d["protein_classifications"]:
            nodes[p["protein_class_id"]] = (p["pref_name"], p["parent_id"])
        nxt = d["page_meta"].get("next")
        if not nxt:
            break
        offset += 1000
    paths = {}
    for cid in nodes:
        chain, cur, guard = [], cid, 0
        while cur is not None and cur in nodes and guard < 20:
            name, parent = nodes[cur]
            chain.append(name)
            cur, guard = parent, guard + 1
        paths[cid] = list(reversed(chain))
    return paths


def fetch_component(acc):
    d = get(f"{ROOT}/target_component.json?accession={urllib.parse.quote(acc)}&limit=5")
    if not d:
        return acc, []
    ids = []
    for tc in (d.get("target_components") or []):
        if tc.get("accession") != acc:
            continue
        for pc in (tc.get("protein_classifications") or []):
            i = pc.get("protein_classification_id")
            if i is not None:
                ids.append(i)
    return acc, ids


def to_dude_class(paths):
    """折成与 DUD-E 可比的粒度；判定顺序从特异到宽泛。"""
    if not paths:
        return None
    t = " ; ".join(" / ".join(p) for p in paths).lower()
    if "g protein-coupled receptor" in t or "gpcr" in t:
        return "GPCR"
    if "nuclear receptor" in t:
        return "核受体"
    if "ion channel" in t:
        return "离子通道"
    if any(k in t for k in ["epigenetic", "bromodomain", "histone", "methyltransferase",
                            "deacetylase", "acetyltransferase", "demethylase"]):
        return "表观"
    if "transporter" in t:
        return "转运体"
    if "kinase" in t:
        return "激酶"
    if "protease" in t or "peptidase" in t:
        return "蛋白酶"
    if "cytochrome p450" in t:
        return "P450"
    if "enzyme" in t:
        return "其他酶"
    if "adhesion" in t:
        return "黏附/PPI"
    if "secreted" in t or "surface antigen" in t or "structural" in t:
        return "其他/未分类"
    return "其他/未分类"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="+", default=["L1", "L2", "L3", "L4"])
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    tree = load_tree()
    print(f"ChEMBL 分类树节点: {len(tree):,}", flush=True)

    by_layer, allt = {}, set()
    for L in args.layers:
        try:
            ups = [json.loads(l)["uniprot"] for l in open(f"{B}/data/t3/eval/{L}.jsonl")]
        except FileNotFoundError:
            continue
        by_layer[L] = ups
        allt |= set(ups)
    allt = sorted(allt)
    print(f"评测集靶点（去重）: {len(allt):,}", flush=True)

    raw = {}
    with ThreadPoolExecutor(args.workers) as ex:
        for i, (acc, ids) in enumerate(ex.map(fetch_component, allt)):
            raw[acc] = [tree.get(x, []) for x in ids]
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(allt)}  有分类的 "
                      f"{sum(1 for v in raw.values() if v)}", flush=True)

    cls = {u: (to_dude_class(raw.get(u)) or "其他/未分类") for u in allt}
    json.dump({"class": cls,
               "chembl_paths": {u: [" / ".join(p) for p in v] for u, v in raw.items()}},
              open(OUT, "w"), indent=1)

    dude = {"激酶": 26, "蛋白酶": 15, "核受体": 11, "GPCR": 5, "离子通道": 2,
            "P450": 2, "其他酶": 36, "其他/未分类": 5, "表观": 0,
            "转运体": 0, "黏附/PPI": 0}
    order = ["激酶", "蛋白酶", "核受体", "GPCR", "离子通道", "P450",
             "表观", "转运体", "黏附/PPI", "其他酶", "其他/未分类"]
    counts = {L: Counter(cls[u] for u in ups) for L, ups in by_layer.items()}

    n_nopath = sum(1 for v in raw.values() if not v)
    print(f"\nChEMBL 查不到分类的: {n_nopath} / {len(allt)}")
    print("=" * 82)
    print("%-12s %12s %s" % ("类别", "DUD-E(102)", " ".join(f"{L:>13s}" for L in by_layer)))
    print("-" * 82)
    for k in order:
        d = dude.get(k, 0)
        cells = [f"{counts[L].get(k,0)} ({counts[L].get(k,0)/max(1,len(by_layer[L]))*100:.0f}%)"
                 for L in by_layer]
        print("%-12s %12s %s" % (k, f"{d} ({d/102*100:.0f}%)",
                                 " ".join(f"{c:>13s}" for c in cells)))
    print("-" * 82)
    print("%-12s %12s %s" % ("合计", "102",
                             " ".join(f"{len(by_layer[L]):>13d}" for L in by_layer)))
    print("=" * 82)
    print("\n各层里样本量 ≥20 的类别（能单独报指标的）:")
    for L, c in counts.items():
        ok = [f"{k}({v})" for k, v in c.most_common() if v >= 20]
        print(f"  {L}: {'  '.join(ok) if ok else '无'}")
    print(f"\n已写入 {OUT}")


if __name__ == "__main__":
    main()
