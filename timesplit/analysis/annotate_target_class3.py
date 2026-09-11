"""Annotate protein class for T3 eval-set targets (v3: ChEMBL's official
classification tree, the working version).

Lessons from two failed versions
---------------------------------
v1 used hand-written UniProt keyword rules -> 21% of L4 fell into
   "unclassified", and spot-checking found it was full of obvious GPCRs
   (dopamine D2/D4, endothelin receptor A/B, PAR-4) — only because the rule
   was written as "g-protein coupled receptor" and missed the hyphenated
   spelling "G-protein-coupled". Hand-written word lists inevitably miss
   cases, and there is no way to self-check how many.
v2 switched to ChEMBL but queried the wrong endpoint: `target.json`'s
   target_components does **not** contain protein_classifications, so all
   868 targets came back empty.

v3's correct path:
   1. Pull the entire classification tree once (905 nodes), rebuild
      id -> full path via parent_id
   2. Hit `target_component.json?accession=<UniProt>` for that protein's
      protein_class_id
   3. Fold via path matching into a granularity comparable to DUD-E

ChEMBL's classification is used instead of a hand-rolled rule set because it
is the de facto standard for target classification in virtual screening, and
the hierarchy is maintained by ChEMBL rather than depending on a word list I
made up.
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
    """Pull all 905 nodes, rebuild id -> ['Enzyme','Kinase',...] via parent_id."""
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
    """Fold into a granularity comparable to DUD-E; checked in order from
    specific to broad."""
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
