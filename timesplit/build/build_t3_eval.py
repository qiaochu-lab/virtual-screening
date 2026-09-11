"""Build the T3 virtual-screening evaluation set (cross-target decoys).

Convention (finalized with the user 2026-08-15)
--------------------------------------------------
active   measured ligands with pAff >= 6 (1 uM) for this target, deduped by InChIKey
decoy    real molecules drawn from T3's global molecule pool that act on dissimilar targets
ratio    1:50 (same order of magnitude as DUD-E, so EF@1% stays meaningful)

Why cross-target decoys instead of property-matched decoys
--------------------------------------------------------------
DUD-E-style property-matched decoys are exactly the bias source this
project set out to criticize (a model can win on physicochemical
properties rather than binding mode). Cross-target decoys are all real
drug-like molecules occupying the same chemical space as the actives, so
they don't introduce a property gap; this also avoids the opposite bias
seen with a large random library (where the random molecules' property
distribution is so far from the actives that a model can separate them
by molecular weight alone).

Triple exclusion (to keep decoys as genuinely non-binding as possible)
--------------------------------------------------------------------------
1. The target's own actives (by InChIKey)
2. Molecules that act on a target in the **same mmseqs 40% cluster** --
   homologous targets often share ligands
3. Molecules with the **same scaffold** (Bemis-Murcko) as any of this
   target's actives

Criteria 2 and 3 leave some targets short of the full 50x ratio; when
that happens, whatever count can actually be assembled is used, and the
real ratio is recorded in the output -- EF depends on library size, so
this number must be carried explicitly.
"""
import json
import os
import random
from collections import defaultdict

B = "/data/work/vs-benchmark"
OUT = f"{B}/data/t3/eval"
PAFF_CUT = 6.0
RATIO = 50
MIN_ACTIVES = 10          # targets with fewer actives than this have too much EF variance and are excluded from the main table
SEED = 0


def load_clusters():
    """mmseqs's cluster.tsv: column 1 is the representative sequence, column 2 is the member."""
    c = {}
    with open(f"{B}/data/t3/cluster/t3_40_cluster.tsv") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                c[p[1]] = p[0]
    return c


def main():
    rng = random.Random(SEED)
    clust = load_clusters()
    os.makedirs(OUT, exist_ok=True)

    # ---------- Read every layer ----------
    rows_by_layer = {}
    for L in ["L1", "L2", "L3", "L4"]:
        rows_by_layer[L] = [json.loads(l) for l in open(f"{B}/data/t3/layers/{L}.jsonl")]

    # ---------- Global molecule pool: inchikey -> (smiles, set of clusters it acts on) ----------
    pool_smi, pool_clust, pool_scaf = {}, defaultdict(set), {}
    for L, rows in rows_by_layer.items():
        for r in rows:
            ik = r["inchikey"]
            pool_smi.setdefault(ik, r["smiles"])
            pool_scaf.setdefault(ik, r.get("scaffold") or "")
            cl = clust.get(r["uniprot"])
            if cl:
                pool_clust[ik].add(cl)
    all_ik = sorted(pool_smi)
    print(f"全局分子池: {len(all_ik):,}", flush=True)

    summary = {}
    for L, rows in rows_by_layer.items():
        by_t = defaultdict(list)
        for r in rows:
            try:
                if float(r["paff"]) >= PAFF_CUT:
                    by_t[r["uniprot"]].append(r)
            except (TypeError, ValueError):
                pass

        out_path = f"{OUT}/{L}.jsonl"
        n_t, n_small, n_short, ratios = 0, 0, 0, []
        with open(out_path, "w") as fo:
            for up, acts in sorted(by_t.items()):
                # dedup actives
                uniq = {}
                for r in acts:
                    uniq.setdefault(r["inchikey"], r)
                acts = list(uniq.values())
                if len(acts) < MIN_ACTIVES:
                    n_small += 1
                    continue

                act_ik = set(uniq)
                act_scaf = {r.get("scaffold") or "" for r in acts} - {""}
                my_cl = clust.get(up)

                want = len(acts) * RATIO
                cands = []
                for ik in all_ik:
                    if ik in act_ik:
                        continue
                    if my_cl and my_cl in pool_clust.get(ik, ()):   # ligand of a same-cluster target
                        continue
                    if pool_scaf.get(ik) in act_scaf:               # scaffold collision
                        continue
                    cands.append(ik)
                rng.shuffle(cands)
                dec = cands[:want]
                if len(dec) < want:
                    n_short += 1

                rec = {
                    "uniprot": up, "layer": L,
                    "n_actives": len(acts), "n_decoys": len(dec),
                    "ratio": round(len(dec) / len(acts), 1),
                    "actives": [{"smiles": r["smiles"], "inchikey": r["inchikey"],
                                 "paff": float(r["paff"])} for r in acts],
                    "decoys": [{"smiles": pool_smi[ik], "inchikey": ik} for ik in dec],
                }
                fo.write(json.dumps(rec) + "\n")
                n_t += 1
                ratios.append(rec["ratio"])

        summary[L] = {"targets": n_t, "dropped_few_actives": n_small,
                      "short_of_ratio": n_short,
                      "median_ratio": sorted(ratios)[len(ratios) // 2] if ratios else 0}
        print(f"{L}: 入选靶点 {n_t:4,}   因 active<{MIN_ACTIVES} 剔除 {n_small:5,}   "
              f"凑不满 1:{RATIO} 的 {n_short:3,}   实际比例中位 1:{summary[L]['median_ratio']:.0f}",
              flush=True)

    json.dump({"paff_cut": PAFF_CUT, "ratio": RATIO, "min_actives": MIN_ACTIVES,
               "decoy_scheme": "cross-target (mmseqs 40% cluster + scaffold exclusion)",
               "seed": SEED, "layers": summary},
              open(f"{OUT}/manifest.json", "w"), indent=1)
    print(f"\n已写入 {OUT}/")


if __name__ == "__main__":
    main()
