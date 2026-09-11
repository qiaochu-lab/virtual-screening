"""T2's third explanation: is the zero ranking ability an artefact of label noise?

Where this question comes from
------------
T2 now has three datasets whose conclusions don't agree:
  * FEP benchmark (same chemical series)        ρ ≈ 0.40
  * CASF-2016 (same target, cross-scaffold)     ρ = 0.42   <- newly measured
  * our own T3 (cross-source extraction)        ρ ≈ 0
The earlier explanation was "ranking works within a series, collapses across
series". **CASF overturns that explanation** — each CASF cluster has five
ligands with different scaffolds, and it still reaches 0.42.

So what's left to explain T3's zero? The prime suspect is the **labels
themselves**: T3's pAffinity mixes Ki / Kd / IC50 / EC50 across labs and
assay formats, while FEP's and CASF's numbers come from one consistent
batch of measurements. IC50 depends on substrate concentration and differs
from Ki by a system-dependent constant — mixing them together for ranking
could let noise swamp the signal outright.

How to test it (no GPU needed; scores are all on disk, this just recomputes
over a different set of indices)
------------
Split into three tiers by label cleanliness, compute per-target Spearman,
then average over targets:
  (1) all actives                                  -- current state
  (2) keep only the target's dominant assay type    -- removes type mixing
  (3) keep only the target's largest **single assay_id** -- one experiment,
      one lab, the cleanest possible
Tier (3) is the closest to FEP/CASF conditions. std_type and assay_id are
only present on ChEMBL-sourced records, so tiers (2) and (3) are computed
only on ChEMBL-sourced actives.

How to read it
------------
* ρ rising monotonically with cleanliness -> "the model can't rank" should be
  rewritten as "T3's label noise was masking the ranking ability"
* ρ stays at zero throughout -> the existing conclusion is even more solid,
  and this last data-side explanation is also ruled out
* tier (3)'s n shrinks and Spearman's variance grows -- so both the target
  count and the median ligand count are reported for every tier
"""
import argparse
import json
import os
import pickle
import sys
from collections import Counter, defaultdict

import lmdb
import numpy as np
from rdkit import Chem, RDLogger
from scipy import stats

RDLogger.DisableLog("rdApp.*")
B = "/data/work/vs-benchmark"
MIN_LIG = 5          # targets with fewer than 5 ligands can't produce a meaningful Spearman


def chembl_index():
    """(uniprot, inchikey) -> [(std_type, assay_id)]; only ChEMBL records carry these fields."""
    idx = defaultdict(list)
    p = f"{B}/data/t3/chembl37_2025plus.jsonl"
    for line in open(p):
        d = json.loads(line)
        m = Chem.MolFromSmiles(d["smiles"])
        if m is None:
            continue
        try:
            ik = Chem.MolToInchiKey(m)
        except Exception:
            continue
        idx[(d["uniprot"], ik)].append((d.get("std_type"), d.get("assay_id")))
    print(f"ChEMBL 记录索引: {len(idx):,} 个 (靶点,分子) 组合")
    return idx


def mol_order(up, L, n_pred, eval_rec):
    """Reconstruct the molecule order the model saw -> [(inchikey, paff or None)];
    returns None if it can't be matched.

    Two layouts: the UniMol family reads lmdb (molecules with no conformer
    are skipped); the rest iterate the jsonl directly. Returns None rather
    than guessing when it doesn't match — a wrong guess would mismatch
    ligands and affinities, which is worse than not doing it at all.
    """
    acts = eval_rec["actives"]
    jsonl = [(m["inchikey"], m["paff"]) for m in acts] + \
            [(m["inchikey"], None) for m in eval_rec["decoys"]]
    if len(jsonl) == n_pred:
        return jsonl
    p = f"{B}/data/T3_6A/{L}/{up}/{up}_lig.lmdb"
    if not os.path.exists(p):
        return None
    # Must read in cursor order: keys are strings, so the model's own iteration
    # follows lexicographic order (0, 1, 10, 100, ...), not numeric order.
    # Reading by numeric index would shift everything out of alignment.
    e = lmdb.open(p, subdir=False, readonly=True, lock=False)
    smis = []
    with e.begin() as t:
        for _k, v in t.cursor():
            smis.append(pickle.loads(v)["smi"])
    e.close()
    if len(smis) != n_pred:
        return None
    by_smi = {m["smiles"]: (m["inchikey"], m["paff"]) for m in acts}
    return [by_smi.get(s, (None, None)) for s in smis]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--layers", nargs="+", default=["L1", "L2", "L3", "L4"])
    args = ap.parse_args()

    CH = chembl_index()
    EV = {}
    for L in args.layers:
        p = f"{B}/data/t3/eval/{L}.jsonl"
        EV[L] = {json.loads(x)["uniprot"]: json.loads(x) for x in open(p)} \
            if os.path.exists(p) else {}

    print("\n%-26s %-4s %26s %26s %26s" %
          ("模型", "层", "① 全部 active", "② 单一测定类型", "③ 单个 assay"))
    print("%-26s %-4s %8s %8s %8s %8s %8s %8s %8s %8s %8s" %
          ("", "", "靶点", "配体中位", "ρ", "靶点", "配体中位", "ρ", "靶点", "配体中位", "ρ"))
    print("-" * 112)

    for m in args.models:
        for L in args.layers:
            d = f"{B}/results/t3_raw/{m}/T3/{L}"
            if not os.path.isdir(d):
                d = f"{B}/results/t3/{m}/{L}"
            if not os.path.isdir(d):
                continue
            res = {k: {"rho": [], "n": []} for k in "123"}
            for up in sorted(os.listdir(d)):
                try:
                    p = np.load(f"{d}/{up}/saved_preds.npy").reshape(-1)
                    y = np.load(f"{d}/{up}/saved_labels.npy")
                except Exception:
                    continue
                rec = EV.get(L, {}).get(up)
                if rec is None or len(p) != len(y):
                    continue
                order = mol_order(up, L, len(p), rec)
                if order is None:
                    continue

                # All actives with a measured affinity
                items = [(i, ik, float(a)) for i, (ik, a) in enumerate(order)
                         if a is not None and ik]
                if len(items) < MIN_LIG:
                    continue

                def rho(sub):
                    if len(sub) < MIN_LIG:
                        return None
                    sc = np.array([p[i] for i, _, _ in sub])
                    aff = np.array([a for _, _, a in sub])
                    if np.std(sc) == 0 or np.std(aff) == 0:
                        return None
                    r = stats.spearmanr(sc, aff).statistic
                    return None if np.isnan(r) else r

                r1 = rho(items)
                if r1 is not None:
                    res["1"]["rho"].append(r1); res["1"]["n"].append(len(items))

                # (2) This target's dominant assay type
                types = Counter()
                for _, ik, _ in items:
                    for t, _a in CH.get((up, ik), []):
                        if t:
                            types[t] += 1
                if types:
                    top_t = types.most_common(1)[0][0]
                    sub = [x for x in items
                           if any(t == top_t for t, _ in CH.get((up, x[1]), []))]
                    r2 = rho(sub)
                    if r2 is not None:
                        res["2"]["rho"].append(r2); res["2"]["n"].append(len(sub))

                # (3) This target's largest single assay
                assays = Counter()
                for _, ik, _ in items:
                    for _t, a in CH.get((up, ik), []):
                        if a:
                            assays[a] += 1
                if assays:
                    top_a = assays.most_common(1)[0][0]
                    sub = [x for x in items
                           if any(a == top_a for _, a in CH.get((up, x[1]), []))]
                    r3 = rho(sub)
                    if r3 is not None:
                        res["3"]["rho"].append(r3); res["3"]["n"].append(len(sub))

            if not res["1"]["rho"]:
                continue
            cells = []
            for k in "123":
                v = res[k]
                cells.append("%8d %8.0f %8.3f" % (len(v["rho"]), np.median(v["n"]),
                                                 np.mean(v["rho"])) if v["rho"]
                             else "%8s %8s %8s" % ("-", "-", "-"))
            print("%-26s %-4s %s" % (m, L, " ".join(cells)))

    print("-" * 112)
    print("\n判读：ρ 若随 ①→②→③ 单调上升，说明 T3 的零主要是标签噪声；"
          "若三档都接近零，则最后一个数据端解释也被排除。")
    print("注意 ③ 的配体数明显更少，Spearman 方差更大——看趋势，别看单个格子。")


if __name__ == "__main__":
    main()
