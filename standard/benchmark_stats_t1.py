"""Descriptive statistics for the classic benchmarks -- the table needed for
Part 1 / Table S1 of the paper.

For each of DUD-E, DEKOIS 2.0 and LIT-PCBA: number of targets, actives,
decoys, active:decoy ratio, and **Bemis-Murcko scaffold count**.

Why scaffold count
--------------
The active count alone hides something: a target's actives could be a few
hundred analogs of one chemical series, or a few dozen unrelated scaffolds.
The former can be solved by "memorizing one series"; the latter cannot.
The scaffold-count / active-count ratio directly quantifies how much
"chemical diversity" a benchmark has.

This is the same yardstick T3 uses to draw the L1/L2 boundary from
Bemis-Murcko scaffolds, so putting them side by side is what lets us say
where T3's difficulty diverges from the classic benchmarks'.

⚠️ Counts the **designed pool** (jsonl), not the molecules actually scored.
On the T1 side the two agree (T1 never goes through the lmdb path), but the
convention is stated here for clarity.
"""
import argparse
import collections
import csv
import json
import os
import statistics as st
from concurrent.futures import ProcessPoolExecutor

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
B = "/data/work/vs-benchmark"
BENCH = [("DUD-E", "DUDE"), ("DEKOIS 2.0", "DEKOIS"), ("LIT-PCBA", "PCBA")]
# Compute T3 with the same yardstick here too -- otherwise there is no
# comparable number for "classic benchmarks vs our benchmark"
T3 = [("T3 L1", "t3/eval/L1"), ("T3 L2", "t3/eval/L2"),
      ("T3 L3", "t3/eval/L3"), ("T3 L4", "t3/eval/L4")]


def scaffold(smi):
    """Bemis-Murcko scaffold SMILES; returns None on parse failure."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=f"{B}/data/t1")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default=f"{B}/results/export/T1_benchmark_stats.csv")
    ap.add_argument("--per-target-out",
                    default=f"{B}/results/export/T1_benchmark_stats_per_target.csv")
    args = ap.parse_args()

    rows = [["benchmark", "n_targets", "n_actives", "n_decoys", "ratio",
             "actives_per_target_median", "actives_per_target_min",
             "actives_per_target_max", "n_active_scaffolds",
             "scaffolds_per_target_median", "scaffold_to_active_ratio_median",
             "singleton_scaffold_frac_median"]]
    per = [["benchmark", "target", "uniprot", "n_actives", "n_decoys",
            "n_scaffolds", "scaffold_to_active", "singleton_frac"]]

    print("传统 benchmark 的构成（Part 1 / Table S1）")
    print("=" * 104)
    for label, key in BENCH + T3:
        p = (f"{args.eval_dir}/{key}.jsonl" if "/" not in key
             else f"{B}/data/{key}.jsonl")
        if not os.path.exists(p):
            print(f"{label}: 缺 {p}"); continue
        recs = [json.loads(x) for x in open(p)]

        # Compute scaffolds in parallel: collect all unique active SMILES
        # first, compute once, then assign back to targets
        uniq = sorted({m["smiles"] for r in recs for m in r["actives"]})
        with ProcessPoolExecutor(args.workers) as ex:
            sc = list(ex.map(scaffold, uniq, chunksize=200))
        S = dict(zip(uniq, sc, strict=True))

        na = nd = 0
        per_t_act, per_t_sc, ratios, singles = [], [], [], []
        allsc = set()
        for r in recs:
            acts = [m["smiles"] for m in r["actives"]]
            na += len(acts); nd += r.get("n_decoys") or len(r.get("decoys", []))
            ss = [S[a] for a in acts if S.get(a)]
            c = collections.Counter(ss)
            allsc |= set(ss)
            per_t_act.append(len(acts)); per_t_sc.append(len(c))
            ratios.append(len(c) / len(acts) if acts else float("nan"))
            singles.append(sum(1 for v in c.values() if v == 1) / len(c) if c else float("nan"))
            per.append([label, r.get("name", ""), r.get("uniprot", ""), len(acts),
                        r.get("n_decoys") or len(r.get("decoys", [])), len(c),
                        f"{len(c)/len(acts):.4f}" if acts else "",
                        f"{sum(1 for v in c.values() if v==1)/len(c):.4f}" if c else ""])

        print(f"\n{label}")
        print(f"  靶点 {len(recs)}   活性 {na:,}   诱饵 {nd:,}   "
              f"活性:诱饵 = 1:{nd/na:.0f}")
        print(f"  每靶点活性：中位 {st.median(per_t_act):.0f}  "
              f"范围 {min(per_t_act)}–{max(per_t_act)}")
        print(f"  活性骨架（Bemis-Murcko）：全库唯一 {len(allsc):,}   "
              f"每靶点中位 {st.median(per_t_sc):.0f}")
        print(f"  骨架/活性 中位 {st.median(ratios):.3f}   "
              f"（1.0 = 每个活性一个骨架，越小越像单一化学系列）")
        print(f"  只出现一次的骨架占比 中位 {st.median(singles):.1%}")
        rows.append([label, len(recs), na, nd, f"1:{nd/na:.1f}",
                     f"{st.median(per_t_act):.0f}", min(per_t_act), max(per_t_act),
                     len(allsc), f"{st.median(per_t_sc):.0f}",
                     f"{st.median(ratios):.4f}", f"{st.median(singles):.4f}"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    with open(args.per_target_out, "w", newline="") as f:
        csv.writer(f).writerows(per)
    print("\n" + "=" * 104)
    print("骨架/活性 比值的读法：越接近 1，该基准的活性化学多样性越高；")
    print("越小说明活性集中在少数骨架上，「记住一个系列」就能拿到高富集。")
    print(f"\n写入 {args.out}")
    print(f"写入 {args.per_target_out}")


if __name__ == "__main__":
    main()
