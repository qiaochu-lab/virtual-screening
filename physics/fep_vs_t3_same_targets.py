"""Compare ranking ability on the FEP data vs. the T3 data, on the same set of targets.

The question: LigUnity scores Spearman +0.396 on the FEP benchmark but only
+0.018 on T3 -- a twentyfold gap. Three possible explanations:
  (1) Ligand properties differ -- FEP is a congeneric series of analogues,
      T3 is drawn cross-database with large chemical diversity
  (2) Target familiarity -- all 16 FEP targets are classic ones, certainly in
      the training set; T3 includes novel targets
  (3) Data quality -- FEP's affinities come from one consistent assay batch,
      so noise is low

This test isolates (2): look only at **the targets that appear in both FEP
and T3**. If T3's data still fails to rank on the same set of targets, the
gap comes from (1)/(3), not (2).
"""
import json, os
import numpy as np
from scipy import stats

B = "/data/work/vs-benchmark"
# FEP.json is a list of [uniprot, pdb, ...]; the target name (pocket) lives in fep_labels.json
fep_ups = {e["uniprot"]: e["pockets"][0]
           for e in json.load(open(f"{B}/code/LigUnity/test_datasets/FEP/fep_labels.json"))}
print(f"FEP 靶点 {len(fep_ups)} 个\n")

s = json.load(open(f"{B}/results/t3/summary_t2.json"))
truth = {}
for L in ["L1","L2","L3","L4"]:
    p = f"{B}/data/t3/eval/{L}.jsonl"
    if not os.path.exists(p): continue
    for line in open(p):
        d = json.loads(line)
        truth.setdefault(d["uniprot"], {})[L] = [a["paff"] for a in d["actives"]]

print("=" * 78)
print("同一批靶点：FEP 数据上的排序 vs T3 数据上的排序（Spearman）")
print("=" * 78)
for m, femodel in [("ligunity_pocket_ranking","ligunity_pocket_ranking"),
                   ("ligunity_protein_ranking","ligunity_protein_ranking")]:
    print(f"\n【{m}】")
    print("  %-9s %-9s %8s %10s %8s %10s" % ("靶点","pocket","FEP n","FEP ρ","T3 n","T3 ρ"))
    print("  " + "-"*60)
    fep_v, t3_v = [], []
    for up, pk in sorted(fep_ups.items()):
        # FEP side
        d = f"{B}/results/fep/{femodel}/FEP/{pk}"
        try:
            pr = np.load(f"{d}/saved_preds.npy"); yy = np.load(f"{d}/saved_labels.npy")
            fr, fn = stats.spearmanr(pr, yy).statistic, len(yy)
        except Exception:
            fr, fn = None, 0
        # T3 side
        tr, tn = None, 0
        for L in ["L1","L2","L3","L4"]:
            dd = f"{B}/results/t3/{m}/{L}/{up}"
            if not os.path.isdir(dd): continue
            pa = truth.get(up, {}).get(L)
            if not pa or len(pa) < 10: continue
            try:
                p2 = np.load(f"{dd}/saved_preds.npy"); l2 = np.load(f"{dd}/saved_labels.npy")
            except Exception: continue
            act = np.nonzero(l2 == 1)[0]
            if len(act) != len(pa) or np.std(p2[act]) == 0: continue
            tr, tn = stats.spearmanr(p2[act], pa).statistic, len(pa)
            break
        if fr is None or tr is None: continue
        fep_v.append(fr); t3_v.append(tr)
        print("  %-9s %-9s %8d %+10.3f %8d %+10.3f" % (up, pk, fn, fr, tn, tr))
    if fep_v:
        print("  " + "-"*60)
        print("  %-19s %8s %+10.3f %8s %+10.3f" %
              ("均值（同一批靶点）", "", np.mean(fep_v), "", np.mean(t3_v)))
        try:
            p = stats.wilcoxon(fep_v, t3_v).pvalue
            print(f"  配对检验 p = {p:.4f}  (n={len(fep_v)})")
        except ValueError:
            pass

print("\n" + "=" * 78)
print("怎么读")
print("=" * 78)
print("· 若同一批靶点上 FEP 仍远高于 T3 → 差异来自数据性质（同系列类似物 vs 跨库抽取），")
print("  不是靶点熟悉度。那么「排不出强弱」这个结论要限定在跨库数据上。")
print("· 若两者接近 → 之前的差异主要来自靶点熟悉度。")
