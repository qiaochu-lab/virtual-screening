"""How many actives coarse retrieval can recover -- the ceiling for cascade rerank.

Reranking can only rerank "molecules that already made the top-N". So
recall@N is the upper bound the entire cascade pipeline can reach: an active
that coarse retrieval missed cannot be saved by a physics method, no matter
how accurate. The first version of the experiment only reported P@5/P@10 and
never reported this, which meant the ceiling was unknown.
"""
import json, os
import numpy as np
B="/data/work/vs-benchmark"
M="ligunity_protein_ranking"
print("%-4s %7s %9s %9s %9s %9s %10s" % ("层","靶点","库大小中位","active中位","recall@50","recall@200","recall@500"))
print("-"*70)
for L in ["L1","L2","L3","L4"]:
    d=f"{B}/results/t3_raw/{M}/T3/{L}"
    if not os.path.isdir(d): continue
    r50,r200,r500,ns,na=[],[],[],[],[]
    for up in sorted(os.listdir(d)):
        if KEEP is not None and (L, up) not in KEEP:
            continue
        try:
            p=np.load(f"{d}/{up}/saved_preds.npy").reshape(-1); y=np.load(f"{d}/{up}/saved_labels.npy")
        except Exception: continue
        if len(p)!=len(y) or y.sum()<5: continue
        o=np.argsort(-p); ys=y[o]
        tot=y.sum()
        r50.append(ys[:50].sum()/tot); r200.append(ys[:200].sum()/tot); r500.append(ys[:500].sum()/tot)
        ns.append(len(y)); na.append(int(tot))
    if r50:
        print("%-4s %7d %9d %9d %8.1f%% %9.1f%% %10.1f%%" % (L,len(r50),np.median(ns),np.median(na),
              np.mean(r50)*100,np.mean(r200)*100,np.mean(r500)*100))
print("\n提示：recall@50 就是「只重排 top-50」这个方案的上限。")
