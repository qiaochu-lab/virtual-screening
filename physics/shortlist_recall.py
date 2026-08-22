"""粗筛能捞回多少 active —— 串联 rerank 的天花板。

重排只能重排「已经进了 top-N 的分子」。所以 recall@N 就是整个级联流程
能达到的上限：粗筛漏掉的 active，物理方法再准也救不回来。
第一版实验只报了 P@5/P@10，没报这个，等于不知道天花板在哪。
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
