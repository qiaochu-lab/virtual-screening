"""L3 随门槛下移，是「小靶点分数系统性偏高」还是「样本太少的抽样噪声」？
直接查：层内 active 数与各指标的相关性。若小靶点确实偏高，相关应为负。"""
import numpy as np, collections, sys
from scipy import stats
sys.path.insert(0, "/data/work/vs/eval")
from metrics import enrichment_factor, bedroc, pr_auc, roc_auc
B="/data/work/vs/results/raw_release"
M=(("EF1%",lambda s,y:enrichment_factor(s,y,0.01)),("BEDROC",lambda s,y:bedroc(s,y,80.5)),
   ("PR-AUC",pr_auc),("AUROC",roc_auc))
models=["ligunity_protein_ranking","hypseek_rk","ligunity_pocket_ranking","litenclip","drugclip"]
agg=collections.defaultdict(lambda: collections.defaultdict(list))
for m in models:
    z=np.load(f"{B}/T3_{m}.npz")
    for t in {k.rsplit("/",1)[0] for k in z.files}:
        p=t.split("/")
        if len(p)<3: continue
        L=p[1]; s,y=z[f"{t}/preds"],z[f"{t}/labels"]
        a=int(y.sum())
        if a<10 or (y==0).sum()<1: continue
        for nm,fn in M:
            try: v=fn(s,y)
            except Exception: continue
            if v==v: agg[(m,L)][nm].append((a,v))
print("层内「active 数」与指标的 Spearman 相关（负 = 小靶点分数更高）")
print("%-22s %-4s %5s %9s %9s %9s %9s" % ("模型","层","靶点","EF1%","BEDROC","PR-AUC","AUROC"))
print("-"*74)
for m in models:
    for L in ("L1","L2","L3","L4"):
        d=agg.get((m,L))
        if not d: continue
        cells=[]
        n=0
        for nm,_ in M:
            v=d[nm]; n=len(v)
            if len(v)<8: cells.append("    —"); continue
            r=stats.spearmanr([x[0] for x in v],[x[1] for x in v])
            cells.append(f"{r.statistic:+.2f}{'*' if r.pvalue<0.05 else ' '}")
        print("%-22s %-4s %5d %9s %9s %9s %9s" % (m,L,n,*cells))
    print()
print("* = p<0.05。若 L3 的负相关显著而其他层不显著，说明是 L3 特有的系统效应；")
print("若各层都不显著，那 L3 的漂移更可能是 48→20 个靶点的抽样噪声。")
