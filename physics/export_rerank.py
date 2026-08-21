import json, glob, os
import numpy as np
from scipy import stats
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
B="/data/work/vs-benchmark"
man=json.load(open(f"{B}/data/t3/rerank_manifest.json"))
aff={}
for p in glob.glob(f"{B}/boltz_rerank_out/shard_*/*/predictions/*/affinity_*.json"):
    n=os.path.basename(p)[9:-5]
    try: aff[n]=json.load(open(p))["affinity_pred_value"]
    except Exception: pass
by=dict()
for e in man["entries"]: by.setdefault(e["uniprot"],[]).append(e)
rows=["target,n_shortlist,n_actives,method,p_at_5,p_at_10,mean_active_rank,auroc"]
for up,items in by.items():
    items=[e for e in items if e["name"] in aff]
    lab=np.array([e["label"] for e in items]); 
    if len(items)<10 or lab.sum()<2 or (lab==0).sum()<2: continue
    ret=np.array([e["pred"] for e in items]); bz=-np.array([aff[e["name"]] for e in items])
    r1=stats.rankdata(-ret); r2=stats.rankdata(-bz); fus=-(r1+r2)/2
    for nm,sc in [("retrieval",ret),("boltz2_rerank",bz),("rank_fusion",fus)]:
        o=np.argsort(-sc); lo=lab[o]; ranks=np.where(lo==1)[0]+1
        auc=stats.mannwhitneyu(sc[lab==1],sc[lab==0],alternative="greater").statistic/((lab==1).sum()*(lab==0).sum())
        rows.append(f"{up},{len(items)},{int(lab.sum())},{nm},{lo[:5].mean():.3f},{lo[:10].mean():.3f},{ranks.mean():.2f},{auc:.4f}")
open(f"{B}/results/export/T6_rerank.csv","w").write("\n".join(rows)+"\n")
print(len(rows)-1,"行")
