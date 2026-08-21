"""改完之后先验证：SMILES 序列里 active 的位置，是否与 saved_labels 一致。
不一致就别往下跑——上一轮就是没验证，白跑了一批 GPU。"""
import json, os, pickle
import lmdb, numpy as np
B="/data/work/vs-benchmark"
ok_all=True
for L in ["L1","L4"]:
    ev={json.loads(x)["uniprot"]: json.loads(x) for x in open(f"{B}/data/t3/eval/{L}.jsonl")}
    root=f"{B}/results/t3_raw/ligunity_protein_ranking/T3/{L}"
    if not os.path.isdir(root): continue
    tot=hit=n=0
    for up in sorted(os.listdir(root))[:40]:
        y=np.load(f"{root}/{up}/saved_labels.npy")
        rec=ev.get(up); path=f"{B}/data/T3_6A/{L}/{up}/{up}_lig.lmdb"
        if rec is None or not os.path.exists(path): continue
        e=lmdb.open(path,subdir=False,readonly=True,lock=False); smis=[]
        with e.begin() as t:
            for _k,v in t.cursor(): smis.append(pickle.loads(v)["smi"])
        e.close()
        if len(smis)!=len(y): continue
        aset={m["smiles"] for m in rec["actives"]}
        ypos=set(np.where(y==1)[0].tolist())
        pos={i for i,s in enumerate(smis) if s in aset}
        tot+=len(ypos); hit+=len(pos & ypos); n+=1
    r=hit/max(tot,1)
    print(f"{L}: {n} 个靶点，active 位置吻合 {hit}/{tot} = {r:.1%}")
    if r < 0.95: ok_all=False
print("结论:", "顺序对上了，可以往下跑" if ok_all else "还是对不上，别跑")
