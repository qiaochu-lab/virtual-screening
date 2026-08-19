"""合并 ChEMBL 37 + BindingDB 的时间切分结果，做 L1–L4 难度分层。

分层依据（PPT slide 12）
------------------------
L1 旧靶点 · 近骨架     训练集见过该靶点，且配体骨架也见过
L2 旧靶点 · 新骨架     训练集见过该靶点，但骨架是新的（scaffold generalization）
L3 新靶点 · 同家族     靶点没见过，但训练集里有同家族蛋白（family transfer）
L4 新靶点 · 新家族     靶点和家族都没见过（最严格 OOD）

去重：canonical InChIKey + UniProt
家族划分：用 figshare 提供的 uniport40.clstr（CD-HIT 40% 序列相似度聚类）
骨架：Bemis–Murcko
"""
import json, os
from collections import Counter, defaultdict

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

B = "/data/work/vs-benchmark"
TD = f"{B}/code/LigUnity/test_datasets"
T3 = f"{B}/data/t3"

# ---------- 1. 训练集基线 ----------
lab = json.load(open(f"{TD}/train_label_blend_seq_full.json"))
train_up = {a["uniprot"] for a in lab if a.get("uniprot")}
train_scaffolds = set()
n_lig = 0
for a in lab:
    for l in a.get("ligands", []):
        smi = l.get("smi") if isinstance(l, dict) else None
        if not smi:
            continue
        n_lig += 1
        try:
            m = Chem.MolFromSmiles(smi)
            if m:
                train_scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(mol=m))
        except Exception:
            pass
print(f"训练集: {len(train_up):,} UniProt, {n_lig:,} 配体 -> {len(train_scaffolds):,} 个骨架", flush=True)

# ---------- 2. 家族聚类（CD-HIT 40%）----------
fam = {}
cid = -1
for line in open(f"{B}/data/raw/figshare/uniport40.clstr"):
    if line.startswith(">Cluster"):
        cid = int(line.split()[1])
    elif "|" in line:
        p = line.split("|")
        if len(p) >= 2:
            fam[p[1]] = cid
train_fams = {fam[u] for u in train_up if u in fam}
print(f"CD-HIT 40% 聚类: {len(set(fam.values())):,} 个家族，训练集覆盖 {len(train_fams):,} 个", flush=True)

# ---------- 3. 合并两源 ----------
recs = []
for src, fn in [("bindingdb", "bindingdb_2025plus.jsonl"), ("chembl37", "chembl37_2025plus.jsonl")]:
    p = f"{T3}/{fn}"
    if not os.path.exists(p):
        continue
    n = 0
    for line in open(p):
        d = json.loads(line)
        d.setdefault("source", src)
        recs.append(d)
        n += 1
    print(f"  {src}: {n:,} 条", flush=True)
print(f"合计 {len(recs):,} 条", flush=True)

# ---------- 4. 去重 + 分层 ----------
seen = set()
layers = Counter()
out = defaultdict(list)
bad_smi = 0
for i, d in enumerate(recs):
    if i and i % 50000 == 0:
        print(f"  ...处理 {i:,}", flush=True)
    smi, up = d.get("smiles"), d.get("uniprot")
    if not smi or not up:
        continue
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            bad_smi += 1
            continue
        ik = Chem.MolToInchiKey(m)
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception:
        bad_smi += 1
        continue
    key = (up, ik)
    if key in seen:
        continue
    seen.add(key)

    if up in train_up:
        layer = "L1" if scaf in train_scaffolds else "L2"
    else:
        f = fam.get(up)
        layer = "L3" if (f is not None and f in train_fams) else "L4"
    layers[layer] += 1
    d["inchikey"] = ik
    d["scaffold"] = scaf
    d["layer"] = layer
    out[layer].append(d)

print()
print("=" * 60)
print(f"去重后总记录: {sum(layers.values()):,}   (无效 SMILES {bad_smi:,} 条已丢弃)")
print("=" * 60)
NAMES = {"L1": "旧靶点·近骨架", "L2": "旧靶点·新骨架",
         "L3": "新靶点·同家族", "L4": "新靶点·新家族(最严 OOD)"}
for L in ["L1", "L2", "L3", "L4"]:
    ups = len({d["uniprot"] for d in out[L]})
    print(f"  {L} {NAMES[L]:26s} {layers[L]:>8,} 条   {ups:>5,} 个靶点")

os.makedirs(f"{T3}/layers", exist_ok=True)
for L in out:
    with open(f"{T3}/layers/{L}.jsonl", "w") as f:
        for d in out[L]:
            f.write(json.dumps(d) + "\n")
print(f"\n已写入 {T3}/layers/L[1-4].jsonl")
