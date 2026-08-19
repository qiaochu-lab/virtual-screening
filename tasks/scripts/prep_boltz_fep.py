"""为 16 个 FEP 体系的每个配体准备 Boltz-2 输入。

为什么要逐配体
--------------
T6 现有的 929 个 Boltz-2 预测**每个靶点只有一个代表配体**，只能做
「跨靶点的绝对亲和力相关」（已测出 Spearman +0.404）。而 FEP 基准测的是
**同一靶点内按结合强弱排序**——必须每个配体各算一次。

为什么用 FEP 这套体系
--------------------
现在有三个数字，但口径不同、严格说不可比：
  · 检索模型在 FEP 同系列内   ρ ≈ 0.4
  · Boltz-2 跨靶点             ρ = +0.404
  · FEP+ 物理方法（文献）      r ≈ 0.6–0.8
在同一批 16 个体系、461 个配体上跑 Boltz-2，三类方法才能真正对齐比较。

⚠️ 已知限制
-----------
Boltz-2 亲和力模块不支持 >128 原子的配体；FEP 集都是类药小分子，应该没问题，
但仍会检查并记录跳过的。
"""
import json
import os

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

B = "/data/yicheng/xqc/vs-benchmark"
FEP = f"{B}/code/LigUnity/test_datasets/FEP"
OUT = f"{B}/boltz_fep"
SHARDS = 3          # 用空闲的 GPU 4/6/7
MAX_ATOMS = 128


def main():
    labels = json.load(open(f"{FEP}/fep_labels.json"))
    # cmet(1390aa) 和 tyk2(1187aa) 超过 Boltz-2 的 1170 上限，
    # 用按激酶结构域截取的序列替换（结合位点 3/3 全覆盖，见 fep_truncate.py）
    trunc = {}
    tp = f"{B}/data/t3/fep_truncation.json"
    if os.path.exists(tp):
        trunc = {k: v["seq"] for k, v in json.load(open(tp)).items()}
    os.makedirs(OUT, exist_ok=True)
    for i in range(SHARDS):
        os.makedirs(f"{OUT}/shard_{i}", exist_ok=True)

    rows, skipped = [], []
    for e in labels:
        pocket = e["pockets"][0]
        seq = trunc.get(e["uniprot"], e["sequence"])
        for j, lig in enumerate(e["ligands"]):
            smi = lig["smi"]
            m = Chem.MolFromSmiles(smi)
            if m is None:
                skipped.append((pocket, j, "SMILES 解析失败")); continue
            if m.GetNumAtoms() > MAX_ATOMS:
                skipped.append((pocket, j, f"{m.GetNumAtoms()} 原子超限")); continue
            rows.append({"pocket": pocket, "uniprot": e["uniprot"], "idx": j,
                         "smi": smi, "act": lig["act"], "seq": seq})

    # 按序列长度轮转分片，让各卡负载均衡（Boltz-2 耗时随长度陡增）
    rows.sort(key=lambda r: -len(r["seq"]))
    manifest = []
    for n, r in enumerate(rows):
        name = f"{r['pocket']}__{r['idx']:03d}"
        y = ("version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: %s\n"
             "  - ligand:\n      id: B\n      smiles: '%s'\n"
             "properties:\n  - affinity:\n      binder: B\n" % (r["seq"], r["smi"]))
        open(f"{OUT}/shard_{n % SHARDS}/{name}.yaml", "w").write(y)
        manifest.append({"name": name, "pocket": r["pocket"], "uniprot": r["uniprot"],
                         "idx": r["idx"], "act": r["act"], "smi": r["smi"]})

    json.dump({"entries": manifest,
               "skipped": [{"pocket": p, "idx": i, "why": w} for p, i, w in skipped]},
              open(f"{B}/data/t3/boltz_fep_manifest.json", "w"), indent=1)

    from collections import Counter
    print(f"生成输入 {len(rows):,} 个（{len(labels)} 个体系）")
    print(f"跳过 {len(skipped)}: {Counter(w for _, _, w in skipped).most_common(3)}")
    ls = sorted(len(r["seq"]) for r in rows)
    print(f"序列长度 中位 {ls[len(ls)//2]}  最长 {ls[-1]}")
    for i in range(SHARDS):
        print(f"  shard_{i}: {len(os.listdir(f'{OUT}/shard_{i}')):,}")


if __name__ == "__main__":
    main()
