"""检验 ConGLUDe 的训练靶点是否与我们 T3 的「新靶点」重叠。

为什么要查：ConGLUDe 2026-01 投稿，训练数据版本未公开。T3 的切分点
2024-12 是按 DrugCLIP/LigUnity 的训练库定的。若 ConGLUDe 训练数据更新，
它可能见过 T3 的测试靶点——那它的 T3 成绩就不是泛化能力。

这个检验不依赖版本号：直接看它的训练靶点 UniProt 与 T3 各层靶点的交集。
L3/L4 按定义是「训练集里没有的新靶点」，若大量出现在 ConGLUDe 训练集里，
就是污染的直接证据。
"""
import json, os, glob
B = "/data/work/vs-benchmark"
D = f"{B}/tmp/conglude_train/LB_train_val/info"

train_up = set()
n_files = 0
for p in glob.glob(f"{D}/info_dicts/*.json"):
    try:
        d = json.load(open(p))
    except Exception:
        continue
    n_files += 1
    for u in (d.get("uniprot_ids") or []):
        if u: train_up.add(u)
    t = d.get("target_name")
    if t and len(t) <= 10: train_up.add(t)

print(f"ConGLUDe 训练条目 {n_files:,} 个 → 唯一 UniProt {len(train_up):,}\n")

# 也读 protein_ids.txt（可能含非 PDB 靶点）
for f in ["train_protein_ids.txt", "vs_train_protein_ids.txt"]:
    p = f"{D}/{f}"
    if os.path.exists(p):
        ids = [x.strip() for x in open(p) if x.strip()]
        print(f"  {f}: {len(ids):,} 条")

print("\n" + "=" * 70)
print("与 T3 各层靶点的重叠")
print("=" * 70)
print("%-6s %10s %14s %12s" % ("层", "T3 靶点", "在训练集里", "占比"))
print("-" * 46)
for L in ["L1", "L2", "L3", "L4"]:
    p = f"{B}/data/t3/eval/{L}.jsonl"
    if not os.path.exists(p): continue
    ups = {json.loads(l)["uniprot"] for l in open(p)}
    ov = ups & train_up
    print("%-6s %10d %14d %11.1f%%" % (L, len(ups), len(ov), len(ov)/len(ups)*100))

print("\n" + "=" * 70)
print("怎么读")
print("=" * 70)
print("· L1/L2 是「已知靶点」，本来就该在各家训练集里出现，重叠高属正常")
print("· L3/L4 是「新靶点」——按 T3 定义不在 DrugCLIP/LigUnity 训练集里。")
print("  若它们大量出现在 ConGLUDe 训练集，说明 ConGLUDe 的数据更新，")
print("  T3 对它不是干净留出集，其结果需单独标注")
