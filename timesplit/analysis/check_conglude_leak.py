"""Check whether ConGLUDe's training targets overlap our T3 "novel targets".

Why this needs checking: ConGLUDe was submitted 2026-01 and its training-data
version was never disclosed. T3's cutoff of 2024-12 was set against
DrugCLIP/LigUnity's training corpus. If ConGLUDe's training data is newer, it
may have seen T3's test targets — in which case its T3 performance would not
be measuring generalization.

This check does not depend on a version number: it looks directly at the
intersection between ConGLUDe's training-target UniProts and each T3 layer's
targets. L3/L4 are by definition "novel targets absent from the training
set"; if a large fraction of them turn up in ConGLUDe's training set, that is
direct evidence of contamination.
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

# Also read protein_ids.txt (may include non-PDB targets)
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
