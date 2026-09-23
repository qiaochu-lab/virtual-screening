#!/usr/bin/env python3
"""把「哪些 T3 靶点被哪一组训练集见过」固化成一张表。

为什么要这张表
--------------
L1–L4 分层原本只用一份训练集判定（`train_label_blend_seq_full.json`，2,196 个
靶点），但那只是 B 家族训练集的一半，更不涵盖 A/C/D 三组。于是 L4 里有靶点
对某些模型其实是见过的。这张表把四组训练集逐靶点摊开，供分层与审计共用。

四组
----
A  结构半      DrugCLIP / BindCLIP ×2            train_no_test_af/train.lmdb
B  亲和力半    LigUnity ×2 / HypSeek / LiTENCLIP / DrugJEPA
               train_label_blend_seq_full.json ∪ train_label_pdbbind_seq.json
C  ConPLex     BindingDB 反查（同一性 ≥0.95 或在 DUD-E 57 里）
D  SPRINT      merged_pos_uniq_train_rand.tsv

⚠️ ConGLUDe 的训练靶点清单至今没拿到（Zenodo 504），**不在这张表里**，
所以「对所有模型都没见过」这句话严格说不含 ConGLUDe。
"""
import csv
import importlib.util
import sys

B = "/data/yicheng/xqc/vs-benchmark"
OUT = f"{B}/results/export/T3_target_exposure.csv"
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("pml", f"{B}/per_model_layers.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

GROUPS = [("A", "DrugCLIP / BindCLIP×2", m.load_A()),
          ("B", "LigUnity×2 / HypSeek / LiTENCLIP / DrugJEPA", m.load_B()),
          ("C", "ConPLex", m.load_C()),
          ("D", "SPRINT", m.load_D())]
Bblend = m.load_B_blend_only()

print("各组训练靶点数")
for g, who, s in GROUPS:
    print(f"  {g}  {who:<44} {len(s):>6}")
print(f"  （B 中原分层只用了 blend 一半：{len(Bblend)}）")

rows = list(csv.DictReader(open(f"{B}/results/export/T3_vsds_matched.csv")))
rel = {r["uniprot"] for r in csv.DictReader(
    open(f"{B}/results/export/T3_target_mirroring_union.csv"))
    if r.get("identity") and float(r["identity"]) >= 0.40}

out = [["uniprot", "layer_original", "layer_corrected", "layer_strict",
        "seen_A", "seen_B", "seen_B_blend_only", "seen_C", "seen_D",
        "seen_any"]]
tally = {}
for r in rows:
    u, lo = r["uniprot"], r["layer"]
    lc = "L3" if (lo == "L4" and u in rel) else lo
    flags = {g: int(u in s) for g, _, s in GROUPS}
    any_ = int(any(flags.values()))
    # strict：只在 L3/L4 上收紧——被任何一组见过就降为 L2（「其实见过这个靶点」）
    ls = lc
    if lc in ("L3", "L4") and any_:
        ls = "L2*"          # 星号表示「原判 L3/L4，但按并集其实见过」
    out.append([u, lo, lc, ls, flags["A"], flags["B"],
                int(u in Bblend), flags["C"], flags["D"], any_])
    tally[ls] = tally.get(ls, 0) + 1

with open(OUT, "w", newline="") as f:
    csv.writer(f).writerows(out)
print(f"\n写出 {OUT}（{len(out)-1} 条）")
print("\nstrict 分层分布")
for k in ("L1", "L2", "L2*", "L3", "L4"):
    if k in tally:
        print(f"  {k:<4} {tally[k]:>4}")
