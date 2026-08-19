"""T1 辅助分析：性能 vs「训练集中最近邻蛋白的距离」（PPT slide 11）。

关键口径说明
------------
距离计算时**排除测试靶点自身**，也排除其他测试靶点——因为 LigUnity_VS
的 checkpoint 在训练时已剔除了 DUD-E/DEKOIS/LIT-PCBA 的测试蛋白
（见 HF 仓库说明）。所以有意义的量是「训练集里**剩下的**蛋白中最近的有多近」。

距离 0 = 序列相同，0.7 = 距离表的截断上限（更远的对不在表里）。
bootstrap 在靶点层面重采样。
"""
import json, os, re, sys
import numpy as np
sys.path.insert(0, "/data/work/vs-benchmark/eval")
from metrics import enrichment_factor

B = "/data/work/vs-benchmark"
TD = f"{B}/code/LigUnity/test_datasets"

train_up = {a["uniprot"] for a in json.load(open(f"{TD}/train_label_blend_seq_full.json"))
            if a.get("uniprot")}
test_up = set()
for jf in ["dude.json", "dekois.json", "PCBA.json"]:
    test_up |= {x[0] for x in json.load(open(f"{TD}/{jf}"))}

acc = re.compile(r"^(?:sp|tr)\|([A-Z0-9]+)\|")
md = {}
for line in open(f"{B}/ckpt/ligunity/LigUnity_VS/sequence_distance.txt"):
    p = line.rstrip("\n").split("\t")
    if len(p) < 3:
        continue
    a1, a2 = acc.match(p[0]), acc.match(p[1])
    if not (a1 and a2):
        continue
    u1, u2 = a1.group(1), a2.group(1)
    if u1 == u2:
        continue
    try:
        d = float(p[2])
    except ValueError:
        continue
    # 测试靶点 → 训练集中「非测试」蛋白的距离
    if u1 in test_up and u2 in train_up and u2 not in test_up:
        md[u1] = min(md.get(u1, 9.9), d)
    if u2 in test_up and u1 in train_up and u1 not in test_up:
        md[u2] = min(md.get(u2, 9.9), d)


def boot_ci(vals, n=2000, seed=0):
    v = np.asarray([x for x in vals if not np.isnan(x)], float)
    if len(v) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


BINS = [(0.0, 0.1, "[0.0,0.1) 近乎同源"), (0.1, 0.3, "[0.1,0.3) 高度相似"),
        (0.3, 0.5, "[0.3,0.5) 中度相似"), (0.5, 0.7, "[0.5,0.7) 远缘"),
        (0.7, float("inf"), ">0.7 训练集无相似蛋白")]

MODELS = [("LigUnity 口袋塔", "pocket_ranking"), ("LigUnity 序列塔", "protein_ranking"),
          ("DrugCLIP", "drugclip"), ("BindCLIP randneg", "bindclip_randneg")]

for bench, jf, upper in [("DUDE", "dude.json", True), ("DEKOIS", "dekois.json", True)]:
    ref = {x[2]: x[0] for x in json.load(open(f"{TD}/{jf}"))}
    print(f"\n{'#'*84}\n# {bench}\n{'#'*84}")
    table = {}
    for mname, mdir in MODELS:
        d0 = f"{B}/results/{mdir}/{bench}"
        if not os.path.isdir(d0) or not os.listdir(d0):
            continue
        per_bin = {b[2]: [] for b in BINS}
        for t in sorted(os.listdir(d0)):
            up = ref.get(t.upper() if upper else t)
            if up is None:
                continue
            p = f"{d0}/{t}"
            if not os.path.exists(f"{p}/saved_labels.npy"):
                continue
            lab = np.load(f"{p}/saved_labels.npy")
            if os.path.exists(f"{p}/saved_preds.npy"):
                sc = np.load(f"{p}/saved_preds.npy")
            else:
                m = np.load(f"{p}/saved_mols_embed.npy"); pk = np.load(f"{p}/saved_target_embed.npy")
                sc = (pk @ m.T).max(axis=0)
            ef = enrichment_factor(np.asarray(sc, float).ravel(), np.asarray(lab).ravel(), 0.01)
            d = md.get(up, 9.9)
            for lo, hi, name in BINS:
                if lo <= d < hi:
                    per_bin[name].append(ef)
                    break
        table[mname] = per_bin

    hdr = "%-26s" % "距离区间"
    for m in table:
        hdr += "%22s" % m
    print(hdr)
    print("-" * len(hdr))
    for lo, hi, name in BINS:
        row = "%-26s" % name
        for m in table:
            v = table[m][name]
            if not v:
                row += "%22s" % "-"
            else:
                lo_, hi_ = boot_ci(v)
                row += "%22s" % ("%.1f (n=%d)" % (np.nanmean(v), len(v)))
        print(row)
