"""第二版：加零假设对照。

第一版直接报「L3/L4 有多少个靶点的最近训练口袋距离 <0.17」，结果四层全是 100%
——不可读。两个原因：

1. 对 9,726 个参照口袋取最小值是**极值统计**，比较次数一多，任何查询都能
   找到一个近的
2. PocketVec 描述符是 1–128 的**排名向量**，两个随机排列的余弦距离期望约
   0.247 而不是 1，所以绝对阈值 0.17 在「取最小」的语境下失去区分力

正确做法是配对对照：同一个查询，同时对
  A = 训练集靶点的口袋
  B = 同样数量的、既不在训练集也不在 T3 里的人类口袋
取最小距离。**A 显著小于 B 才说明「离训练集特别近」**，否则只是「离任何一
大堆口袋都不远」。
"""
import collections, json, csv
import numpy as np
from scipy import stats

R = "/data/work/vs-benchmark"
FIG = f"{R}/data/raw/figshare"
rng = np.random.default_rng(0)

z = np.load("/tmp/pocketvec.npz", allow_pickle=False)
names = [str(x) for x in z["names"]]
M = z["mat"].astype(np.float32)
M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
by_up = collections.defaultdict(list)
for i, n in enumerate(names):
    by_up[n.split("|")[1].split("_")[0]].append(i)

t3 = collections.defaultdict(set)
for L in ("L1", "L2", "L3", "L4"):
    for line in open(f"{R}/data/t3/eval/{L}.jsonl"):
        t3[L].add(json.loads(line)["uniprot"])
all_t3 = set().union(*t3.values())

train = {a["uniprot"] for a in json.load(open(f"{FIG}/train_label/train_label_pdbbind_seq.json"))}
train |= {a["uniprot"] for a in json.load(open(f"{FIG}/train_label_blend_seq_full.json"))}

ref_ups = sorted(train & set(by_up))
ref_idx = np.array([i for u in ref_ups for i in by_up[u]])
ref_owner = np.array([u for u in ref_ups for _ in by_up[u]])
A = M[ref_idx]

# 对照池：既不在训练集也不在 T3 的人类靶点
ctrl_ups = sorted(set(by_up) - train - all_t3)
ctrl_all = np.array([i for u in ctrl_ups for i in by_up[u]])
ctrl_idx = rng.choice(ctrl_all, size=min(len(ref_idx), len(ctrl_all)), replace=False)
B = M[ctrl_idx]
print(f"A 训练集口袋 {len(ref_idx)}（{len(ref_ups)} 靶点）")
print(f"B 对照口袋   {len(ctrl_idx)}（从 {len(ctrl_ups)} 个非训练非测试靶点里抽）")

# 随机排列基线：两个无关排名向量的距离期望
rp = np.vstack([rng.permutation(128) for _ in range(2000)]).astype(np.float32)
rp /= np.linalg.norm(rp, axis=1, keepdims=True)
print(f"随机排名向量两两距离中位 {np.median(1 - rp[:1000] @ rp[1000:].T):.3f}")

rows = [["layer", "uniprot", "d_train", "d_control", "delta", "nearest_train"]]
print()
print("%-4s %6s %11s %11s %10s %12s %10s" %
      ("层", "n", "对训练集", "对对照池", "Δ(训练-对照)", "配对 p", "训练更近"))
for L in ("L1", "L2", "L3", "L4"):
    da, db, keep = [], [], []
    for u in sorted(t3[L] & set(by_up)):
        Q = M[by_up[u]]
        d1 = 1.0 - Q @ A.T
        m = (ref_owner == u)
        if m.any():
            d1[:, m] = np.inf
        if not np.isfinite(d1).any():
            continue
        d2 = 1.0 - Q @ B.T
        da.append(float(d1.min())); db.append(float(d2.min())); keep.append(u)
    a, b = np.array(da), np.array(db)
    try:
        p = stats.wilcoxon(a, b).pvalue
    except Exception:
        p = float("nan")
    print("%-4s %6d %11.4f %11.4f %10.4f %12.3g %9.0f%%" %
          (L, len(a), np.median(a), np.median(b), np.median(a - b), p,
           100 * (a < b).mean()))
    for u, x, y in zip(keep, da, db, strict=True):
        rows.append([L, u, f"{x:.4f}", f"{y:.4f}", f"{x-y:.4f}", ""])

with open(f"{R}/results/export/T3_pocket_neighbors.csv", "w", newline="") as f:
    csv.writer(f).writerows(rows)
print(f"\n写入 {R}/results/export/T3_pocket_neighbors.csv")
