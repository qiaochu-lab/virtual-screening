"""Second version: with a null-hypothesis control.

The first version directly reported "how many L3/L4 targets have a nearest
training-pocket distance < 0.17", and all four layers came out at 100% —
unreadable. Two reasons:

1. Taking the minimum over 9,726 reference pockets is **extreme-value
   statistics** — with enough comparisons, any query will find something close.
2. The PocketVec descriptor is a **rank vector** over 1–128, and the expected
   cosine distance between two random permutations is about 0.247, not 1, so
   the absolute threshold of 0.17 loses its discriminative power once you're
   taking a minimum.

The correct approach is a paired control: for the same query, take the
minimum distance to both
  A = pockets of training-set targets
  B = the same number of human pockets that are neither in the training set
      nor in T3
**Only if A is significantly smaller than B does that mean "specifically
close to the training set"** — otherwise it's just "not far from any large
pile of pockets".
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

# Control pool: human targets that are neither in the training set nor in T3
ctrl_ups = sorted(set(by_up) - train - all_t3)
ctrl_all = np.array([i for u in ctrl_ups for i in by_up[u]])
ctrl_idx = rng.choice(ctrl_all, size=min(len(ref_idx), len(ctrl_all)), replace=False)
B = M[ctrl_idx]
print(f"A 训练集口袋 {len(ref_idx)}（{len(ref_ups)} 靶点）")
print(f"B 对照口袋   {len(ctrl_idx)}（从 {len(ctrl_ups)} 个非训练非测试靶点里抽）")

# Random-permutation baseline: expected distance between two unrelated rank vectors
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
