"""fig4：每靶点 active 数的分布，以及它对 EF 精度的影响。

左：四层的 active 数分布（对数横轴），标出 ≥10/20/30/50 四道门槛
右：EF@1% 的步长随 active 数怎么变，为什么小靶点的 EF 是粗糙的
"""
import csv
import gzip
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

B = "/data/work/vs"
OUT = f"{B}/results/figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 160})

LAYERS = ("L1", "L2", "L3", "L4")
COL = {"L1": "#2a5d9f", "L2": "#4a8ac4", "L3": "#c0632a", "L4": "#8b2f1f"}
THR = (10, 20, 30, 50)

rows = list(csv.DictReader(gzip.open(f"{B}/data/release/targets.csv.gz", "rt")))
by = defaultdict(list)
for r in rows:
    by[r["layer"]].append(int(r["n_actives"]))

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6),
                              gridspec_kw={"width_ratios": [1.45, 1]})

# ---- 左：分布
bins = np.logspace(np.log10(10), np.log10(4000), 34)
for L in LAYERS:
    ax.hist(by[L], bins=bins, histtype="step", lw=1.7, color=COL[L],
            label=f"{L}   n={len(by[L])}, median {int(np.median(by[L]))}")
for t in THR:
    ax.axvline(t, color="#999", ls=":", lw=.9)
    ax.text(t, ax.get_ylim()[1] * .97, f"≥{t}", fontsize=6.5, color="#777",
            ha="center", va="top", rotation=90)
ax.set_xscale("log")
ax.set_xlabel("actives per target (log)")
ax.set_ylabel("targets")
ax.set_title("T3: actives per target span two orders of magnitude", fontsize=9.5, loc="left")
ax.legend(fontsize=7, frameon=False)

# ---- 右：EF 步长
a = np.logspace(np.log10(10), np.log10(3000), 200)
# 池子按 1:50 计，N = 51A；前 1% 有 k = ceil(N/100) 个位置
N = 51 * a
k = np.ceil(N / 100)
step = 1.0 / (k * a / N)
ax2.plot(a, step, color="#14161f", lw=1.8)
for t, lab in ((10, "floor"), (24, "L1 median"), (66, "L2 median"), (665, "L1 max")):
    kk = math.ceil(51 * t / 100)
    s = 1.0 / (kk * t / (51 * t))
    ax2.plot([t], [s], "o", ms=4.5, color="#c0632a")
    ax2.annotate(f"{lab}\nA={t}, step {s:.1f}", (t, s), xytext=(6, 6),
                 textcoords="offset points", fontsize=6.6, color="#c0632a")
ax2.axhspan(8, 39, color="#2a5d9f", alpha=.10, lw=0)
ax2.text(1400, 20, "layer EF1%\nmeans\n8-39", fontsize=6.8, color="#2a5d9f",
         ha="center", va="center")
ax2.set_xscale("log"); ax2.set_yscale("log")
ax2.set_xlabel("actives per target")
ax2.set_ylabel("smallest step in EF@1%")
ax2.set_title("One extra hit moves EF@1% by this much", fontsize=9.5, loc="left")

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/fig4_actives_per_target.{ext}", bbox_inches="tight")
print("写入", f"{OUT}/fig4_actives_per_target.png")
