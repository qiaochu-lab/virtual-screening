"""Benchmark 的三张图。用 dock 环境（唯一装了 matplotlib 的）。

fig1  T3 各层 AUROC 衰减        —— 主发现：时间外推下所有模型一致衰减
fig2  T1 模型 x 基准 EF1% 热图  —— 训练数据分层比架构更能区分性能
fig3  T6 物理方法重排对比        —— 两种物理方法都没带来收益
"""
import csv, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

B = "/data/yicheng/xqc/vs-benchmark"
OUT = f"{B}/results/figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 160})

def rd(p):
    with open(p) as f:
        return list(csv.DictReader(f))

NICE = {"hypseek_rk": "HypSeek", "ligunity_protein_ranking": "LigUnity-protein",
        "ligunity_pocket_ranking": "LigUnity-pocket", "litenclip": "LiTENCLIP",
        "drugclip": "DrugCLIP", "bindclip_randneg": "BindCLIP-randneg",
        "bindclip_hardneg": "BindCLIP-hardneg", "conglude": "ConGLUDe",
        "conplex": "ConPLex", "sprint": "SPRINT",
        "ligunity_pocket": "LigUnity-pocket", "ligunity_protein": "LigUnity-protein"}
# 训练数据分组：PocketAffDB 一组，DrugCLIP set 一组，其余各自
POCKETAFF = {"hypseek_rk", "hypseek", "ligunity_pocket_ranking", "ligunity_protein_ranking",
             "ligunity_pocket", "ligunity_protein", "litenclip"}
DRUGCLIPSET = {"drugclip", "bindclip_randneg", "bindclip_hardneg"}

# ---------------- fig 1: T3 decay
rows = rd(f"{B}/results/export/T3_main.csv")
by = defaultdict(dict)
for r in rows:
    by[r["model"]][r["layer"]] = float(r["AUROC"])
layers = ["L1", "L2", "L3", "L4"]
fig, ax = plt.subplots(figsize=(5.4, 3.8))
labs = []
for m in sorted(by, key=lambda x: -by[x].get("L1", 0)):
    y = [by[m].get(L) for L in layers]
    if any(v is None for v in y):
        continue
    c = "#2a5d9f" if m in POCKETAFF else ("#c0632a" if m in DRUGCLIPSET else "#7a7f8a")
    ax.plot(range(4), y, "o-", color=c, lw=1.6, ms=4, alpha=.9)
    labs.append([y[3], NICE.get(m, m), c])
# 标签互相顶开，避免重叠：自上而下扫一遍，间距不足就往下推
labs.sort(key=lambda t: -t[0])
GAP = 0.023
for i in range(1, len(labs)):
    if labs[i-1][0] - labs[i][0] < GAP:
        labs[i][0] = labs[i-1][0] - GAP
for ly, txt, c in labs:
    ax.annotate(txt, (3.06, ly), va="center", fontsize=7, color=c, annotation_clip=False)
ax.axhline(0.5, color="#bbb", ls="--", lw=.8)
ax.text(1.5, 0.507, "random", fontsize=6.5, color="#999", ha="center")
ax.set_xticks(range(4))
ax.set_xticklabels(["L1\nseen target\nseen scaffold", "L2\nseen target\nnew scaffold",
                    "L3\nnew target\nseen family", "L4\nnew target\nnew family"], fontsize=6.8)
ax.set_ylabel("AUROC"); ax.set_ylim(0.45, 1.0); ax.set_xlim(-0.15, 4.3)
ax.set_title("T3  Generalisation decays with time-split difficulty", fontsize=9.5, loc="left")
h = [plt.Line2D([], [], color=c, lw=2) for c in ("#2a5d9f", "#c0632a", "#7a7f8a")]
ax.legend(h, ["trained on PocketAffDB", "trained on DrugCLIP set", "other training data"],
          fontsize=6.8, frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_t3_decay.png", bbox_inches="tight")
fig.savefig(f"{OUT}/fig1_t3_decay.pdf", bbox_inches="tight"); plt.close(fig)

# ---------------- fig 2: T1 heatmap
rows = rd(f"{B}/results/export/T1_main.csv")
bench = ["DUDE", "DEKOIS", "PCBA"]
M = defaultdict(dict)
for r in rows:
    if r["benchmark"] in bench:
        M[r["model"]][r["benchmark"]] = float(r["EF1"])
models = sorted(M, key=lambda m: -M[m].get("DUDE", 0))
Z = np.array([[M[m].get(b, np.nan) for b in bench] for m in models])
fig, ax = plt.subplots(figsize=(3.6, 0.34 * len(models) + 1.3))
im = ax.imshow(Z, cmap="YlGnBu", aspect="auto", vmin=0, vmax=np.nanmax(Z))
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        if not np.isnan(Z[i, j]):
            ax.text(j, i, f"{Z[i,j]:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if Z[i, j] > np.nanmax(Z) * .55 else "#222")
ax.set_xticks(range(3)); ax.set_xticklabels(["DUD-E", "DEKOIS", "LIT-PCBA"], fontsize=7.5)
ax.set_yticks(range(len(models)))
ax.set_yticklabels([NICE.get(m, m) for m in models], fontsize=7.5)
for t, m in zip(ax.get_yticklabels(), models):
    t.set_color("#2a5d9f" if m in POCKETAFF else ("#c0632a" if m in DRUGCLIPSET else "#555"))
ax.set_title("T1  EF at 1%", fontsize=9.5, loc="left")
fig.colorbar(im, ax=ax, fraction=.05, pad=.03).ax.tick_params(labelsize=6.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_t1_heatmap.png", bbox_inches="tight")
fig.savefig(f"{OUT}/fig2_t1_heatmap.pdf", bbox_inches="tight"); plt.close(fig)

# ---------------- fig 3: T6 physics
def agg(path, mcol, metric):
    rows = rd(path); d = defaultdict(list)
    for r in rows:
        if r.get("coverage") and float(r["coverage"]) < 0.99:
            continue
        d[r[mcol]].append(float(r[metric]))
    return {k: (np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v)), len(v)) for k, v in d.items()}

bo = agg(f"{B}/results/export/T6_rerank3.csv", "method", "auroc")
dk = agg(f"{B}/results/export/T6_dock.csv", "method", "auroc")
groups = [("Boltz-2 rerank\n(top-50 shortlist)", bo,
           [("retrieval", "retrieval"), ("boltz2_rerank", "physics rerank"), ("rank_fusion", "rank fusion")]),
          ("smina docking\n(top-200 shortlist)", dk,
           [("retrieval", "retrieval"), ("smina_rerank", "physics rerank"), ("rank_fusion", "rank fusion")])]
fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.2), sharey=True)
cols = ["#4a5568", "#c0632a", "#2a5d9f"]
for ax, (title, data, keys) in zip(axes, groups):
    xs, ys, es, ns = [], [], [], []
    for k, lab in keys:
        if k in data:
            m, s, n = data[k]; xs.append(lab); ys.append(m); es.append(s); ns.append(n)
    ax.bar(range(len(xs)), ys, yerr=es, capsize=3, color=cols[:len(xs)], width=.62)
    ax.axhline(ys[0], color="#999", ls="--", lw=.9)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, fontsize=7.5)
    ax.set_title(f"{title}   n={ns[0]}", fontsize=8.5, loc="left")
    for i, (y, e) in enumerate(zip(ys, es)):
        ax.text(i, y + e + .012, f"{y:.3f}", ha="center", fontsize=7)
axes[0].set_ylabel("AUROC within shortlist"); axes[0].set_ylim(0, 1.0)
fig.suptitle("T6  Physics rescoring of a retrieval shortlist gives no benefit",
             fontsize=9.5, x=.01, ha="left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_t6_physics.png", bbox_inches="tight")
fig.savefig(f"{OUT}/fig3_t6_physics.pdf", bbox_inches="tight"); plt.close(fig)

print("写入", OUT)
for f in sorted(os.listdir(OUT)):
    print("  ", f, os.path.getsize(f"{OUT}/{f}") // 1024, "KB")
