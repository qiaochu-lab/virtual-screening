"""T1 辅助分析（PPT slide 11）：性能 vs 测试靶点到训练集的序列相似度。

统计口径：bootstrap 在**靶点层面**重采样（分析单位是靶点，不是分子）。
把各靶点分子拼起来再重采样会低估方差，且改变了 EF 的定义。
"""
import json, os, re, sys
import numpy as np
sys.path.insert(0, "/data/work/vs-benchmark/eval")
from metrics import enrichment_factor

B = "/data/work/vs-benchmark"
TD = f"{B}/code/LigUnity/test_datasets"

train_up = {a["uniprot"] for a in json.load(open(f"{TD}/train_label_blend_seq_full.json"))
            if a.get("uniprot")}
print(f"训练集 UniProt: {len(train_up):,}", flush=True)

acc = re.compile(r"^(?:sp|tr)\|([A-Z0-9]+)\|")
min_dist = {}
with open(f"{B}/ckpt/ligunity/LigUnity_VS/sequence_distance.txt") as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) < 3:
            continue
        a1, a2 = acc.match(p[0]), acc.match(p[1])
        if not (a1 and a2):
            continue
        u1, u2 = a1.group(1), a2.group(1)
        try:
            d = float(p[2])
        except ValueError:
            continue
        if u2 in train_up and u1 != u2:
            min_dist[u1] = min(min_dist.get(u1, 9.9), d)
        if u1 in train_up and u2 != u1:
            min_dist[u2] = min(min_dist.get(u2, 9.9), d)
print(f"距离表覆盖蛋白: {len(min_dist):,}", flush=True)


def boot_ci(vals, n=2000, seed=0):
    """靶点层面 bootstrap：重采样靶点，算均值分布。"""
    vals = np.asarray([v for v in vals if not np.isnan(v)], float)
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    m = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


BENCH = [("DUDE", "dude.json", True), ("DEKOIS", "dekois.json", True), ("PCBA", "PCBA.json", False)]
MODELS = [("LigUnity 口袋塔", "pocket_ranking"), ("DrugCLIP", "drugclip"),
          ("BindCLIP randneg", "bindclip_randneg")]

for bench, jf, upper in BENCH:
    ref = {x[2]: x[0] for x in json.load(open(f"{TD}/{jf}"))}
    for mname, mdir in MODELS:
        d0 = f"{B}/results/{mdir}/{bench}"
        if not os.path.isdir(d0) or not os.listdir(d0):
            continue
        rows = []
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
            rows.append({"t": t, "up": up, "in_train": up in train_up,
                         "dist": min_dist.get(up),
                         "ef1": enrichment_factor(np.asarray(sc, float).ravel(),
                                                  np.asarray(lab).ravel(), 0.01)})
        if not rows:
            continue

        print(f"\n{'='*80}\n{bench} — {mname}（n={len(rows)}）\n{'='*80}", flush=True)

        def bucket(r):
            if r["in_train"]:
                return "① 靶点本身在训练集"
            if r["dist"] is None:
                return "④ 训练集无相似蛋白 (>0.7)"
            if r["dist"] <= 0.3:
                return "② 有高度相似蛋白 (≤0.3)"
            return "③ 有中度相似蛋白 (0.3~0.7)"

        groups = {}
        for r in rows:
            groups.setdefault(bucket(r), []).append(r)

        print("%-30s %6s %10s %24s" % ("分组", "靶点数", "EF1% 均值", "95% CI (靶点级 bootstrap)"))
        print("-" * 80)
        for k in sorted(groups):
            efs = [r["ef1"] for r in groups[k]]
            lo, hi = boot_ci(efs)
            print("%-30s %6d %10.2f %11.2f ~ %-11.2f" % (k, len(groups[k]), np.nanmean(efs), lo, hi), flush=True)

        ok = sorted([r for r in rows if not np.isnan(r["ef1"])], key=lambda r: r["ef1"])
        print("\n  最差 5 个靶点：")
        for r in ok[:5]:
            dd = "在训练集中" if r["in_train"] else (
                f"最近距离 {r['dist']:.3f}" if r["dist"] is not None else "训练集无相似蛋白")
            print("    %-14s EF1%%=%6.2f  %-8s %s" % (r["t"], r["ef1"], r["up"], dd))
