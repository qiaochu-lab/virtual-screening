"""构建 T3 虚筛评测集（跨靶点 decoy）。

口径（用户 2026-08-15 确定）
---------------------------
active   该靶点 pAff ≥ 6（1 µM）的实测配体，按 InChIKey 去重
decoy    从 T3 全局分子池里抽「作用于不相似靶点」的真实分子
比例     1 : 50（与 DUD-E 同量级，保证 EF@1% 有意义）

为什么用跨靶点 decoy 而不是属性匹配
-----------------------------------
DUD-E 式的属性匹配 decoy 正是本项目要批评的偏倚来源（模型可能靠理化性质
而非结合模式取胜）。跨靶点 decoy 全是真实类药分子，与 active 同处一个
化学空间，不引入属性差；同时也避免了随机大库那种反方向的偏倚
（随机分子性质分布与 active 差太远，模型只靠分子量就能分开）。

三重排除（保证 decoy 尽可能真的不结合）
---------------------------------------
1. 该靶点自己的 active（按 InChIKey）
2. 作用于**同一 mmseqs 40% 簇**靶点的分子 —— 同源靶点常共享配体
3. 与该靶点任一 active **骨架相同**（Bemis-Murcko）的分子

第 2、3 条会让部分靶点凑不满 50 倍，此时按实际能凑到的数量给，
并在输出里记录真实比例——EF 依赖库大小，这个数必须显式带着。
"""
import json
import os
import random
from collections import defaultdict

B = "/data/yicheng/xqc/vs-benchmark"
OUT = f"{B}/data/t3/eval"
PAFF_CUT = 6.0
RATIO = 50
MIN_ACTIVES = 10          # 少于这个数的靶点，EF 方差过大，不入正式表
SEED = 0


def load_clusters():
    """mmseqs 的 cluster.tsv：第 1 列是代表序列，第 2 列是成员。"""
    c = {}
    with open(f"{B}/data/t3/cluster/t3_40_cluster.tsv") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                c[p[1]] = p[0]
    return c


def main():
    rng = random.Random(SEED)
    clust = load_clusters()
    os.makedirs(OUT, exist_ok=True)

    # ---------- 读全部层 ----------
    rows_by_layer = {}
    for L in ["L1", "L2", "L3", "L4"]:
        rows_by_layer[L] = [json.loads(l) for l in open(f"{B}/data/t3/layers/{L}.jsonl")]

    # ---------- 全局分子池：inchikey -> (smiles, 作用的簇集合) ----------
    pool_smi, pool_clust, pool_scaf = {}, defaultdict(set), {}
    for L, rows in rows_by_layer.items():
        for r in rows:
            ik = r["inchikey"]
            pool_smi.setdefault(ik, r["smiles"])
            pool_scaf.setdefault(ik, r.get("scaffold") or "")
            cl = clust.get(r["uniprot"])
            if cl:
                pool_clust[ik].add(cl)
    all_ik = sorted(pool_smi)
    print(f"全局分子池: {len(all_ik):,}", flush=True)

    summary = {}
    for L, rows in rows_by_layer.items():
        by_t = defaultdict(list)
        for r in rows:
            try:
                if float(r["paff"]) >= PAFF_CUT:
                    by_t[r["uniprot"]].append(r)
            except (TypeError, ValueError):
                pass

        out_path = f"{OUT}/{L}.jsonl"
        n_t, n_small, n_short, ratios = 0, 0, 0, []
        with open(out_path, "w") as fo:
            for up, acts in sorted(by_t.items()):
                # active 去重
                uniq = {}
                for r in acts:
                    uniq.setdefault(r["inchikey"], r)
                acts = list(uniq.values())
                if len(acts) < MIN_ACTIVES:
                    n_small += 1
                    continue

                act_ik = set(uniq)
                act_scaf = {r.get("scaffold") or "" for r in acts} - {""}
                my_cl = clust.get(up)

                want = len(acts) * RATIO
                cands = []
                for ik in all_ik:
                    if ik in act_ik:
                        continue
                    if my_cl and my_cl in pool_clust.get(ik, ()):   # 同簇靶点的配体
                        continue
                    if pool_scaf.get(ik) in act_scaf:               # 骨架撞车
                        continue
                    cands.append(ik)
                rng.shuffle(cands)
                dec = cands[:want]
                if len(dec) < want:
                    n_short += 1

                rec = {
                    "uniprot": up, "layer": L,
                    "n_actives": len(acts), "n_decoys": len(dec),
                    "ratio": round(len(dec) / len(acts), 1),
                    "actives": [{"smiles": r["smiles"], "inchikey": r["inchikey"],
                                 "paff": float(r["paff"])} for r in acts],
                    "decoys": [{"smiles": pool_smi[ik], "inchikey": ik} for ik in dec],
                }
                fo.write(json.dumps(rec) + "\n")
                n_t += 1
                ratios.append(rec["ratio"])

        summary[L] = {"targets": n_t, "dropped_few_actives": n_small,
                      "short_of_ratio": n_short,
                      "median_ratio": sorted(ratios)[len(ratios) // 2] if ratios else 0}
        print(f"{L}: 入选靶点 {n_t:4,}   因 active<{MIN_ACTIVES} 剔除 {n_small:5,}   "
              f"凑不满 1:{RATIO} 的 {n_short:3,}   实际比例中位 1:{summary[L]['median_ratio']:.0f}",
              flush=True)

    json.dump({"paff_cut": PAFF_CUT, "ratio": RATIO, "min_actives": MIN_ACTIVES,
               "decoy_scheme": "cross-target (mmseqs 40% cluster + scaffold exclusion)",
               "seed": SEED, "layers": summary},
              open(f"{OUT}/manifest.json", "w"), indent=1)
    print(f"\n已写入 {OUT}/")


if __name__ == "__main__":
    main()
