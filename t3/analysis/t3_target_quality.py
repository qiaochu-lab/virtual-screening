"""按结构质量给 T3 靶点分级，并给出「高质量子集」。

来源：项目要求——
「最好保留的数据是有结构（rcsb 有蛋白晶体结构）或者结构预测比较精准的
（iptm ptm plddt 比较高），以及多样性，要涉及到激酶、gpcr、表观等等」

分三级
------
A 级  有 RCSB 实验晶体结构（口袋直接从共晶复合物截出）
B 级  Boltz-2 预测且置信度达标
C 级  Boltz-2 预测但置信度不达标

置信度阈值参考 AlphaFold/Boltz 的通行口径：
  · complex_plddt ≥ 0.70   —— 主链局部置信度，<0.5 基本不可信
  · iptm         ≥ 0.60    —— 链间界面置信度，对「配体摆得对不对」最关键
两者都要满足才算 B 级。注意这里的 iptm 是蛋白-配体界面，
比单看 plddt 更贴近「口袋位置对不对」这个我们真正关心的问题。

输出高质量子集（A+B）的名单，供后续做敏感性分析：
若主结论在高质量子集上仍成立，说明结论不依赖低质量预测结构。
"""
import glob
import json
import os
import pickle
from collections import Counter, defaultdict

import lmdb
import numpy as np

B = "/data/work/vs-benchmark"
PLDDT_MIN, IPTM_MIN = 0.70, 0.60


def load_conf():
    """uniprot -> Boltz-2 置信度。"""
    out = {}
    for d in ["boltz_batch_out", "boltz_retry_out", "boltz_gap_out", "boltz_r2_out"]:
        for p in glob.glob(f"{B}/{d}/**/confidence_*.json", recursive=True):
            name = os.path.basename(p)
            up = name.replace("confidence_", "").split("_model")[0]
            try:
                out[up] = json.load(open(p))
            except Exception:
                pass
    return out


def pocket_source():
    """uniprot -> 'pdb_holo' | 'boltz2_pred'（与组装 T3 数据时一致）。"""
    src = {}
    for pref, tag in [("pocket", "boltz2_pred"), ("pdb_pocket", "pdb_holo")]:
        p = f"{B}/data/t3/pockets/{pref}_6.0A.lmdb"
        if not os.path.exists(p):
            continue
        e = lmdb.open(p, subdir=False, readonly=True, lock=False)
        with e.begin() as t:
            for _, v in t.cursor():
                src[pickle.loads(v)["pocket"]] = tag   # pdb 源后加载，自然覆盖
        e.close()
    return src


def main():
    conf = load_conf()
    src = pocket_source()
    cls = json.load(open(f"{B}/data/t3/target_class.json"))["class"]

    eval_targets = {}
    for L in ["L1", "L2", "L3", "L4"]:
        p = f"{B}/data/t3/eval/{L}.jsonl"
        if not os.path.exists(p):
            continue
        for line in open(p):
            eval_targets.setdefault(json.loads(line)["uniprot"], []).append(L)

    print(f"评测集靶点: {len(eval_targets):,}   Boltz-2 置信度记录: {len(conf):,}\n")

    grade, detail = {}, {}
    for up in eval_targets:
        s = src.get(up)
        if s == "pdb_holo":
            grade[up] = "A 实验结构"
            continue
        c = conf.get(up)
        if not c:
            grade[up] = "C 无置信度记录"
            continue
        pl, ip = c.get("complex_plddt", 0), c.get("iptm", 0)
        detail[up] = (pl, ip)
        grade[up] = ("B 预测·置信达标" if pl >= PLDDT_MIN and ip >= IPTM_MIN
                     else "C 预测·置信不足")

    print("=" * 66)
    print(f"结构质量分级（plddt ≥ {PLDDT_MIN}, iptm ≥ {IPTM_MIN}）")
    print("=" * 66)
    tot = len(eval_targets)
    for g, n in sorted(Counter(grade.values()).items()):
        print(f"  {g:20s} {n:5,}  ({n/tot*100:5.1f}%)")

    if detail:
        pls = np.array([v[0] for v in detail.values()])
        ips = np.array([v[1] for v in detail.values()])
        print(f"\n预测结构的置信度分布（n={len(detail):,}）:")
        print(f"  complex_plddt  中位 {np.median(pls):.3f}   ≥{PLDDT_MIN} 的 {(pls>=PLDDT_MIN).mean()*100:.1f}%")
        print(f"  iptm           中位 {np.median(ips):.3f}   ≥{IPTM_MIN} 的 {(ips>=IPTM_MIN).mean()*100:.1f}%")

    # 按层
    print("\n" + "=" * 66)
    print("各层的质量构成")
    print("=" * 66)
    per_layer = defaultdict(Counter)
    for up, ls in eval_targets.items():
        for L in ls:
            per_layer[L][grade[up][0]] += 1     # 取首字母 A/B/C
    print("%-5s %8s %8s %8s %8s %10s" % ("层", "A实验", "B达标", "C不足", "合计", "高质量占比"))
    print("-" * 56)
    for L in ["L1", "L2", "L3", "L4"]:
        c = per_layer[L]
        n = sum(c.values())
        hq = c["A"] + c["B"]
        if n:
            print("%-5s %8d %8d %8d %8d %9.1f%%" % (L, c["A"], c["B"], c["C"], n, hq/n*100))

    # 高质量子集的类别覆盖 —— 质量之外还要求类别多样性
    print("\n" + "=" * 66)
    print("高质量子集（A+B）的靶点类别覆盖")
    print("=" * 66)
    hq_targets = [u for u in eval_targets if grade[u][0] in "AB"]
    new_hq = [u for u in hq_targets if any(L in ("L3", "L4") for L in eval_targets[u])]
    allc = Counter(cls.get(u, "其他/未分类") for u in hq_targets)
    newc = Counter(cls.get(u, "其他/未分类") for u in new_hq)
    print("%-14s %10s %14s" % ("类别", "全部层", "新靶点层"))
    print("-" * 42)
    for k in ["GPCR", "激酶", "表观", "蛋白酶", "离子通道", "核受体",
              "P450", "转运体", "其他酶", "其他/未分类"]:
        print("%-14s %10d %14d" % (k, allc.get(k, 0), newc.get(k, 0)))

    out = f"{B}/data/t3/target_quality.json"
    json.dump({"grade": grade,
               "high_quality": sorted(hq_targets),
               "thresholds": {"complex_plddt": PLDDT_MIN, "iptm": IPTM_MIN},
               "confidence": {u: {"complex_plddt": v[0], "iptm": v[1]}
                              for u, v in detail.items()}},
              open(out, "w"), indent=1)
    print(f"\n高质量靶点 {len(hq_targets):,} 个，名单已写入 {out}")
    print("下一步：在高质量子集上重算主表，检验结论是否依赖低质量预测结构")


if __name__ == "__main__":
    main()
