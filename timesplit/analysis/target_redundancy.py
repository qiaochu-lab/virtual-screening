"""选出的靶点子集内部，彼此的序列相似性有多高——要不要再删。

背景
----
合作者 2026-09-05 的问题：「目前剩余的 200+ 彼此的序列相似性在多少？」
测试集内部如果有高度同源的靶点，等于同一个东西数了两遍：指标被重复计数，
配对检验的独立性假设也不成立。公开基准一般会做去冗余（DUD-E 用 UniProt
层面去重，LIT-PCBA 明确排除同源靶点）。

两种口径都给
------------
1. cd-hit 在若干阈值下聚类 —— 直接回答「按 X% 去冗余会剩多少个靶点」
2. 全对全的局部比对一致性 —— 给分布、每个靶点的最近邻、以及超阈值的具体配对

一致性怎么算
------------
局部比对（BLOSUM62，gap -11/-1，与 BLAST 默认一致），
identity = 完全匹配的残基数 / 较短序列长度。用较短序列作分母而不是比对长度，
是为了不让「一个短结构域命中一个长蛋白」显示成低相似——那种情况恰恰是要抓的。
"""
import argparse
import csv
import collections
import itertools
import json
import os
import subprocess
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

B = "/data/work/vs-benchmark"
CDHIT = "/data/work/tools/cdhit/cd-hit"
SEQS = {}


def _init(seqs):
    global SEQS
    SEQS = seqs


def _identity(pair):
    from Bio import Align
    from Bio.Align import substitution_matrices
    al = Align.PairwiseAligner(mode="local", open_gap_score=-11,
                               extend_gap_score=-1)
    al.substitution_matrix = substitution_matrices.load("BLOSUM62")
    a, b = pair
    sa, sb = SEQS[a], SEQS[b]
    try:
        aln = al.align(sa, sb)[0]
        # aln.aligned 给的是两条序列上互相对应的区间，比解析带 gap 的字符串稳，
        # 不依赖 Biopython 版本对 aln[0] 的返回类型（1.80 前后改过）。
        same = 0
        for (i0, i1), (j0, j1) in zip(*aln.aligned, strict=True):
            same += sum(1 for k in range(i1 - i0) if sa[i0 + k] == sb[j0 + k])
    except Exception:
        return a, b, float("nan")
    return a, b, same / min(len(sa), len(sb))


def cdhit_curve(seqs, thresholds=(0.9, 0.8, 0.7, 0.6, 0.5, 0.4)):
    """cd-hit 在各阈值下还剩多少个代表序列。word size 必须跟阈值匹配。"""
    out = {}
    with tempfile.TemporaryDirectory() as td:
        fa = f"{td}/in.fa"
        with open(fa, "w") as f:
            for k, v in seqs.items():
                f.write(f">{k}\n{v}\n")
        for c in thresholds:
            n = 5 if c >= 0.7 else (4 if c >= 0.6 else (3 if c >= 0.5 else 2))
            r = subprocess.run([CDHIT, "-i", fa, "-o", f"{td}/o{c}",
                                "-c", str(c), "-n", str(n), "-M", "4000",
                                "-T", "8", "-d", "0"],
                               capture_output=True, text=True)
            if r.returncode != 0 or not os.path.exists(f"{td}/o{c}"):
                out[c] = None
                print(f"  cd-hit c={c} 失败: "
                      f"{(r.stderr or r.stdout or '').strip().splitlines()[-1:]}" )
                continue
            out[c] = sum(1 for l in open(f"{td}/o{c}") if l.startswith(">"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=f"{B}/results/export/T3_vsds_matched.csv",
                    help="靶点清单 CSV，需含 uniprot / layer / protein_class 列")
    ap.add_argument("--sequences", default=f"{B}/data/t3/sequences.json")
    ap.add_argument("--sequences-extra",
                    default=f"{B}/data/t3/sequences_extra.json")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--report-above", type=float, default=0.40)
    ap.add_argument("--out", default=f"{B}/results/export/T3_target_redundancy.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.targets)))
    allseq = json.load(open(args.sequences))
    if args.sequences_extra and os.path.exists(args.sequences_extra):
        extra = json.load(open(args.sequences_extra))
        allseq.update(extra)                      # UniProt REST 补的那批
        print(f"补充序列 {len(extra)} 条")
    # sequences.json 是 {uniprot: {"seq":..., "length":..., "name":...}}
    seqs = {r["uniprot"]: allseq[r["uniprot"]]["seq"]
            for r in rows if r["uniprot"] in allseq and allseq[r["uniprot"]].get("seq")}
    meta = {r["uniprot"]: (r.get("layer", ""), r.get("protein_class", "")) for r in rows}
    missing = [r["uniprot"] for r in rows if r["uniprot"] not in allseq]
    print(f"靶点 {len(rows)} 个，拿到序列 {len(seqs)} 个"
          + (f"，缺 {len(missing)}: {missing[:5]}" if missing else ""))

    print("\n=== cd-hit 去冗余：按各阈值聚类后还剩多少 ===")
    cur = cdhit_curve(seqs)
    print("%-10s %10s %10s" % ("阈值", "剩余靶点", "删掉"))
    print("-" * 32)
    for c, n in cur.items():
        print("%-10s %10s %10s" % (f"{c:.0%}", n if n else "失败",
                                   len(seqs) - n if n else "-"))

    ups = sorted(seqs)
    pairs = list(itertools.combinations(ups, 2))
    print(f"\n=== 全对全局部比对：{len(pairs):,} 对 ===", flush=True)
    res = []
    with ProcessPoolExecutor(args.workers, initializer=_init,
                             initargs=(seqs,)) as ex:
        for i, r in enumerate(ex.map(_identity, pairs, chunksize=64)):
            res.append(r)
            if (i + 1) % 5000 == 0:
                print(f"  {i+1:,}/{len(pairs):,}", flush=True)

    import math
    bad = sum(1 for x in res if math.isnan(x[2]))
    if bad:
        print(f"⚠️ {bad} 对比对失败，已从统计剔除")
    res = [x for x in res if not math.isnan(x[2])]
    vals = sorted(x[2] for x in res)
    q = lambda p: vals[int(p * (len(vals) - 1))]
    print(f"\n所有配对的一致性分布：")
    print(f"  中位 {q(.5):.1%}   90% 分位 {q(.9):.1%}   99% 分位 {q(.99):.1%}   最大 {vals[-1]:.1%}")
    for t in (0.9, 0.7, 0.5, 0.4, 0.3):
        n = sum(1 for v in vals if v >= t)
        print(f"  ≥{t:.0%}: {n:,} 对 ({100*n/len(vals):.2f}%)")

    # 按类别拆开：激酶之间本来就同源，跨类别（激酶 vs GPCR）几乎为零，
    # 混在一起算中位数只反映类别构成，不反映冗余。
    cls = {u: meta[u][1] for u in ups}
    within = defaultdict(list)
    for a, b, v in res:
        if cls[a] == cls[b]:
            within[cls[a]].append(v)
    print("\n=== 类内两两一致性（判断冗余要看这个）===")
    print("%-12s %7s %9s %9s %9s %9s %9s" %
          ("类别", "靶点", "配对", "中位", "90%", "最大", "≥40%对"))
    print("-" * 68)
    n_by_cls = collections.Counter(cls.values())
    for c, v in sorted(within.items(), key=lambda x: -len(x[1])):
        v = sorted(v)
        if len(v) < 3:
            continue
        print("%-12s %7d %9d %8.1f%% %8.1f%% %8.1f%% %9d" %
              (c, n_by_cls[c], len(v), 100 * v[len(v)//2],
               100 * v[int(.9*(len(v)-1))], 100 * v[-1],
               sum(1 for x in v if x >= .4)))
    cross = [v for a, b, v in res if cls[a] != cls[b]]
    cross.sort()
    print("-" * 68)
    print("%-12s %7s %9d %8.1f%% %8.1f%% %8.1f%% %9d" %
          ("(跨类别)", "", len(cross), 100 * cross[len(cross)//2],
           100 * cross[int(.9*(len(cross)-1))], 100 * cross[-1],
           sum(1 for x in cross if x >= .4)))

    nn = defaultdict(float)
    for a, b, v in res:
        nn[a] = max(nn[a], v)
        nn[b] = max(nn[b], v)
    nnv = sorted(nn.values())
    print(f"\n每个靶点「最近邻」的一致性：中位 {nnv[len(nnv)//2]:.1%}   "
          f"最大 {nnv[-1]:.1%}")
    for t in (0.9, 0.7, 0.5, 0.4):
        n = sum(1 for v in nn.values() if v >= t)
        print(f"  有 ≥{t:.0%} 同源伙伴的靶点: {n} / {len(nn)} ({100*n/len(nn):.0f}%)")

    hi = sorted([x for x in res if x[2] >= args.report_above],
                key=lambda x: -x[2])
    print(f"\n=== 一致性 ≥{args.report_above:.0%} 的配对（{len(hi)} 对）===")
    for a, b, v in hi[:40]:
        print(f"  {v:6.1%}  {a} [{meta[a][0]} {meta[a][1]}]  ×  "
              f"{b} [{meta[b][0]} {meta[b][1]}]")
    if len(hi) > 40:
        print(f"  …还有 {len(hi)-40} 对，见 CSV")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uniprot_a", "layer_a", "class_a",
                    "uniprot_b", "layer_b", "class_b", "identity"])
        for a, b, v in sorted(res, key=lambda x: -x[2]):
            if v >= 0.20:
                w.writerow([a, *meta[a], b, *meta[b], f"{v:.4f}"])
    print(f"\n≥20% 的配对写入 {args.out}")


if __name__ == "__main__":
    main()
