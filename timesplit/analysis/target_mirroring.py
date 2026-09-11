"""How homologous T3's targets are to the targets in the models' training
set — just how "novel" is a "novel target".

Background
------------
Mattsson & Walters (bioRxiv 2026.06.29.735309) call this phenomenon
**target mirroring**: homologous proteins can have highly correlated
binding profiles even when overall sequence identity is very low, so
splitting by sequence identity alone isn't enough to prevent leakage; on
ChEMBL 36 they found leakage persisting down to identities as low as
**0.2**.

Our L3/L4 are defined as "target unseen"; the family boundary comes from
LigUnity's supplied uniport40.clstr (CD-HIT 40% clustering). CD-HIT gates on
coverage as well as identity, so a pair it calls "different families" may
still have substantial local identity — within the T3 subset we've already
found VEGFR1(L4) at 45% identity to VEGFR2(L1). This script swaps in the
**full training set** as the comparison target, to give the true lower
bound.

Why mmseqs instead of cd-hit
------------
cd-hit's word-table mechanism makes it unreliable below 40% identity, which
is exactly the 20-40% range we need to look at. mmseqs' sensitive search
(-s 7.5) can reach remote homologues.

How to read the results
------------
* L1/L2 targets are already in the training set, so a 100% self-hit is
  expected and just verifies the pipeline is wired correctly.
* What matters is L3/L4's highest hit: if a large share land above 40%,
  "novel target" is a misnomer, these two layers' scores are inflated, and
  the reported decay is an underestimate.
"""
import argparse
import collections
import csv
import json
import os
import subprocess

B = "/data/work/vs-benchmark"
MMSEQS = "/data/work/tools/mmseqs/bin/mmseqs"


def write_fasta(path, seqs):
    with open(path, "w") as f:
        for k, v in seqs.items():
            f.write(f">{k}\n{v}\n")
    return len(seqs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=f"{B}/results/export/T3_targets.csv")
    ap.add_argument("--sequences", default=f"{B}/data/t3/sequences.json")
    ap.add_argument("--sequences-extra", default=f"{B}/data/t3/sequences_extra.json")
    ap.add_argument("--train-label", nargs="+",
                    default=[f"{B}/data/raw/figshare/train_label_blend_seq_full.json",
                             f"{B}/data/raw/figshare/train_label/train_label_pdbbind_seq.json"],
                    help="训练集标签文件，可给多个取并集。"
                         "LigUnity 系训练时读**两个**（train_task.py:523-524）："
                         "亲和力半 2,196 个 UniProt + 结构半 3,468 个，并集 4,847。"
                         "早先只给了前者，同源命中率因此被系统性低估。")
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--sensitivity", type=float, default=7.5)
    ap.add_argument("--min-cov", type=float, default=0.50,
                    help="比对须覆盖 query 和 target 各这么多，滤掉碎片命中")
    ap.add_argument("--max-evalue", type=float, default=1e-3)
    ap.add_argument("--reuse-hits", action="store_true",
                    help="复用已有的 hits.tsv，只换过滤口径重算")
    ap.add_argument("--workdir", default="/tmp/mirroring")
    ap.add_argument("--out", default=f"{B}/results/export/T3_target_mirroring.csv")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)

    # ---- Query set: T3 targets
    allseq = json.load(open(args.sequences))
    if os.path.exists(args.sequences_extra):
        allseq.update(json.load(open(args.sequences_extra)))
    rows = list(csv.DictReader(open(args.targets)))
    layers = collections.defaultdict(set)
    for r in rows:
        layers[r["uniprot"]].add(r["layer"])
    q = {u: allseq[u]["seq"] for u in layers
         if u in allseq and allseq[u].get("seq")}
    nq = write_fasta(f"{args.workdir}/query.fa", q)
    print(f"T3 唯一靶点 {len(layers)}，有序列 {nq}")

    # ---- Reference set: training-set targets (union of multiple label files, deduplicated by uniprot)
    paths = ([args.train_label] if isinstance(args.train_label, str)
             else list(args.train_label))
    lab, t = [], {}
    for p_ in paths:
        if not os.path.exists(p_):
            print(f"  跳过（不存在）{p_}")
            continue
        d = json.load(open(p_))
        lab += d
        n0 = len(t)
        for a in d:
            u, s = a.get("uniprot"), a.get("sequence")
            if u and s and u not in t:
                t[u] = s
        print(f"  {os.path.basename(p_):40} {len(d):6,} 条，新增 {len(t)-n0:5,} 个靶点")
    nt = write_fasta(f"{args.workdir}/train.fa", t)
    print(f"训练集条目合计 {len(lab):,}，去重后靶点 {nt:,}")

    overlap = set(q) & set(t)
    print(f"两边都出现的 UniProt: {len(overlap)}（L1/L2 本来就该在训练集里）\n")

    # ---- mmseqs sensitive search
    res = f"{args.workdir}/hits.tsv"
    cmd = [MMSEQS, "easy-search", f"{args.workdir}/query.fa",
           f"{args.workdir}/train.fa", res, f"{args.workdir}/tmp",
           "-s", str(args.sensitivity), "--max-seqs", "2000",
           "-e", "10000", "--threads", str(args.threads),
           "--format-output",
           "query,target,fident,alnlen,qcov,tcov,evalue,bits"]
    if args.reuse_hits and os.path.exists(res):
        print(f"复用已有 {res}")
    else:
        print("跑 mmseqs easy-search …", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:]); print(r.stderr[-2000:])
            raise SystemExit("mmseqs 失败")

    # ---- For each T3 target, the best homologue after excluding self-hits
    best_self, best_other = {}, {}
    for line in open(res):
        p = line.rstrip("\n").split("\t")
        if len(p) < 8:
            continue
        qu, tu, fid, aln, qc, tc = p[0], p[1], float(p[2]), int(p[3]), float(p[4]), float(p[5])
        # Without filtering, this hits a pile of "a few residues 100% identical"
        # fragments: the first pass already turned up fident=1.00 with qcov=1%
        # results, which are pure noise.
        if float(p[6]) > args.max_evalue or qc < args.min_cov or tc < args.min_cov:
            continue
        if qu == tu:
            best_self[qu] = max(best_self.get(qu, 0), fid)
            continue
        cur = best_other.get(qu)
        if cur is None or fid > cur[0]:
            best_self.setdefault(qu, 0.0)
            best_other[qu] = (fid, tu, qc, tc, aln)

    def layer_of(u):
        L = layers[u]
        return "L1/L2" if L & {"L1", "L2"} else "L3/L4"

    print(f"过滤口径：覆盖两条序列各 ≥{args.min_cov:.0%}，E ≤ {args.max_evalue:g}\n")
    print("=" * 70)
    print("自身命中检查（应为 L1/L2 命中 100%、L3/L4 无自身命中）")
    for grp in ("L1/L2", "L3/L4"):
        us = [u for u in q if layer_of(u) == grp]
        hit = [u for u in us if best_self.get(u, 0) > 0.99]
        print(f"  {grp}: {len(us)} 个靶点，自身 100% 命中 {len(hit)} 个")
        if grp == "L3/L4" and hit:
            print(f"    ⚠️ 这些不该出现在训练集里: {hit[:10]}")

    print("\n" + "=" * 70)
    print("与训练集靶点的最高一致性（已排除自身）")
    print("%-8s %7s %8s %8s %8s %8s %8s %8s" %
          ("层", "靶点", "中位", "≥70%", "≥50%", "≥40%", "≥30%", "≥20%"))
    print("-" * 70)
    for grp in ("L1/L2", "L3/L4"):
        us = [u for u in q if layer_of(u) == grp]
        v = sorted(best_other.get(u, (0.0,))[0] for u in us)
        if not v:
            continue
        cells = [f"{sum(1 for x in v if x >= t)}" for t in (.7, .5, .4, .3, .2)]
        print("%-8s %7d %7.1f%% %8s %8s %8s %8s %8s" %
              (grp, len(us), 100 * v[len(v) // 2], *cells))
    print("-" * 70)

    new = [u for u in q if layer_of(u) == "L3/L4"]
    hi = sorted(((best_other[u][0], u) + best_other[u][1:] for u in new
                 if u in best_other), reverse=True)
    print(f"\n「新靶点」里同源最高的 20 个（对训练集）：")
    print("%-10s %8s  %-10s %7s %7s" % ("T3 靶点", "一致性", "训练集靶点", "qcov", "tcov"))
    for fid, u, tu, qc, tc, aln in hi[:20]:
        print("%-10s %7.1f%%  %-10s %6.0f%% %6.0f%%" %
              (u, 100 * fid, tu, 100 * qc, 100 * tc))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uniprot", "layers", "best_train_hit", "identity",
                    "qcov", "tcov", "alnlen", "self_hit"])
        for u in sorted(q):
            b = best_other.get(u)
            w.writerow([u, "|".join(sorted(layers[u])),
                        b[1] if b else "", f"{b[0]:.4f}" if b else "",
                        f"{b[2]:.3f}" if b else "", f"{b[3]:.3f}" if b else "",
                        b[4] if b else "", f"{best_self.get(u, 0):.3f}"])
    print(f"\n逐靶点写入 {args.out}")


if __name__ == "__main__":
    main()
