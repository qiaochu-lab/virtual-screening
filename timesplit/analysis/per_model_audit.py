"""逐模型的训练集新颖度审计。

为什么需要
----------
现在 T3 的 L1–L4 分层**只用 LigUnity 的训练集**（PocketAffDB）判定，
所以「L1」字面意思是「PocketAffDB 见过这个靶点」。用这个标签去比较
「用 PocketAffDB 训的模型」和「用 DrugCLIP 集训的模型」，再得出
「训练数据比架构更重要」，是循环论证——标签本身就是按其中一组的训练集画的。

正确做法是逐模型算它自己的 seen/unseen：

    S_protein^(m)(t) = max_{p ∈ Train_m} Identity(t, p)
    S_ligand^(m)(x)  = max_{z ∈ Train_m} Tanimoto(x, z)

两套训练集
----------
七个口袋系模型实际只有两套（见 build_train_union.py）：
  A  train_no_test_af  →  DrugCLIP、BindCLIP-randneg、BindCLIP-hardneg
  B  PocketAffDB       →  LigUnity ×2、LiTENCLIP、HypSeek
ConPLex 官方只发布训练序列不发布 accession，所以它的清单是用 mmseqs 反查的
（conplex_t3_cov.py，双向覆盖 ≥50%、同一性 ≥95%），单列为第三套 C。
ConGLUDe / SPRINT 的清单仍未获得，输出里标 unavailable。
"""
import argparse, collections, csv, json, os, pickle, sys
import numpy as np

B = "/data/work/vs-benchmark"
GROUP = {
    "A(train_no_test_af)": ["drugclip", "bindclip_randneg", "bindclip_hardneg"],
    "B(PocketAffDB)": ["ligunity_pocket_ranking", "ligunity_protein_ranking",
                       "litenclip", "hypseek_rk"],
    "C(ConPLex BindingDB)": ["conplex"],
    "unavailable": ["conglude", "sprint"],
}
# ConPLex 的训练靶点是序列反查来的，同一性门限
CONPLEX_IDENT = 0.95


def load_B():
    """B 组（LigUnity ×2 / LiTENCLIP / HypSeek）的训练靶点 = 两个标签文件的并集。

    train_task.py:523-524 同时读 train_label_pdbbind_seq.json（结构半，3,468 个
    UniProt / 16,744 个 PDB）和 train_label_blend_seq_full.json（亲和力半，
    2,196 个）。此前只算了后者。

    而结构半覆盖的 16,744 个 PDB 与 DrugCLIP 的 train_no_test_af **完全相同**
    （交集 16,744，各自独有 0），所以 A 组训练结构是 B 组的真子集。
    """
    ups = set()
    for f in ("train_label_blend_seq_full.json",
              f"train_label/train_label_pdbbind_seq.json"):
        p = f"{B}/data/raw/figshare/{f}"
        if not os.path.exists(p):
            continue
        for a in json.load(open(p)):
            if a.get("uniprot"):
                ups.add(a["uniprot"])
    return ups


def load_B_blend_only():
    """只有亲和力半 —— 用来把「亲和力标签」和「结构」两件事分开。"""
    lab = json.load(open(f"{B}/data/raw/figshare/train_label_blend_seq_full.json"))
    return {a["uniprot"] for a in lab if a.get("uniprot")}


def load_C():
    """ConPLex 的 BindingDB 训练序列，mmseqs 反查到 T3 靶点上的结果。"""
    p = f"{B}/results/export/T3_conplex_train_coverage.csv"
    if not os.path.exists(p):
        return None
    return {r["uniprot"] for r in csv.DictReader(open(p))
            if float(r["best_identity_conplex_bindingdb"]) >= CONPLEX_IDENT
            or r["in_dude_57"] == "1"}


def load_A():
    pdb2up = json.load(open(f"{B}/data/t3/drugclip_pdb2uniprot.json"))
    import lmdb
    e = lmdb.open(f"{B}/data/train_no_test_af/train.lmdb",
                  subdir=False, readonly=True, lock=False)
    ups = set()
    with e.begin() as t:
        for _, v in t.cursor():
            d = pickle.loads(v)
            pk = d.get("pocket")
            if not pk:
                continue
            for u in pdb2up.get(str(pk).split("_")[0].upper()[:4], []):
                ups.add(u)
    e.close()
    return ups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=f"{B}/results/export/T3_vsds_matched.csv")
    ap.add_argument("--out", default=f"{B}/results/export/T3_per_model_audit.csv")
    args = ap.parse_args()

    rows_t = list(csv.DictReader(open(args.subset)))
    by_layer = collections.defaultdict(set)
    for r in rows_t:
        by_layer[r["layer"]].add(r["uniprot"])

    print("载入两套训练集的靶点…", flush=True)
    A, Bs, C = load_A(), load_B(), load_C()
    print(f"  A train_no_test_af: {len(A):,} UniProt")
    print(f"  B PocketAffDB:      {len(Bs):,} UniProt")
    print(f"  两套的交集:          {len(A & Bs):,}   并集: {len(A | Bs):,}")
    if C is None:
        print("  C ConPLex:          缺 T3_conplex_train_coverage.csv，跳过")
    else:
        print(f"  C ConPLex(≥{CONPLEX_IDENT:.0%}):    {len(C):,} UniProt（覆盖到 T3 的部分）")

    print("\n各层靶点被两套训练集覆盖的比例（精确 UniProt 匹配）")
    print("=" * 74)
    print("%-6s %8s %14s %14s %14s %14s" %
          ("层", "靶点", "A 见过", "B 见过", "C 见过", "三套都没见过"))
    print("-" * 90)
    out = [["layer", "n_targets", "seen_by_A", "pct_A", "seen_by_B", "pct_B",
            "seen_by_C", "pct_C", "seen_by_none", "pct_none"]]
    for L in ("L1", "L2", "L3", "L4"):
        ts = by_layer.get(L)
        if not ts:
            continue
        a = sum(1 for t in ts if t in A)
        b = sum(1 for t in ts if t in Bs)
        c = sum(1 for t in ts if C and t in C)
        n = sum(1 for t in ts
                if t not in A and t not in Bs and not (C and t in C))
        print("%-6s %8d %7d (%3.0f%%) %7d (%3.0f%%) %7d (%3.0f%%) %7d (%3.0f%%)" %
              (L, len(ts), a, 100*a/len(ts), b, 100*b/len(ts),
               c, 100*c/len(ts), n, 100*n/len(ts)))
        out.append([L, len(ts), a, f"{100*a/len(ts):.1f}", b, f"{100*b/len(ts):.1f}",
                    c, f"{100*c/len(ts):.1f}", n, f"{100*n/len(ts):.1f}"])
    print("-" * 90)

    print("\n分层标签是按 B 画的，所以 B 的覆盖率就是分层定义本身；")
    print("A 的覆盖率与之相差多少，就是「用 A 训练的模型」被这套标签错配了多少。")

    print("\n模型 → 训练集归属")
    for g, ms in GROUP.items():
        print(f"  {g:22} {', '.join(ms)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(out)
    print(f"\n写入 {args.out}")


if __name__ == "__main__":
    main()
