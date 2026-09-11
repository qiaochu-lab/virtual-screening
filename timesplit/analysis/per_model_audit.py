"""Per-model training-set novelty audit.

Why this is needed
--------------------
T3's L1-L4 layering currently is decided **only using LigUnity's training
set** (PocketAffDB), so "L1" literally means "PocketAffDB has seen this
target". Using that label to compare "models trained on PocketAffDB" against
"models trained on DrugCLIP's set" and then concluding "training data
matters more than architecture" is circular -- the label itself was drawn
against one of the two groups' training set.

The correct approach is to compute each model's own seen/unseen:

    S_protein^(m)(t) = max_{p in Train_m} Identity(t, p)
    S_ligand^(m)(x)  = max_{z in Train_m} Tanimoto(x, z)

Two training sets
-------------------
The seven pocket-family models actually reduce to only two (see
build_train_union.py):
  A  train_no_test_af  ->  DrugCLIP, BindCLIP-randneg, BindCLIP-hardneg
  B  PocketAffDB       ->  LigUnity x2, LiTENCLIP, HypSeek
ConPLex's official release publishes training sequences but no accessions,
so its list is reverse-looked-up with mmseqs (conplex_t3_cov.py,
bidirectional coverage >=50%, identity >=95%), listed separately as a third
set C.
ConGLUDe / SPRINT's lists are still unobtained, and marked unavailable in the
output.
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
# ConPLex's training targets are reverse-looked-up from sequences; identity threshold
CONPLEX_IDENT = 0.95


def load_B():
    """Group B's (LigUnity x2 / LiTENCLIP / HypSeek) training targets = the
    union of two label files.

    train_task.py:523-524 reads both train_label_pdbbind_seq.json (the
    structure half, 3,468 UniProt / 16,744 PDB) and
    train_label_blend_seq_full.json (the affinity half, 2,196). Only the
    latter was counted previously.

    And the structure half's 16,744 PDB entries are **exactly identical** to
    DrugCLIP's train_no_test_af (intersection 16,744, neither has any
    exclusive entries), so group A's training structures are a proper
    subset of group B's.
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
    """The affinity half only -- used to separate "affinity labels" from
    "structure"."""
    lab = json.load(open(f"{B}/data/raw/figshare/train_label_blend_seq_full.json"))
    return {a["uniprot"] for a in lab if a.get("uniprot")}


def load_C():
    """ConPLex's BindingDB training sequences, reverse-looked-up onto T3
    targets with mmseqs."""
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
