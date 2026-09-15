"""Per-target training-set membership for the 350-quota subset, as a committed table.

Why this exists
----
Two claims in the write-up rest on which training half a target belongs to, and
until now both could only be checked by re-deriving the sets from files that are
not in this repository:

1. The layer labels are contaminated — some "sequence-novel" targets are in a
   training set (`LIMITATIONS.md`: 4 of 19 L3 and 8 of 75 L4 in the subset were
   seen through the structure half).
2. The default mirroring table misses them — those same targets carry
   `self_hit = 0` in `T3_target_mirroring.csv` while the union table reports a
   100% self-match.

Those are the same 12 targets, not two separate problems, and that is worth
stating in a paper. A reader cannot check it from aggregate tables: every
committed membership table (`T3_train_set_crossover.csv`,
`T3_per_model_seen_effect.csv`, `T3_novelty_by_own_train.csv`) is per model, with
no uniprot column. Hence this script, and the per-target CSV it writes.

⚠️ "The structure half" has two non-equivalent derivations in this project
-------
| Derivation | Source | UniProts | Subset L4 hits |
|---|---|---|---|
| the label file's `uniprot` field | `train_label/train_label_pdbbind_seq.json` | 3,468 | **8** |
| lmdb pockets mapped through PDB→UniProt | `train_no_test_af/train.lmdb` + `drugclip_pdb2uniprot.json` | 3,551 | 15 |

The published 4 / 8 use the first, which is what `train_set_crossover.py`
means by P, and this script follows it. The second (`build_train_union.py`'s
`set_a_pairs`) counts 15 in subset L4. Both are defensible readings of "DrugCLIP
trained on this target"; they are not interchangeable, so any statement about
structure-half membership has to say which one it used.

Output columns
-------
`in_P` / `in_L` are membership in the structure half and the affinity half.
`plain_*` / `union_*` come from the two mirroring tables, and `plain_bin` is the
homology bin a target falls into under the **default** mirroring table — the
column that shows what the default gets wrong.
"""
import argparse
import csv
import json
import os

B = "/data/work/vs-benchmark"

P_FILE = f"{B}/data/raw/figshare/train_label/train_label_pdbbind_seq.json"
L_FILE = f"{B}/data/raw/figshare/train_label_blend_seq_full.json"


def uniprots(path):
    """UniProt set of one training-label file, read exactly as train_set_crossover.py reads it."""
    if not os.path.exists(path):
        raise SystemExit(f"missing training-label file: {path}")
    return {a["uniprot"] for a in json.load(open(path)) if a.get("uniprot")}


def mirroring(path):
    return {r["uniprot"]: r for r in csv.DictReader(open(path))}


def as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def bin_of(identity):
    """The homology bin used by the protein axis: no hit / <20% / 20-60% / >=60%."""
    if identity is None:
        return "no_hit"
    if identity >= 0.60:
        return ">=60%"
    if identity >= 0.20:
        return "20-60%"
    return "<20%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=f"{B}/results/export/T3_vsds_matched.csv")
    ap.add_argument("--plain", default=f"{B}/results/export/T3_target_mirroring.csv")
    ap.add_argument("--union", default=f"{B}/results/export/T3_target_mirroring_union.csv")
    ap.add_argument("--out", default=f"{B}/results/export/T3_subset_train_membership.csv")
    a = ap.parse_args()

    P, L = uniprots(P_FILE), uniprots(L_FILE)
    MP, MU = mirroring(a.plain), mirroring(a.union)
    rows = list(csv.DictReader(open(a.subset)))

    out = [["layer", "uniprot", "in_P_structure_half", "in_L_affinity_half",
            "plain_self_hit", "plain_identity", "plain_best_train_hit", "plain_bin",
            "union_self_hit", "union_identity", "union_bin"]]
    for r in rows:
        u, L_ = r["uniprot"], r["layer"]
        p, v = MP.get(u, {}), MU.get(u, {})
        pi, ui = as_float(p.get("identity")), as_float(v.get("identity"))
        out.append([L_, u,
                    int(u in P), int(u in L),
                    as_float(p.get("self_hit")) or 0.0, "" if pi is None else f"{pi:.4f}",
                    p.get("best_train_hit", ""), bin_of(pi),
                    as_float(v.get("self_hit")) or 0.0, "" if ui is None else f"{ui:.4f}",
                    bin_of(ui)])

    with open(a.out, "w", newline="") as f:
        csv.writer(f).writerows(out)

    # ---- the two claims this table exists to make checkable ----
    body = out[1:]
    print(f"P (structure half) {len(P)}  ·  L (affinity half) {len(L)}  ·  "
          f"subset records {len(body)}")
    print("\nSubset records in the structure half, by layer "
          "(this reproduces LIMITATIONS' 4 of 19 L3 / 8 of 75 L4):")
    for layer in ("L1", "L2", "L3", "L4"):
        inlayer = [r for r in body if r[0] == layer]
        inP = [r for r in inlayer if r[2]]
        onlyP = [r for r in inP if not r[3]]
        print(f"  {layer}: {len(inlayer):>3} records · in P {len(inP):>3} · "
              f"in P only {len(onlyP):>3}")

    novel = [r for r in body if r[0] in ("L3", "L4") and r[8] >= 1.0 and r[4] == 0.0]
    print(f"\nSequence-novel layers where the union table reports a 100% self-match "
          f"but the default table reports none: {len(novel)}")
    print("  the default table calls every one of them 'not in training'; "
          "where it lands them:")
    for b in (">=60%", "20-60%", "<20%", "no_hit"):
        got = [r[1] for r in novel if r[7] == b]
        if got:
            print(f"    {b:8} {len(got)}  {' '.join(sorted(got))}")
    print(f"\n  of those, in the structure half: {sum(1 for r in novel if r[2])}"
          f" · in the affinity half: {sum(1 for r in novel if r[3])}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
