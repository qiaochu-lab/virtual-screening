"""T6's second physics method: molecular docking (smina).

Why it is needed
-----------------
T6's physics side currently has only Boltz-2, so every conclusion reads as
"what Boltz-2 does" rather than "what physics methods do". Boltz-2 also
takes tens of seconds per complex, which cannot support two things:
  . physics-only screening at full scale (T3 has thousands of molecules per
    target)
  . reranking a deep shortlist -- recall@50 on L4 is only 17.5%, and ranking
    deeper requires something cheap
smina takes seconds per molecule and fills exactly those two gaps.

How the box is defined
------------------------
Use the 6A pocket we already extracted directly: take the bounding box of
all pocket atom coordinates, with a 4A margin on each side. This way
**docking and the retrieval models see the same pocket**, which is what
makes the comparison valid -- finding the pocket separately with something
like fpocket would turn the comparison into one about "pocket definition"
instead.

Receptor preparation
----------------------
The pocket lmdb holds only atom types and coordinates, no connectivity or
charges. smina can read PDB directly, so a PDB is written out from the
pocket atoms, then obabel adds hydrogens and Gasteiger charges and converts
to pdbqt.
Warning: this is "pocket-slice docking", not whole-protein docking: residues
at the slice's edge lack their neighbor constraints, which biases the score.
But the effect on **the relative ranking of the same batch of molecules** is
limited, and that is exactly the quantity we are measuring.
"""
import argparse
import json
import os
import pickle
import subprocess

import lmdb
import numpy as np

B = "/data/work/vs"
DOCK = "/data/work/envs/dock/bin"
PAD = 4.0


def write_pocket_pdb(rec, path):
    """Pocket atoms -> PDB. Residue info is already lost, so everything is written as POC/A/1; smina only uses coordinates and element."""
    coords = np.asarray(rec["pocket_coordinates"], dtype=float)
    atoms = rec["pocket_atoms"]
    with open(path, "w") as f:
        # must raise on a length mismatch between atom names and coordinates -- a silent truncation would
        # write out a corrupted pocket PDB, and docking would still run to completion with a wrong box and receptor.
        for i, (a, c) in enumerate(zip(atoms, coords, strict=True), 1):
            el = "".join(ch for ch in a if ch.isalpha())[:1] or "C"
            f.write(f"ATOM  {i:5d} {a[:4]:<4s} POC A   1    "
                    f"{c[0]:8.3f}{c[1]:8.3f}{c[2]:8.3f}  1.00  0.00          {el:>2s}\n")
        f.write("END\n")
    return coords


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default="L4")
    ap.add_argument("--targets", type=int, default=20)
    ap.add_argument("--topn", type=int, default=200,
                    help="每靶点取检索模型的 top-N 送对接（深于 Boltz-2 那轮的 50）")
    ap.add_argument("--model", default="ligunity_protein_ranking")
    ap.add_argument("--subset", default=None,
                    help="csv：只从这批靶点里选（例如 350 配额子集）")
    ap.add_argument("--seed", type=int, default=1,
                    help="靶点抽样种子。不给 --subset 时保持旧行为（字母序取前 N）")
    ap.add_argument("--max-box", type=float, default=0.0,
                    help="盒子体积上限 Å³，0 = 不限。"
                         "大盒子会撞 run_dock.sh 的 90 分钟上限、一个分数都出不来，"
                         "所以选靶点时就该排除，而不是跑完才发现")
    ap.add_argument("--report-boxes", action="store_true",
                    help="只打印候选靶点的盒子体积分布，不生成任何文件")
    ap.add_argument("--out", default=f"{B}/dock")
    args = ap.parse_args()

    pockets = {}
    for pref in ("pocket", "pdb_pocket"):
        p = f"{B}/data/t3/pockets/{pref}_6.0A.lmdb"
        if not os.path.exists(p):
            continue
        e = lmdb.open(p, subdir=False, readonly=True, lock=False)
        with e.begin() as t:
            for _k, v in t.cursor():
                d = pickle.loads(v)
                pockets[d["pocket"]] = d
        e.close()

    ev = {json.loads(x)["uniprot"]: json.loads(x)
          for x in open(f"{B}/data/t3/eval/{args.layer}.jsonl")}
    root = f"{B}/results/t3_raw/{args.model}/T3/{args.layer}"
    hq = set(json.load(open(f"{B}/data/t3/target_quality.json"))["high_quality"])

    keep = None
    if args.subset:
        import csv as _csv
        keep = {r["uniprot"] for r in _csv.DictReader(open(args.subset))
                if r.get("layer") == args.layer}
        print(f"限定在子集的 {args.layer}：{len(keep)} 个靶点")

    # collect all qualifying targets first, then sample. The old behaviour was sorted()[:N] --
    # that is **alphabetical order**, neither random nor quality-based, effectively selecting
    # targets by UniProt accession. When --subset is given, switch to a fixed-seed random draw
    # to avoid that bias.
    cand = []
    for up in sorted(os.listdir(root)):
        if up not in pockets or up not in ev or up not in hq:
            continue
        if keep is not None and up not in keep:
            continue
        try:
            pr = np.load(f"{root}/{up}/saved_preds.npy").reshape(-1)
            y = np.load(f"{root}/{up}/saved_labels.npy")
        except Exception:
            continue
        if len(pr) != len(y) or y.sum() < 5:
            continue
        cand.append((up, pr, y))
    if keep is not None and len(cand) > args.targets:
        import random
        random.Random(args.seed).shuffle(cand)
    # box volume must be checked at target-selection time: it is determined by pocket size and
    # directly affects whether the run can finish. The old version discovered infeasible targets
    # only after selecting them, wasting machine time.
    vol = {}
    for up, _pr, _y in cand:
        c = np.asarray(pockets[up]["pocket_coordinates"], dtype=float)
        sz = (c.max(0) + PAD) - (c.min(0) - PAD)
        vol[up] = float(sz[0] * sz[1] * sz[2])
    if args.report_boxes:
        for up in sorted(vol, key=vol.get):
            print("  %-10s %9.0f Å³" % (up, vol[up]))
        v = sorted(vol.values())
        print(f"\n候选 {len(v)} 个：中位 {v[len(v)//2]:.0f}，"
              f"四分位 {v[len(v)//4]:.0f}–{v[3*len(v)//4]:.0f}，"
              f"范围 {v[0]:.0f}–{v[-1]:.0f}")
        for t in (25000, 30000, 35000):
            print(f"  ≤{t:,} Å³ 的有 {sum(1 for x in v if x <= t)} 个")
        return
    if args.max_box > 0:
        n0 = len(cand)
        cand = [x for x in cand if vol[x[0]] <= args.max_box]
        print(f"盒子体积 ≤{args.max_box:,.0f} Å³：{len(cand)}/{n0} 个候选通过")

    picked, manifest = cand[:args.targets], []
    print(f"合格靶点 {len(cand)}，选中 {len(picked)}"
          + (f"（seed={args.seed} 随机抽）" if keep is not None else "（字母序前 N）"),
          flush=True)

    os.makedirs(args.out, exist_ok=True)
    n_lig = 0
    for up, pr, y in picked:
        d = f"{args.out}/{up}"
        os.makedirs(d, exist_ok=True)
        coords = write_pocket_pdb(pockets[up], f"{d}/pocket.pdb")
        lo, hi = coords.min(0) - PAD, coords.max(0) + PAD
        center, size = (lo + hi) / 2, hi - lo
        # receptor pdbqt
        subprocess.run([f"{DOCK}/obabel", f"{d}/pocket.pdb", "-O", f"{d}/pocket.pdbqt",
                        "-xr", "-p", "7.4"], capture_output=True)
        # ligands: take the top-N by retrieval score, keeping order so they can be compared against retrieval's own order
        rec = ev[up]
        # warning: the molecule order must be hard-verified, matching length alone is not enough: the model
        # reads the lmdb, whose cursor order is lexicographic (0, 1, 10, 100, ...), which differs from the
        # jsonl order while having the same length, causing a silent mismatch.
        # This trap has bitten this project three times. Verification checks that "the positions labelled 1
        # really are that target's actives" -- both orderings are tried, and it is skipped if neither passes.
        act = {m["smiles"] for m in rec["actives"]}

        def ok(seq):
            if seq is None or len(seq) != len(pr):
                return None
            got = {seq[i] for i in range(len(seq)) if y[i] == 1}
            return seq if got == act else None

        smis = ok([m["smiles"] for m in rec["actives"]]
                  + [m["smiles"] for m in rec["decoys"]])
        if smis is None:
            try:
                e = lmdb.open(f"{B}/data/T3_6A/{args.layer}/{up}/{up}_lig.lmdb",
                              subdir=False, readonly=True, lock=False)
                cur = []
                with e.begin() as t:
                    for _k, v in t.cursor():
                        cur.append(pickle.loads(v)["smi"])
                e.close()
                smis = ok(cur)
            except Exception:
                smis = None
        if smis is None:
            print(f"  {up} 分子顺序校验不过，跳过")
            continue
        top = np.argsort(-pr)[:args.topn]
        rows = [{"idx": int(i), "rank": r, "smiles": smis[i], "label": int(y[i]),
                 "retrieval_score": float(pr[i])} for r, i in enumerate(top)]
        with open(f"{d}/ligands.smi", "w") as f:
            for j, x in enumerate(rows):
                f.write(f"{x['smiles']}\tlig{j}\n")
        json.dump({"uniprot": up, "center": center.tolist(), "size": size.tolist(),
                   "ligands": rows}, open(f"{d}/manifest.json", "w"))
        manifest.append(up)
        n_lig += len(rows)

    json.dump({"layer": args.layer, "model": args.model, "topn": args.topn,
               "targets": manifest}, open(f"{args.out}/manifest.json", "w"), indent=1)
    print(f"准备完成：{len(manifest)} 个靶点，{n_lig:,} 个配体待对接")
    print(f"盒子取自 6Å 口袋的包围盒 + {PAD}Å 余量——与检索模型同一个口袋")


if __name__ == "__main__":
    main()
