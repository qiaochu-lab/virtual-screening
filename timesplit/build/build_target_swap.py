"""Build the target-swap evaluation tree: the candidate molecule pool stays fixed, only the target identity is swapped for another one.

The question this answers
--------------------------
A ligand-only baseline that reaches 98.7% of the theoretical EF ceiling
without looking at the protein at all is easy to (mis)read as "the model
doesn't look at the protein at all." We can't support that strong a
claim. Target swap tests this directly: swap the target while leaving
every molecule in the candidate pool untouched, and see how much the
score drops.

The conclusion this can support is "the model does use target
information, but that signal isn't enough to support genuine chemical
extrapolation" -- more precise than "the model only looks at the ligand",
and harder to argue against.

Why the directory name has to be swapped too
-----------------------------------------------
The evaluation code fetches the sequence like this:

    s = (_seqs.get(target) or {}).get("seq")     # target is the directory name
    ...
    pocket_forward(protein_sequences=s, **pocket_from_lmdb)

The sequence is **looked up by directory name**, not stored in the lmdb.
So swapping only `_pocket.lmdb` would create a chimera of "A's sequence +
B's pocket" -- if the score drops, there's no way to tell whether that's
because the target changed or because the input is self-contradictory.

The correct approach is to name the directory after the swapped-in
target T':

    swap_root/{layer}/{T'}/{T'}_pocket.lmdb  ->  T''s own pocket
    swap_root/{layer}/{T'}/{T'}_lig.lmdb     ->  **T's** ligand pool (symlinked, renamed)

The model then sees a complete T' identity (sequence + pocket) paired
with T's candidate pool. The label follows the ligand, so the label is
still the correct label for T.

Three controlled confounds
---------------------------
1. Pocket size: the swapped-in target's pocket_atoms must differ from the
   original target's by no more than --size-tol, otherwise what's being
   measured could just be a size effect.
2. Pairing scheme: uses a derangement, guaranteeing no target is swapped
   with itself, and directory names never collide.
3. Sampling variance: --rounds sets how many repeats to run, each an
   independent derangement.
"""
import argparse
import json
import os
import random


def load_manifest(root):
    """{layer: {target: pocket atom count}}"""
    man = json.load(open(f"{root}/manifest.json"))
    out = {}
    for L, d in man.items():
        pt = d.get("per_target") or {}
        out[L] = {up: (info.get("pocket_atoms") or 0) for up, info in pt.items()}
    return out


def derange(targets, atoms, tol, rng, tries=4000):
    """Find a derangement: swap each target with another target whose pocket is close in size.

    Greedy + retry. Assign the most-constrained targets (fewest candidates) first,
    otherwise the last few can end up with no solution.
    """
    ok = {t: [u for u in targets
              if u != t and atoms.get(t) and atoms.get(u)
              and abs(atoms[u] - atoms[t]) <= tol * atoms[t]]
          for t in targets}
    unmatched = [t for t in targets if not ok[t]]
    pool = [t for t in targets if ok[t]]
    for _ in range(tries):
        used, m = set(), {}
        for t in sorted(pool, key=lambda x: len(ok[x])):
            cand = [u for u in ok[t] if u not in used]
            if not cand:
                break
            pick = rng.choice(cand)
            m[t] = pick
            used.add(pick)
        if len(m) == len(pool):
            return m, unmatched
    return None, unmatched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t3-root", default="/data/work/vs-benchmark/data/T3_6A")
    ap.add_argument("--out-root", required=True, help="生成的 swap 树放哪")
    ap.add_argument("--layers", nargs="+", default=["L4"])
    ap.add_argument("--subset", default=None,
                    help="靶点清单 CSV（uniprot/layer 列）；不给则用该层全部")
    ap.add_argument("--limit", type=int, default=0, help="每层最多取几个靶点（0=不限）")
    ap.add_argument("--size-tol", type=float, default=0.30,
                    help="替身口袋原子数允许的相对偏差")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    atoms_by_layer = load_manifest(args.t3_root)
    keep = None
    if args.subset:
        import csv
        keep = {(r["layer"], r["uniprot"]) for r in csv.DictReader(open(args.subset))}

    rng = random.Random(args.seed)
    manifest = {}
    for rnd in range(1, args.rounds + 1):
        for L in args.layers:
            src = f"{args.t3_root}/{L}"
            if not os.path.isdir(src):
                print(f"⚠️ 没有 {src}，跳过")
                continue
            targets = sorted(os.listdir(src))
            if keep is not None:
                targets = [t for t in targets if (L, t) in keep]
            atoms = atoms_by_layer.get(L, {})
            targets = [t for t in targets if atoms.get(t)]
            if args.limit:
                targets = targets[:args.limit]

            m, unmatched = derange(targets, atoms, args.size_tol, rng)
            if m is None:
                print(f"❌ {L} 第 {rnd} 轮求不出错排，放宽 --size-tol 再试")
                continue
            if unmatched:
                print(f"⚠️ {L}: {len(unmatched)} 个靶点在 ±{args.size_tol:.0%} 内"
                      f"找不到尺寸相近的替身，已排除")

            out = f"{args.out_root}/round{rnd}/{L}"
            n = 0
            for orig, sub in m.items():
                # directory name = swapped-in target sub; the model looks up sub's sequence from it
                d = f"{out}/{sub}"
                os.makedirs(d, exist_ok=True)
                # pocket: the swapped-in target's own
                for suf in ("_pocket.lmdb", "_pocket.lmdb-lock"):
                    s_, t_ = f"{src}/{sub}/{sub}{suf}", f"{d}/{sub}{suf}"
                    if os.path.exists(s_) and not os.path.exists(t_):
                        os.symlink(s_, t_)
                # ligands: the original target's, renamed to the swapped-in target's name
                for suf in ("_lig.lmdb", "_lig.lmdb-lock"):
                    s_, t_ = f"{src}/{orig}/{orig}{suf}", f"{d}/{sub}{suf}"
                    if os.path.exists(s_) and not os.path.exists(t_):
                        os.symlink(s_, t_)
                n += 1
            manifest[f"round{rnd}/{L}"] = {
                "pairs": [{"ligand_pool_from": o, "identity_from": s,
                           "atoms_orig": atoms[o], "atoms_sub": atoms[s]}
                          for o, s in sorted(m.items())],
                "excluded_no_size_match": unmatched,
            }
            sz = [abs(atoms[s] - atoms[o]) / atoms[o] for o, s in m.items()]
            print(f"round{rnd}/{L}: {n} 对，口袋大小偏差 中位 "
                  f"{100*sorted(sz)[len(sz)//2]:.0f}% 最大 {100*max(sz):.0f}%")

    os.makedirs(args.out_root, exist_ok=True)
    with open(f"{args.out_root}/swap_manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"\n映射写入 {args.out_root}/swap_manifest.json")
    print("跑评测时把 --t3-root 指到 round<N> 目录，结果目录名是**替身靶点**，"
          "回读时用 swap_manifest.json 映射回原靶点。")


if __name__ == "__main__":
    main()
