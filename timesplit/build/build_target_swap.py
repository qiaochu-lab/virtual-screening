"""构造 target-swap 评测树：候选分子池不变，只把靶点身份换成另一个。

要回答的问题
------------
纯配体基线（不看蛋白就把 EF 打到理论上限的 98.7%）很容易被读成
「模型根本不看蛋白」。我们支持不了这个强结论。Target swap 直接测：
把靶点换掉、候选池一个分子不动，成绩掉多少。

能支撑的结论是「模型确实使用了靶点信息，但该信号不足以支撑真正的化学外推」——
比「模型只看配体」精确，也更难反驳。

为什么要连目录名一起换
----------------------
评测代码这样取序列：

    s = (_seqs.get(target) or {}).get("seq")     # target 就是目录名
    ...
    pocket_forward(protein_sequences=s, **pocket_from_lmdb)

序列**按目录名查**，不在 lmdb 里。所以只替换 `_pocket.lmdb` 会造出
「A 的序列 + B 的口袋」的嵌合体——掉分了也分不清是靶点错了还是输入自相矛盾。

正确做法是把目录命名成替身靶点 T'：

    swap_root/{层}/{T'}/{T'}_pocket.lmdb  ->  T' 自己的口袋
    swap_root/{层}/{T'}/{T'}_lig.lmdb     ->  **T 的**配体池（软链，改名）

模型于是看到完整的 T' 身份（序列 + 口袋）配 T 的候选池。标签跟着配体走，
所以标签仍然是 T 的正确标签。

三个受控的混杂
--------------
1. 口袋大小：替身的 pocket_atoms 与原靶点相差不超过 --size-tol，
   否则测到的可能只是尺寸效应。
2. 配对方式：用错排（derangement），保证没有靶点被换成自己，
   且目录名互不冲突。
3. 抽样方差：--rounds 指定重复几轮，每轮一个独立错排。
"""
import argparse
import json
import os
import random


def load_manifest(root):
    """{层: {靶点: 口袋原子数}}"""
    man = json.load(open(f"{root}/manifest.json"))
    out = {}
    for L, d in man.items():
        pt = d.get("per_target") or {}
        out[L] = {up: (info.get("pocket_atoms") or 0) for up, info in pt.items()}
    return out


def derange(targets, atoms, tol, rng, tries=4000):
    """求一个错排：每个靶点换到另一个靶点，且口袋大小接近。

    贪心 + 重试。先给候选少的靶点分配（最受限优先），否则最后几个会无解。
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
                # 目录名 = 替身 sub，模型据此取到 sub 的序列
                d = f"{out}/{sub}"
                os.makedirs(d, exist_ok=True)
                # 口袋：替身自己的
                for suf in ("_pocket.lmdb", "_pocket.lmdb-lock"):
                    s_, t_ = f"{src}/{sub}/{sub}{suf}", f"{d}/{sub}{suf}"
                    if os.path.exists(s_) and not os.path.exists(t_):
                        os.symlink(s_, t_)
                # 配体：原靶点的，但改名成替身的名字
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
