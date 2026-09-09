"""给序列类模型（ConPLex / ConGLUDe / SPRINT）建 target-swap 的评测集。

口袋类模型的 swap 是换 `_pocket.lmdb` 所在的目录树（build_target_swap.py）。
序列类模型不读口袋——它们从评测集 jsonl 的 `uniprot` 字段去查序列/结构，
所以对它们来说，「换靶点」= 换记录里的 uniprot，配体池原样保留。

两边**共用同一份 swap_manifest.json**，所以配对关系完全一致，
跨模型的结果可以直接放在一张表里比。

输出记录的 uniprot 是**替身**（与口袋树的目录命名一致），
回读时同样用 manifest 映射回原靶点。
"""
import argparse, json, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="/data/work/vs-benchmark/data/t3/eval")
    ap.add_argument("--manifest", required=True, help="build_target_swap.py 产出的 swap_manifest.json")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--layers", nargs="+", default=["L1", "L4"])
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    for rnd in range(1, args.rounds + 1):
        for L in args.layers:
            key = f"round{rnd}/{L}"
            if key not in man:
                print(f"⚠️ manifest 里没有 {key}，跳过")
                continue
            pairs = man[key]["pairs"]
            # 原靶点 -> 替身
            sub = {p["ligand_pool_from"]: p["identity_from"] for p in pairs}

            recs = {json.loads(l)["uniprot"]: json.loads(l)
                    for l in open(f"{args.eval_dir}/{L}.jsonl")}
            out_dir = f"{args.out_root}/round{rnd}"
            os.makedirs(out_dir, exist_ok=True)
            n = 0
            with open(f"{out_dir}/{L}.jsonl", "w") as f:
                for orig, subst in sub.items():
                    r = recs.get(orig)
                    if r is None:
                        continue
                    r = dict(r)
                    r["uniprot"] = subst          # 身份换成替身
                    r["swap_ligand_pool_from"] = orig   # 留痕，回读时不必再查 manifest
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n += 1
            print(f"round{rnd}/{L}: 写出 {n} 条（配体池不变，身份换成替身）")
    print(f"\n写入 {args.out_root}")


if __name__ == "__main__":
    main()
