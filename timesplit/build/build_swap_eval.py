"""Build the target-swap evaluation set for the sequence-based models (ConPLex / ConGLUDe / SPRINT).

For pocket-based models, the swap is done by swapping the directory tree
that `_pocket.lmdb` lives in (build_target_swap.py). Sequence-based
models don't read a pocket -- they look up the sequence/structure from
the `uniprot` field in the evaluation jsonl, so for them "swapping the
target" means swapping the uniprot in the record while leaving the
ligand pool untouched.

Both sides **share the same swap_manifest.json**, so the pairing is
exactly consistent and results across models can be placed directly in
one table for comparison.

The uniprot in the output records is the **swapped identity** (matching
the pocket tree's directory naming); reading it back likewise uses the
manifest to map back to the original target.
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
            # original target -> swapped identity
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
                    r["uniprot"] = subst          # identity replaced with the swapped identity
                    r["swap_ligand_pool_from"] = orig   # keep a trace so reading it back doesn't need the manifest again
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n += 1
            print(f"round{rnd}/{L}: 写出 {n} 条（配体池不变，身份换成替身）")
    print(f"\n写入 {args.out_root}")


if __name__ == "__main__":
    main()
