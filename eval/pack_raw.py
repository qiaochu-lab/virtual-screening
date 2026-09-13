"""Pack the raw per-molecule scores so anyone can recompute every metric here.

One .npz per model per task. Keys are paths, always task-prefixed:

    T3_hypseek_official_vs.npz
      T3/L1/C7C422/preds     float32, one score per molecule
      T3/L1/C7C422/labels    int8, 1 = active

Scores are cast to float32 (originals are float64; every metric in this
repository is identical to four decimals either way), labels to int8.

⚠️ Two source roots, not one
----------------------------
T3 per-target scores live under **two** directories, and which one a model is
in is an accident of when it was run:

    results/t3_raw/<model>/T3/<layer>/<uniprot>/saved_preds.npy
    results/t3/<model>/<layer>/<uniprot>/saved_preds.npy      (note: no T3/)

The analysis scripts already read both with a fallback. An earlier version of
this script walked only `t3_raw`, so `conglude` and `conplex` — which exist
**only** under `results/t3` — were silently absent from the archive. Two of the
ten published models, omitted with no error and no missing-file message,
because the loop simply never saw a directory to skip. The key prefix is
normalised here so both roots produce the documented `T3/<layer>/<uniprot>` form.

⚠️ Missing models are an error, not a skip
------------------------------------------
`--expect` names the models that back published tables. Anything on that list
that cannot be found stops the run. Silence is how the previous omission
survived; an archive that is quietly incomplete is worse than no archive.
"""
import argparse
import hashlib
import json
import os

import numpy as np

B = "/data/work/vs-benchmark"

# Models backing the published tables. T3_main.csv carries both HypSeek weights
# (`_vs` is the benchmark row, `_rk` is kept for the analyses that predate the
# 2026-09-12 switch), so both are packed.
EXPECT_T3 = ["drugclip", "bindclip_randneg", "bindclip_hardneg",
             "ligunity_pocket_ranking", "ligunity_protein_ranking", "litenclip",
             "hypseek_official_vs", "hypseek_rk", "conglude", "conplex", "sprint"]

# task -> [(root, needs_task_prefix), ...]; earlier roots win when a model is in both
ROOTS = {
    "T3": [(f"{B}/results/t3_raw", False), (f"{B}/results/t3", True)],
    "T1": [(f"{B}/results/t1_raw", False), (f"{B}/results/t1", True)],
}


def collect(root, model, add_prefix, task):
    """Read one model's per-target arrays. Returns {key: array}, target count."""
    mroot = os.path.join(root, model)
    if not os.path.isdir(mroot) or os.path.islink(mroot):
        return {}, 0
    blobs, n = {}, 0
    for dirpath, _dirs, files in os.walk(mroot):
        if "saved_preds.npy" not in files or "saved_labels.npy" not in files:
            continue
        rel = os.path.relpath(dirpath, mroot).replace(os.sep, "/")
        if add_prefix:                       # results/t3/<m>/<layer>/... lacks it
            rel = f"{task}/{rel}"
        try:
            p = np.load(os.path.join(dirpath, "saved_preds.npy")).astype(np.float32)
            y = np.load(os.path.join(dirpath, "saved_labels.npy")).astype(np.int8)
        except Exception as e:
            print(f"  skip {model}/{rel}: {e}")
            continue
        if p.shape[-1] != y.shape[-1]:
            print(f"  skip {model}/{rel}: length mismatch {p.shape} vs {y.shape}")
            continue
        blobs[f"{rel}/preds"], blobs[f"{rel}/labels"] = p, y
        n += 1
    return blobs, n


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?", default=f"{B}/results/raw_release")
    ap.add_argument("--task", default="T3", choices=["T3", "T1"])
    ap.add_argument("--all", action="store_true",
                    help="pack every model found, not just the expected list "
                         "(adds the swap rounds and radius variants)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    roots = ROOTS[args.task]
    found = {}                                  # model -> (root, add_prefix)
    for root, pref in roots:
        if not os.path.isdir(root):
            print(f"  root absent, skipped: {root}")
            continue
        for m in sorted(os.listdir(root)):
            if m not in found and os.path.isdir(os.path.join(root, m)):
                found[m] = (root, pref)

    wanted = sorted(found) if args.all else EXPECT_T3 if args.task == "T3" else sorted(found)
    missing = [m for m in wanted if m not in found]
    if missing:
        raise SystemExit(
            f"⛔ expected models not found under {[r for r, _ in roots]}: {missing}\n"
            "   Refusing to write a quietly incomplete archive.")

    manifest = {}
    for m in wanted:
        root, pref = found[m]
        blobs, n = collect(root, m, pref, args.task)
        if not blobs:
            raise SystemExit(f"⛔ {m}: found the directory but no usable arrays in {root}")
        fn = os.path.join(args.out_dir, f"{args.task}_{m}.npz")
        np.savez_compressed(fn, **blobs)
        mb = os.path.getsize(fn) / 1e6
        manifest[f"{args.task}_{m}"] = {
            "targets": n, "file": os.path.basename(fn),
            "size_mb": round(mb, 2), "source_root": os.path.basename(root),
            "sha256": sha256(fn)}
        print(f"{args.task:3} {m:28} {n:5} targets  {mb:7.2f} MB  ({os.path.basename(root)})")

    mpath = os.path.join(args.out_dir, f"manifest_{args.task}.json")
    json.dump(manifest, open(mpath, "w"), indent=1)
    tot = sum(v["size_mb"] for v in manifest.values())
    print(f"\n{len(manifest)} packages, {tot:.1f} MB -> {args.out_dir}")
    print(f"manifest with sha256 -> {mpath}")


if __name__ == "__main__":
    main()
