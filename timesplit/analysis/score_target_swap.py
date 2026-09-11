"""Target-swap results, aggregated: how much discriminative power the model
retains after swapping the target identity.

Pairing
------
The swap result's directory name is the **substitute target T'**, because
the model's protein sequence is looked up by directory name (the comment in
build_target_swap.py explains why it has to be wired this way). So the
pairing is:

    correct: results/t3_raw/{model}/T3/{layer}/{T}       T's pocket + T's ligand pool
    swap:    results/t3_raw/swap_{model}_r{N}/T3/{layer}/{T'}   T''s pocket + T's ligand pool

Both sides are scored on the same batch of molecules; the only variable is
target identity.

⚠️ Labels must be verified to match
------------------
During a trial run, I looked up the swap directory by T and got back "T's
pocket + someone else's ligands" — the molecule counts were completely
different on the two sides and it went unnoticed, nearly reporting an
invalid comparison as a result. So every pair here has its label arrays
compared, and a mismatch is dropped and reported.

Multiple rounds
----
Random swap draws several substitutes per target (round1/2/3); the
per-target median across rounds is taken before the paired test, to wash
out the luck of any single draw.
"""
import argparse, csv, json, os, sys
import numpy as np

B = "/data/work/vs-benchmark"
LAYERS = ["L1", "L4"]
METRICS = ["ef1", "bedroc", "auroc"]


def find_dir(base_candidates, layer, target):
    """The results directory layout isn't uniform; try each candidate in turn.

    Pocket-family models (UniMol / LigUnity / HypSeek) write
    results/t3_raw/{m}/T3/{layer}/{target}; sequence-family models
    (ConPLex / ConGLUDe / SPRINT) are missing the T3 level, and their
    baseline lives under results/t3/. Rather than scattering ifs across
    every call site, list every candidate path here once.
    """
    for b in base_candidates:
        for mid in (f"/T3/{layer}/{target}", f"/{layer}/{target}"):
            d = b + mid
            if os.path.isdir(d):
                return d
    return None


def load(d):
    try:
        p = np.load(f"{d}/saved_preds.npy").reshape(-1)
        y = np.load(f"{d}/saved_labels.npy").reshape(-1)
    except Exception:
        return None, None
    if len(p) != len(y) or y.sum() == 0 or y.sum() == len(y):
        return None, None
    return p, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--manifest", default=f"{B}/data/T3_swap_full/swap_manifest.json")
    ap.add_argument("--prefix", default="swap",
                    help="结果目录前缀。随机 swap 是 swap_{模型}_r{N}，"
                         "同家族 swap 是 swapfam_{模型}_r{N}")
    ap.add_argument("--raw", default=f"{B}/results/t3_raw")
    ap.add_argument("--metrics-dir", default=f"{B}/eval")
    ap.add_argument("--out", default=f"{B}/results/export/T3_target_swap.csv")
    args = ap.parse_args()

    sys.path.insert(0, args.metrics_dir)
    from metrics import enrichment_factor, bedroc, roc_auc
    from scipy import stats

    man = json.load(open(args.manifest))
    rows = [["model", "layer", "n_targets", "metric",
             "correct_mean", "swap_mean", "delta", "delta_pct", "p_wilcoxon"]]

    print("Target swap：候选配体池不变，只换靶点身份")
    print("=" * 96)
    for m in args.models:
        printed = False
        for L in LAYERS:
            # Collect this round's swap metrics, per target
            per = {}          # original target -> {"correct": (...), "swap": [per round]}
            skipped = 0
            for rnd in range(1, args.rounds + 1):
                key = f"round{rnd}/{L}"
                if key not in man:
                    continue
                for pr in man[key]["pairs"]:
                    o, s = pr["ligand_pool_from"], pr["identity_from"]
                    base = os.path.dirname(args.raw)   # .../results
                    dc = find_dir([f"{args.raw}/{m}", f"{base}/t3/{m}"], L, o)
                    ds = find_dir([f"{args.raw}/{args.prefix}_{m}_r{rnd}"], L, s)
                    if dc is None or ds is None:
                        continue
                    pc, yc = load(dc)
                    ps, ys = load(ds)
                    if pc is None or ps is None:
                        continue
                    if len(yc) != len(ys) or not np.array_equal(yc, ys):
                        skipped += 1
                        continue
                    e = per.setdefault(o, {"correct": None, "swap": []})
                    if e["correct"] is None:
                        e["correct"] = (enrichment_factor(pc, yc, 0.01),
                                        bedroc(pc, yc, 80.5), roc_auc(pc, yc))
                    e["swap"].append((enrichment_factor(ps, ys, 0.01),
                                      bedroc(ps, ys, 80.5), roc_auc(ps, ys)))
            per = {k: v for k, v in per.items() if v["correct"] and v["swap"]}
            if len(per) < 5:
                continue
            if not printed:
                print(f"\n{m}")
                print("-" * 96)
                printed = True

            A = np.array([v["correct"] for v in per.values()])
            Sw = np.array([np.median(v["swap"], axis=0) for v in per.values()])
            nr = np.median([len(v["swap"]) for v in per.values()])
            note = f"  ⚠️ {skipped} 对标签不一致已剔除" if skipped else ""
            print(f"  {L}  n={len(per)}  每靶点 {nr:.0f} 轮{note}")
            for i, nm in enumerate(METRICS):
                d = Sw[:, i] - A[:, i]
                try:
                    p = stats.wilcoxon(A[:, i], Sw[:, i]).pvalue
                except Exception:
                    p = float("nan")
                pct = 100 * d.mean() / A[:, i].mean() if A[:, i].mean() else float("nan")
                star = " *" if p < 0.05 else ""
                print("      %-8s 正确 %8.3f → swap %8.3f   Δ %+8.3f (%+6.1f%%)  p=%.4f%s"
                      % (nm, A[:, i].mean(), Sw[:, i].mean(), d.mean(), pct, p, star))
                rows.append([m, L, len(per), nm,
                             f"{A[:, i].mean():.4f}", f"{Sw[:, i].mean():.4f}",
                             f"{d.mean():.4f}", f"{pct:.1f}", f"{p:.4g}"])
            # Look separately at targets with signal: the ones where EF>0 under the correct pocket
            has = A[:, 0] > 0
            if has.sum() >= 3:
                lost = ((Sw[has, 0] == 0)).sum()
                print("      有富集的靶点 %d 个，swap 后掉到 0 的 %d 个；"
                      "AUROC %.3f → %.3f"
                      % (has.sum(), lost, A[has, 2].mean(), Sw[has, 2].mean()))
    print("\n" + "=" * 96)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"写入 {args.out}")


if __name__ == "__main__":
    main()
