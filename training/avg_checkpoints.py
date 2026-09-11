"""Average the weights of epochs 41-50, matching the official convention.

Why this is necessary
------------------
The officially released weight file is literally named
`checkpoint_avg_41-50_rk.pt` — **it is an average of the last ten epochs**,
whereas our evaluation used a single best checkpoint. Weight averaging
(an SWA-style technique) typically brings a sizeable improvement on its own;
without removing that difference, reading "our _vs 43.3 vs the official _rk
56.4" as "the screening objective is worse than the ranking objective"
conflates two separate variables.

Averaging is only applied to floating-point tensors; integer/boolean entries
(such as counters like num_updates) keep the first checkpoint's value,
otherwise they'd end up as meaningless fractions.
"""
import argparse
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--start", type=int, default=41)
    ap.add_argument("--end", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = [f"{args.dir}/checkpoint{i}.pt" for i in range(args.start, args.end + 1)]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        raise SystemExit("没有可平均的 checkpoint")
    print(f"平均 {len(paths)} 个 checkpoint: {args.start}-{args.end}")

    base = torch.load(paths[0], map_location="cpu")
    avg = {k: v.clone().float() if v.is_floating_point() else v.clone()
           for k, v in base["model"].items()}
    n_float = sum(1 for v in base["model"].values() if v.is_floating_point())

    for p in paths[1:]:
        sd = torch.load(p, map_location="cpu")["model"]
        for k in avg:
            if avg[k].is_floating_point() and k in sd:
                avg[k] += sd[k].float()
    for k in avg:
        if avg[k].is_floating_point():
            avg[k] /= len(paths)
            avg[k] = avg[k].to(base["model"][k].dtype)

    base["model"] = avg
    torch.save(base, args.out)
    print(f"浮点张量 {n_float} 个已平均，其余按原值保留 -> {args.out}")


if __name__ == "__main__":
    main()
