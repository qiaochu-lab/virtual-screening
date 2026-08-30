"""按官方口径平均第 41–50 轮的权重。

为什么必须做这一步
------------------
官方发布的权重文件名就是 `checkpoint_avg_41-50_rk.pt`——**它是后十轮的平均**，
而我们评测用的是单个 best checkpoint。权重平均（SWA 类做法）通常带来可观提升，
不消掉这个差异就把「我们的 _vs 43.3 vs 官方 _rk 56.4」当成
「虚筛目标不如排序目标」，是把两个变量混在一起。

平均只对浮点张量做；整型/布尔（如 num_updates 之类的计数）取第一个，
否则会得到无意义的小数。
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
