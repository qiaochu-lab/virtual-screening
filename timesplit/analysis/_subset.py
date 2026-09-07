"""共享的靶点子集过滤器。

导师 2026-09-04 定了新口径（活性 ≥50、类别构成对齐 VSDS-vd），
产出一个 242 条（222 个唯一靶点）的子集。所有下游分析都要能切到这个子集上，
而每个靶点的指标本来就是在它自己的候选池里算的、靶点之间互不影响，
所以「换子集」只是换一批数求平均，不需要重跑任何模型。

用法：
    from _subset import load_subset
    keep = load_subset(args.subset)          # None 表示不过滤
    if keep and (L, up) not in keep: continue
"""
import csv
import os


def add_subset_arg(ap, default=None):
    ap.add_argument("--subset", default=default,
                    help="靶点子集 CSV（需含 uniprot/layer 列）；不给则用全量")


def load_subset(path):
    """返回 {(layer, uniprot)} 或 None。"""
    if not path:
        return None
    if not os.path.exists(path):
        raise SystemExit(f"子集文件不存在: {path}")
    keep = {(r["layer"], r["uniprot"]) for r in csv.DictReader(open(path))}
    print(f"子集过滤：{len(keep)} 条（{len({u for _, u in keep})} 个唯一靶点） "
          f"← {os.path.basename(path)}")
    return keep
