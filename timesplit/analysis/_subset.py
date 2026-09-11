"""Shared target-subset filter.

The advisor set a new criterion on 2026-09-04 (≥50 actives, class composition
matched to VSDS-vd); the final quota was set at 350, producing a subset of
**328 entries (293 unique targets)** (an earlier 250-quota version had 242
entries / 222 targets and is no longer used). Every downstream analysis needs
to be able to slice down to this subset, and since each target's metric is
computed within its own candidate pool independently of other targets,
"switching subsets" is just averaging over a different set of numbers — no
model needs to be re-run.

Usage:
    from _subset import load_subset
    keep = load_subset(args.subset)          # None means no filtering
    if keep and (L, up) not in keep: continue
"""
import csv
import os


def add_subset_arg(ap, default=None):
    ap.add_argument("--subset", default=default,
                    help="靶点子集 CSV（需含 uniprot/layer 列）；不给则用全量")


def load_subset(path):
    """Returns {(layer, uniprot)} or None."""
    if not path:
        return None
    if not os.path.exists(path):
        raise SystemExit(f"子集文件不存在: {path}")
    keep = {(r["layer"], r["uniprot"]) for r in csv.DictReader(open(path))}
    print(f"子集过滤：{len(keep)} 条（{len({u for _, u in keep})} 个唯一靶点） "
          f"← {os.path.basename(path)}")
    return keep
