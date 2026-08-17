"""为 T3 靶点生成 SaProt 结构感知序列（SPRINT 的蛋白塔要）。

SaProt 序列把每个残基编码成「氨基酸 + foldseek 3Di 结构 token」两个字符，
例如 MdEvKp...。SPRINT 发布的 checkpoint 就是这么训的；
不给结构 token 它会用 mask 顶替，但那样等于废掉一半输入，
所以我们用已有的结构真算一遍。

结构来源与 run_t3_conglude.py 完全一致（PDB 实验结构优先、Boltz-2 预测补位），
这样 SPRINT 与 ConGLUDe 的结构条件相同，两者的差异只来自模型本身。

pLDDT 掩码
----------
SaProt 的惯例是对低置信区域（pLDDT < 70）把结构 token 掩掉。
但 pLDDT 只有预测结构才有：
  - Boltz-2 预测结构 → B-factor 列是 pLDDT，开掩码
  - PDB 实验结构     → B-factor 列是真实 B 因子，**必须关掉掩码**，
                        否则会把 B 因子当 pLDDT 误掩一大片
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

B = "/data/yicheng/xqc/vs-benchmark"
FOLDSEEK = f"{B}/tools/foldseek/bin/foldseek"
sys.path.insert(0, f"{B}/code/panspecies-dti")
# get_struc_seq 在 utils/structure_to_saprot.py 里，不是包内模块，得单独加路径
sys.path.insert(0, f"{B}/code/panspecies-dti/utils")


def build_boltz_index():
    idx = {}
    for d in ["boltz_batch_out", "boltz_retry_out", "boltz_gap_out", "boltz_r2_out"]:
        p = f"{B}/{d}"
        if not os.path.isdir(p):
            continue
        for root, _, files in os.walk(p):
            for fn in files:
                if fn.endswith("_model_0.pdb"):
                    idx.setdefault(fn.replace("_model_0.pdb", ""), os.path.join(root, fn))
    return idx


def one(task):
    up, path, is_pred, pid = task
    from structure_to_saprot import get_struc_seq
    try:
        d = get_struc_seq(FOLDSEEK, path, chains=None,
                          process_id=pid, plddt_mask=is_pred)
        if not d:
            return up, None, "foldseek 无输出"
        # get_struc_seq 返回 {chain: (aa_seq, struc_seq, combined)}；取最长的链
        best = max(d.values(), key=lambda v: len(v[0]))
        combined = best[2]
        if not combined:
            return up, None, "结构序列为空"
        return up, {"saprot": combined, "len": len(best[0]),
                    "source": "boltz2_pred" if is_pred else "pdb_holo"}, None
    except Exception as e:                      # noqa: BLE001 逐条容错
        return up, None, f"{type(e).__name__}: {e}"[:120]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--pdb_dir", default=f"{B}/data/t3/pdb_for_struct")
    ap.add_argument("--out", default=f"{B}/data/t3/saprot_seqs.json")
    args = ap.parse_args()

    # 复用 ConGLUDe 那一轮已经落地的 PDB 文件（实验结构 + 预测结构都在里面）
    cands = {}
    for L in ["L1", "L2", "L3", "L4"]:
        d = f"{B}/code/conglude/data/datasets/predict_datasets/t3_{L}/raw/pdb_files"
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.endswith(".pdb"):
                cands.setdefault(fn[:-4], os.path.join(d, fn))

    boltz = build_boltz_index()
    for up, p in boltz.items():
        cands.setdefault(up, p)

    # 判定来源：文件在 boltz 输出目录下即为预测结构
    tasks = []
    for i, (up, p) in enumerate(sorted(cands.items())):
        is_pred = "/boltz_" in p or (up in boltz and boltz[up] == p)
        tasks.append((up, p, is_pred, i % max(1, args.procs)))
    print(f"待处理结构: {len(tasks):,}", flush=True)

    from collections import Counter
    out, fails = {}, Counter()
    with Pool(args.procs) as pool:
        for i, (up, rec, err) in enumerate(pool.imap_unordered(one, tasks, chunksize=8)):
            if rec is None:
                fails[err.split(":")[0]] += 1
            else:
                out[up] = rec
            if (i + 1) % 200 == 0:
                print(f"  {i+1:,}/{len(tasks):,}  成功 {len(out):,}", flush=True)

    json.dump(out, open(args.out, "w"))
    print(f"\n成功 {len(out):,} / {len(tasks):,}")
    print(f"  {dict(Counter(v['source'] for v in out.values()))}")
    for k, v in fails.most_common(6):
        print(f"  失败 {k}: {v}")
    print(f"已写入 {args.out}")


if __name__ == "__main__":
    main()
