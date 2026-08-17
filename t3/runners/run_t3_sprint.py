"""在 T3 上跑 SPRINT，输出统一评测层要的原始分数。

SPRINT 的工作方式
-----------------
不像其他模型那样一次前向出分数，而是分两步：
  1. `ultrafast-embed --moltype target/drug` 分别把蛋白和分子编码成向量
  2. 打分 = 两者的**余弦相似度**（见 ultrafast/compute_topk.py）

蛋白侧输入是 **SaProt 结构感知序列**（氨基酸 + foldseek 3Di token 交替），
由 gen_saprot_seqs.py 生成，1,460 个靶点全部成功。
结构来源与 ConGLUDe / 口袋类模型一致（PDB 实验结构优先、Boltz-2 预测补位），
保证横评时结构条件相同。

一次编码全层的唯一蛋白和唯一分子，再按靶点切分——
分子在靶点间高度重复，逐靶点编码会浪费几十倍算力。
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

B = "/data/yicheng/xqc/vs-benchmark"
SP = f"{B}/code/panspecies-dti"
EMBED = "/data/yicheng/xqc/envs/sprint/bin/ultrafast-embed"


def _embed_once(data_file, moltype, out_path, ckpt, gpu, batch=32):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
               HF_ENDPOINT="https://hf-mirror.com", HF_HOME=f"{B}/hf_cache")
    cmd = [EMBED, "--data-file", data_file, "--moltype", moltype,
           "--checkpoint", ckpt, "--output-path", out_path,
           "--batch-size", str(batch), "--num-workers", "0"]   # 0 个 worker：
           # 每个 DataLoader worker 都会打开一份 h5/lmdb，146k 条数据下句柄会耗尽
    p = subprocess.run(cmd, cwd=SP, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0 or not os.path.exists(out_path):
        print(p.stdout[-3000:], file=sys.stderr)
        raise SystemExit(f"embed {moltype} 失败 (returncode={p.returncode})")
    return np.load(out_path)


def run_embed(data_file, moltype, out_path, ckpt, gpu, batch=32, chunk=10**9):
    """分批 embed 后拼接。

    整批交给 ultrafast-embed 处理 14.6 万个分子会卡死——实测跑了 4 小时
    特征缓存一条新记录都没加，而同一路径下 L3 的 10.9 万个只用了 8 分钟。
    拆成小批能规避（每批独立进程，卡住也只影响一批）。
    """
    lines = open(data_file).read().rstrip("\n").split("\n")
    header, rows = lines[0], lines[1:]
    if len(rows) <= chunk:
        return _embed_once(data_file, moltype, out_path, ckpt, gpu, batch)

    parts = []
    for i in range(0, len(rows), chunk):
        sub = f"{data_file}.part{i//chunk}"
        with open(sub, "w") as f:
            f.write(header + "\n" + "\n".join(rows[i:i + chunk]) + "\n")
        op = f"{out_path}.part{i//chunk}.npy"
        print(f"    分批 {i//chunk + 1}/{(len(rows)+chunk-1)//chunk}"
              f"（{len(rows[i:i+chunk]):,} 条）", flush=True)
        parts.append(_embed_once(sub, moltype, op, ckpt, gpu, batch))
    out = np.concatenate(parts, axis=0)
    np.save(out_path, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="+", default=["L3", "L4", "L1", "L2"])
    ap.add_argument("--ckpt", default=f"{B}/ckpt/sprint/sprint.ckpt")
    ap.add_argument("--out_dir", default=f"{B}/results/t3_raw/sprint")
    ap.add_argument("--work", default=f"{B}/tmp/sprint_t3")
    ap.add_argument("--gpu", type=int, default=7)
    ap.add_argument("--limit", type=int, default=None,
                    help="每层只取前 N 个靶点。用于在小子集上验证接法是否正确——"
                         "整层 14.6 万分子会让 embed 耗尽文件句柄。")
    args = ap.parse_args()

    seqs = json.load(open(f"{B}/data/t3/saprot_seqs.json"))
    os.makedirs(args.work, exist_ok=True)
    print(f"SaProt 序列库: {len(seqs):,}", flush=True)

    for L in args.layers:
        p = f"{B}/data/t3/eval/{L}.jsonl"
        if not os.path.exists(p):
            print(f"[{L}] 评测集不存在，跳过", flush=True)
            continue
        recs = [json.loads(x) for x in open(p)]
        if args.limit:
            recs = recs[:args.limit]

        usable = [r for r in recs if r["uniprot"] in seqs]
        print(f"[{L}] 靶点 {len(recs)} → 有 SaProt 序列的 {len(usable)}", flush=True)
        if not usable:
            continue

        # 蛋白侧：每个靶点一条
        prots = [r["uniprot"] for r in usable]
        ptsv = f"{args.work}/{L}_target.tsv"
        # 必须写成两列并用 tab 分隔：SPRINT 用 pd.read_table(sep=None) 自动嗅探
        # 分隔符，而表头 "Target Sequence" 里带空格——单列时嗅探器会把空格当分隔符，
        # 列名被拆成 Target / Sequence，报 KeyError。多一列 tab 就能让它认准 tab。
        with open(ptsv, "w") as f:
            f.write("Target Sequence\tID\n")
            for u in prots:
                f.write(seqs[u]["saprot"] + "\t" + u + "\n")

        # 分子侧：全层去重（分子在靶点间高度重复）
        smi_idx, smiles = {}, []
        per_target = {}
        for r in usable:
            lab = {}
            for kind, y in (("actives", 1), ("decoys", 0)):
                for m in r[kind]:
                    s = m["smiles"]
                    if s not in smi_idx:
                        smi_idx[s] = len(smiles)
                        smiles.append(s)
                    lab[s] = y
            per_target[r["uniprot"]] = lab
        dtsv = f"{args.work}/{L}_drug.tsv"
        with open(dtsv, "w") as f:
            f.write("SMILES\tID\n")
            for i, sm in enumerate(smiles):
                f.write(sm + "\tm%d\n" % i)
        print(f"[{L}] 唯一分子 {len(smiles):,}", flush=True)

        pe = run_embed(ptsv, "target", f"{args.work}/{L}_target.npy", args.ckpt, args.gpu)
        de = run_embed(dtsv, "drug", f"{args.work}/{L}_drug.npy", args.ckpt, args.gpu)
        print(f"[{L}] 蛋白向量 {pe.shape}  分子向量 {de.shape}", flush=True)
        if pe.shape[0] != len(prots) or de.shape[0] != len(smiles):
            raise SystemExit(f"[{L}] 向量数与输入不符，拒绝继续")

        # 余弦相似度：先各自 L2 归一化，再内积
        pn = pe / (np.linalg.norm(pe, axis=1, keepdims=True) + 1e-9)
        dn = de / (np.linalg.norm(de, axis=1, keepdims=True) + 1e-9)

        n_ok = 0
        for row, up in enumerate(prots):
            lab = per_target[up]
            cols = [smi_idx[s] for s in lab]
            sc = dn[cols] @ pn[row]
            ys = np.array([lab[s] for s in lab], dtype=np.int8)
            if ys.sum() == 0 or ys.sum() == len(ys):
                continue
            d = f"{args.out_dir}/T3/{L}/{up}"
            os.makedirs(d, exist_ok=True)
            np.save(f"{d}/saved_preds.npy", sc.astype(np.float32))
            np.save(f"{d}/saved_labels.npy", ys)
            n_ok += 1
        print(f"[{L}] 落盘 {n_ok} 个靶点", flush=True)


if __name__ == "__main__":
    main()
