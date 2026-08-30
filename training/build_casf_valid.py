"""把 LigUnity 打包的 casf.lmdb 拆成 HypSeek 训练时验证集要的三个文件。

为什么要拆
----------
HypSeek 的 CASF 验证分支读 `valid_lig.lmdb` / `valid_prot.lmdb` /
`valid_label_seq.json` 三个文件，而 LigUnity 把 CASF 打包成一个 casf.lmdb
（每条记录同时含配体和口袋字段）。两边只是打包方式不同，数据是同一份。

用 CASF 作验证集是必须的：它决定了 best checkpoint 按 `valid_bedroc` 选，
也就是"虚筛权重" _vs 的选法。若改用 FEP 验证集，选出来的就是已公开的 _rk。

拆的时候保持 pocket 名去重：口袋 lmdb 每个蛋白一条，配体 lmdb 每个复合物一条，
与 HypSeek 的 load_pockets_dataset / load_mols_dataset 的预期一致。
"""
import json
import os
import pickle

import lmdb

B = "/data/work/vs"
SRC = f"{B}/code/LigUnity/test_datasets/casf.lmdb"
OUT = f"{B}/code/LigUnity/test_datasets"


def main():
    e = lmdb.open(SRC, subdir=False, readonly=True, lock=False)
    mols, pockets, seen = [], [], set()
    with e.begin() as t:
        for _k, v in t.cursor():
            d = pickle.loads(v)
            mols.append({k: d[k] for k in ("atoms", "coordinates", "smi", "mol", "label")
                         if k in d} | {"pocket": d["pocket"]})
            if d["pocket"] not in seen:
                seen.add(d["pocket"])
                pockets.append({"pocket": d["pocket"], "pocket_index": d.get("pocket_index", 0),
                                "pocket_atoms": d["pocket_atoms"],
                                "pocket_coordinates": d["pocket_coordinates"]})
    e.close()
    print(f"读到 {len(mols)} 个复合物，{len(pockets)} 个唯一口袋")

    for name, recs in (("valid_lig.lmdb", mols), ("valid_prot.lmdb", pockets)):
        p = f"{OUT}/{name}"
        if os.path.exists(p):
            os.remove(p)
        env = lmdb.open(p, subdir=False, map_size=1 << 34)
        with env.begin(write=True) as w:
            for i, r in enumerate(recs):
                w.put(str(i).encode(), pickle.dumps(r))
        env.close()
        print(f"写入 {p}（{len(recs)} 条）")

    # 标签文件内容与 casf_label_seq.json 完全相同，做个软链避免两份不同步
    src = f"{OUT}/casf_label_seq.json"
    dst = f"{OUT}/valid_label_seq.json"
    if not os.path.exists(dst):
        os.symlink(src, dst)
    print(f"软链 {dst} -> casf_label_seq.json（内容同一份，避免不同步）")


if __name__ == "__main__":
    main()
