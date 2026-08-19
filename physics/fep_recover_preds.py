"""从已存的 embedding 复原 FEP 打分，避免重跑。

LigUnity 的 FEP 分支没落盘 saved_preds.npy（补丁只进了 LiTENCLIP），
但打分的定义就是 pocket_emb @ mol_emb.T 再按口袋取 max —— 与
test_fep_target 里的 `res.max(axis=0)` 逐字一致，所以可以无损复原。
"""
import glob, os
import numpy as np
B = "/data/work/vs-benchmark"

for m in sorted(os.listdir(f"{B}/results/fep")):
    root = f"{B}/results/fep/{m}/FEP"
    if not os.path.isdir(root):
        continue
    n_new = n_have = 0
    for t in sorted(os.listdir(root)):
        d = f"{root}/{t}"
        if os.path.exists(f"{d}/saved_preds.npy"):
            n_have += 1
            continue
        try:
            mol = np.load(f"{d}/saved_mols_embed.npy")
            poc = np.load(f"{d}/saved_target_embed.npy")
        except Exception:
            continue
        res = poc @ mol.T
        np.save(f"{d}/saved_preds.npy", res.max(axis=0))
        n_new += 1
    print(f"  {m:28s} 已有 {n_have:2d}  复原 {n_new:2d}")
