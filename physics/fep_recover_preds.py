"""Recover FEP scores from the saved embeddings, avoiding a rerun.

LigUnity's FEP branch never wrote saved_preds.npy to disk (the patch only
went into LiTENCLIP), but the score is defined as pocket_emb @ mol_emb.T
followed by a max over the pocket -- identical, line for line, to
`res.max(axis=0)` in test_fep_target, so it can be recovered losslessly.
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
