"""把逐分子的原始打分打包，让合作者能重算这里的每一个指标。

每个模型一个 .npz：键是 "<task>/<layer>/<uniprot>/preds" 和 ".../labels"。
打分转 float32（原始是 float64，指标算下来完全一样），标签转 int8。

用法：python pack_raw.py [输出目录]
"""
import json, os, sys
import numpy as np

B = "/data/work/vs"
OUT = sys.argv[1] if len(sys.argv) > 1 else f"{B}/results/raw_release"
os.makedirs(OUT, exist_ok=True)

SRC = [("T3", f"{B}/results/t3_raw"), ("T1", f"{B}/results/t1_raw")]
manifest = {}

for task, root in SRC:
    if not os.path.isdir(root):
        continue
    for model in sorted(os.listdir(root)):
        mroot = os.path.join(root, model)
        if not os.path.isdir(mroot) or os.path.islink(mroot):
            continue
        blobs, n_t = {}, 0
        for dirpath, _dirs, files in os.walk(mroot):
            if "saved_preds.npy" not in files or "saved_labels.npy" not in files:
                continue
            rel = os.path.relpath(dirpath, mroot).replace(os.sep, "/")
            try:
                p = np.load(os.path.join(dirpath, "saved_preds.npy")).astype(np.float32)
                y = np.load(os.path.join(dirpath, "saved_labels.npy")).astype(np.int8)
            except Exception as e:
                print(f"  跳过 {model}/{rel}: {e}")
                continue
            if p.shape[-1] != y.shape[-1]:
                print(f"  跳过 {model}/{rel}: 长度不一致 {p.shape} vs {y.shape}")
                continue
            blobs[f"{rel}/preds"] = p
            blobs[f"{rel}/labels"] = y
            n_t += 1
        if not blobs:
            continue
        fn = f"{OUT}/{task}_{model}.npz"
        np.savez_compressed(fn, **blobs)
        mb = os.path.getsize(fn) / 1e6
        manifest[f"{task}_{model}"] = {"targets": n_t, "file": os.path.basename(fn),
                                       "size_mb": round(mb, 2)}
        print(f"{task:3} {model:26} {n_t:5} 个靶点  {mb:7.2f} MB")

json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=1)
tot = sum(v["size_mb"] for v in manifest.values())
print(f"\n共 {len(manifest)} 个包，{tot:.1f} MB -> {OUT}")
