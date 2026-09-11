"""Make the FEP task save the raw scores, so the unified evaluation layer can recompute metrics.

The official implementation only saves embeddings and labels and reports R^2
alone, and zeroes R^2 outright whenever corr<0 -- which conflates "ranked
backwards" with "completely unrelated". A negative correlation is a
meaningful signal (it means the model is systematically ranking the strong
ones last) and should not be erased.

This only adds one line, saved_preds.npy -- it does not touch any model
logic.
"""
import sys
B = "/data/work/vs-benchmark"
targets = {
    "LigUnity":  f"{B}/code/LigUnity/unimol/tasks/test_task.py",
    "HypSeek":   f"{B}/code/HypSeek/unimol/tasks/test_task.py",
    "LiTENCLIP": f"{B}/code/LiTENCLIP/unimol/tasks/test_task.py",
}
OLD = '''        np.save(f"{self.args.results_path}/FEP/{target}/saved_labels.npy", real_dg)'''
NEW = '''        np.save(f"{self.args.results_path}/FEP/{target}/saved_labels.npy", real_dg)
        # PATCH: 落盘原始打分，供统一评测层重算（官方只存 embedding，
        # 且 R² 在 corr<0 时被归零，会掩盖「排序方向反了」这个有意义的信号）
        np.save(f"{self.args.results_path}/FEP/{target}/saved_preds.npy", pred_dg)'''

for name, p in targets.items():
    try:
        s = open(p).read()
    except FileNotFoundError:
        print(f"  {name}: 文件不存在"); continue
    if "FEP/{target}/saved_preds.npy" in s:
        print(f"  {name}: 已打过补丁"); continue
    if OLD not in s:
        print(f"  {name}: 找不到插入点"); continue
    n = s.count(OLD)
    s = s.replace(OLD, NEW)
    open(p, "w").write(s)
    print(f"  {name}: 已在 {n} 处插入 saved_preds")
