"""让 FEP 任务把原始打分存下来，供统一评测层重算。

官方实现只存 embedding 和标签，指标只报 R²，而且 corr<0 时直接把 R² 归零——
这会把「排序方向反了」和「完全无关」混为一谈。负相关是有意义的信号
（说明模型系统性地把强的排后面），不该被抹掉。

这里只补一行 saved_preds.npy，不动任何模型逻辑。
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
