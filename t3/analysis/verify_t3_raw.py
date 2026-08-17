"""检查 t3_raw 下的原始打分文件是否完整可用，剔除坏的。

为什么需要：中途 kill 过重复启动的进程，可能有 .npy 正写到一半。
坏文件在后面算指标时才暴露就晚了，这里提前扫一遍。

判定：能加载、长度一致、标签既有正也有负、分数无 NaN/Inf。
不合格的整个靶点目录删掉——宁可少一个靶点，也不要脏数据进结果表。
"""
import argparse, os, shutil
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="/data/yicheng/xqc/vs-benchmark/results/t3_raw")
a = ap.parse_args()

if not os.path.isdir(a.raw):
    print(f"{a.raw} 不存在"); raise SystemExit(0)

for m in sorted(os.listdir(a.raw)):
    root = f"{a.raw}/{m}/T3"
    if not os.path.isdir(root):
        continue
    ok = bad = 0
    reasons = {}
    for layer in sorted(os.listdir(root)):
        for t in sorted(os.listdir(f"{root}/{layer}")):
            d = f"{root}/{layer}/{t}"
            why = None
            try:
                s = np.load(f"{d}/saved_preds.npy")
                l = np.load(f"{d}/saved_labels.npy")
                if len(s) != len(l):
                    why = "长度不一致"
                elif l.sum() == 0 or l.sum() == len(l):
                    why = "标签全同"
                elif not np.isfinite(s).all():
                    why = "分数含 NaN/Inf"
            except Exception as e:
                why = type(e).__name__
            if why:
                shutil.rmtree(d, ignore_errors=True)
                reasons[why] = reasons.get(why, 0) + 1
                bad += 1
            else:
                ok += 1
    print(f"  {m}: 可用 {ok}  剔除 {bad}" + (f"  {reasons}" if reasons else ""))
