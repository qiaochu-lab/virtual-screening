"""把各模型落在 t3_raw 下的原始打分归集到统一结果目录。"""
import argparse, os, shutil
B = "/data/work/vs-benchmark"
ap = argparse.ArgumentParser()
ap.add_argument("--raw", default=f"{B}/results/t3_raw")
ap.add_argument("--suffix", default="", help="模型名后缀，如 _5a 用于区分 5Å 对照")
a = ap.parse_args()
if not os.path.isdir(a.raw):
    print(f"{a.raw} 不存在"); raise SystemExit(0)
for m in sorted(os.listdir(a.raw)):
    src = f"{a.raw}/{m}/T3"
    if not os.path.isdir(src): continue
    n = 0
    for layer in sorted(os.listdir(src)):
        for t in sorted(os.listdir(f"{src}/{layer}")):
            d = f"{B}/results/t3/{m}{a.suffix}/{layer}/{t}"
            os.makedirs(d, exist_ok=True)
            ok = True
            for f in ["saved_preds.npy", "saved_labels.npy"]:
                p = f"{src}/{layer}/{t}/{f}"
                if os.path.exists(p): shutil.copyfile(p, f"{d}/{f}")
                else: ok = False
            n += ok
    print(f"  {m}{a.suffix}: 归集 {n} 个靶点")
