"""补两个卡住 T1 的 fork bug（HypSeek / LiTENCLIP，都是 LigUnity 的分支）。

① CASF 标签路径写死成绝对路径
   两个 fork 里都是 `open(f"/casf_label_seq.json")`——少了 `{self.args.data}/`，
   直接 FileNotFoundError。LigUnity 原版是对的，是 fork 时手滑。
   CASF 这条分支对我们不只是补 T1：CASF-2016 是 T2 缺的第三套数据。

② HypSeek 把单条蛋白向量复制 N_lig 份再做矩阵乘
       prot_reps = np.repeat(prot_np, mol_reps.shape[0], axis=0)   # [N_lig, D]
       sim_prot  = prot_reps @ mol_reps.T                          # [N_lig, N_lig]
       prot_scores = sim_prot.max(axis=0)
   N 行完全相同，取 max 之后等于单行点积，但内存是 O(N²)。
   DUD-E 最大靶点 52,056 个分子 → 10.8 GB，硬扛过去了；
   LIT-PCBA 最大 361,997 个 → **488 GiB**，直接 MemoryError。
   去掉 repeat，sim_prot 变成 [1, N_lig]，max(axis=0) 结果逐位相同——
   所以已经跑完的 DUD-E / DEKOIS 数字不受影响，不用重跑。
"""
import re
import shutil
import sys

B = "/data/work/vs-benchmark"
FILES = {
    "HypSeek":   f"{B}/code/HypSeek/unimol/tasks/test_task.py",
    "LiTENCLIP": f"{B}/code/LiTENCLIP/unimol/tasks/test_task.py",
}

BAD_CASF = 'json.load(open(f"/casf_label_seq.json"))'
GOOD_CASF = 'json.load(open(f"{self.args.data}/casf_label_seq.json"))'

BAD_REPEAT = """        if prot_np.ndim == 2 and prot_np.shape[0] == 1:
            prot_reps = np.repeat(prot_np, mol_reps.shape[0], axis=0)
        else:
            prot_reps = prot_np  # [B_pr, D]"""
GOOD_REPEAT = """        # 不要按分子数复制：sim_prot 会变成 [N_lig, N_lig]，
        # LIT-PCBA 最大靶点 361,997 个分子时要 488 GiB。
        # 保持 [1 或 B_pr, D]，下面 max(axis=0) 的结果逐位相同。
        prot_reps = prot_np"""


def main():
    for name, path in FILES.items():
        s = open(path).read()
        orig = s
        if BAD_CASF in s:
            s = s.replace(BAD_CASF, GOOD_CASF)
            print(f"{name}: CASF 路径已修")
        elif GOOD_CASF in s:
            print(f"{name}: CASF 路径本来就是对的")
        else:
            print(f"{name}: ⚠️ 没找到 CASF 那行，需要人看一眼")

        if BAD_REPEAT in s:
            s = s.replace(BAD_REPEAT, GOOD_REPEAT)
            print(f"{name}: O(N²) 复制已去掉")

        if s != orig:
            shutil.copy(path, path + ".bak")
            open(path, "w").write(s)
            print(f"{name}: 已写回（原文件备份为 .bak）")
        else:
            print(f"{name}: 无需改动")


if __name__ == "__main__":
    main()
