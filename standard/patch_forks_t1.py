"""Fix two fork bugs blocking T1 (HypSeek / LiTENCLIP, both LigUnity forks).

(1) The CASF label path is hardcoded as an absolute path
   Both forks have `open(f"/casf_label_seq.json")` -- missing the
   `{self.args.data}/` prefix, which raises FileNotFoundError outright.
   LigUnity's original is correct; this is a slip introduced by forking.
   This CASF branch matters beyond fixing T1: CASF-2016 is the third
   dataset T2 is missing.

(2) HypSeek replicates a single protein vector N_lig times before the
    matrix multiply
       prot_reps = np.repeat(prot_np, mol_reps.shape[0], axis=0)   # [N_lig, D]
       sim_prot  = prot_reps @ mol_reps.T                          # [N_lig, N_lig]
       prot_scores = sim_prot.max(axis=0)
   All N rows are identical, so the result after max is the same as a
   single-row dot product, but memory is O(N^2).
   DUD-E's largest target has 52,056 molecules -> 10.8 GB, survivable;
   LIT-PCBA's largest has 361,997 -> **488 GiB**, an outright MemoryError.
   Dropping the repeat makes sim_prot [1, N_lig], and max(axis=0) is
   identical element-for-element -- so the already-completed DUD-E /
   DEKOIS numbers are unaffected and do not need to be rerun.
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
