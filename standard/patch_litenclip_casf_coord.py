"""LiTENCLIP 的 CASF 数据集漏了 mol_src_coord。

现象：改用 mol_forward 之后仍然报
    TypeError: mol_forward() missing 1 required positional argument: 'mol_src_coord'

原因：CASF（pdbbind）那条 load_dataset 分支里，分子坐标 `coord_dataset` 明明
已经建好、也做了 PrependAndAppend，却唯独没有塞进 net_input——
口袋侧的 `pocket_src_coord` 塞了，分子侧漏了。而 LiTENCLIP 的分子塔要用坐标
（它是 LiTEN 力场式的编码器，不像 UniMol 只吃距离矩阵）。
其它分支（DUD-E/DEKOIS/LIT-PCBA/T3）都是全的，只有这条漏。

补一行即可，不动模型也不动数据。
"""
import shutil

P = "/data/work/vs-benchmark/code/LiTENCLIP/unimol/tasks/test_task.py"
OLD = """                    "pocket_src_coord": RightPadDatasetCoord(
                        coord_pocket_dataset,
                        pad_idx=0,
                    ),
                    "mol_len": RawArrayDataset(mol_len_dataset),"""
NEW = """                    "pocket_src_coord": RightPadDatasetCoord(
                        coord_pocket_dataset,
                        pad_idx=0,
                    ),
                    # 原代码漏了分子坐标：分子塔要用它，口袋侧有、分子侧没有
                    "mol_src_coord": RightPadDatasetCoord(
                        coord_dataset,
                        pad_idx=0,
                    ),
                    "mol_len": RawArrayDataset(mol_len_dataset),"""

s = open(P).read()
if NEW in s:
    print("已经补过了")
elif OLD in s:
    shutil.copy(P, P + ".coord.bak")
    open(P, "w").write(s.replace(OLD, NEW, 1))
    print("已补 mol_src_coord")
else:
    print("⚠️ 没找到目标代码块")
