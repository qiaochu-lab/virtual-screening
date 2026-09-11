"""LiTENCLIP's CASF dataset is missing mol_src_coord.

Symptom: after switching to mol_forward, it still raises
    TypeError: mol_forward() missing 1 required positional argument: 'mol_src_coord'

Cause: in the CASF (pdbbind) branch of load_dataset, the molecule
coordinates `coord_dataset` are already built and already have
PrependAndAppend applied, but are the only ones never inserted into
net_input -- the pocket side's `pocket_src_coord` is inserted, the molecule
side is not. And LiTENCLIP's molecule tower needs coordinates (it is a
LiTEN force-field-style encoder, unlike UniMol which consumes only a
distance matrix). Every other branch (DUD-E/DEKOIS/LIT-PCBA/T3) has it;
only this one is missing it.

A one-line fix; touches neither the model nor the data.
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
