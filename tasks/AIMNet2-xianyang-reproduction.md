# AIMNet2：xianyang 仓库复现与自建 T3 测试

**日期：2026-10-07。** 本页记录两项工作：独立重跑 xianyang 仓库中的主要可追溯实验，以及把其新版流程用于我们冻结的 T3 数据。它是补充分析；本项目主结果仍按 [2026-10-06 canonical tables](../results/CANONICAL_TABLES.md) 引用。

主要发现是：**FEP 的亲和力排序信号复现了；CASF 的能量重排没有改善 retrieval 候选排序；原 T3 的多数分层指标接近，但 fresh docking 和部分优化终点没有逐项复现。我们自己的 T3 抽样也没有显示稳定的亲和力排序优势。**

因此，“结果基本对上”需要区分两个意思：

1. **复现原实验**：使用原数据、科学代码、权重和参数重新计算，对照原公开结果。
2. **在我们的数据上测试**：沿用原流程，在另一批 T3 分子上得到类似的有限排序信号。这里说的是结论趋势，不能要求两个不同样本的数值一一对应。

## 1. 原工作做了什么，我们复现了什么

来源为 [xianyang123-bit/aimnet_score_pipelines](https://github.com/xianyang123-bit/aimnet_score_pipelines)，固定最新提交 [`e6a598bf80f5`](https://github.com/xianyang123-bit/aimnet_score_pipelines/tree/e6a598bf80f5e284e8a6515b507a2b3192546b62)。已删除的输入与早期结果从 Git 历史提交 [`18072caf3d3d`](https://github.com/xianyang123-bit/aimnet_score_pipelines/tree/18072caf3d3db434f328f44ca3e5e39366382f5b) 恢复，下载时验证 Git blob 身份。

原仓库用公开 AIMNet2 势能模型构造两种分数：新版只计算固定受体下的相互作用能 `Eint = Ecomplex − Epocket − Eligand`；早期 composite 另加去溶剂化与局部配体应变。能量越低，排序越靠前。这是公开表达式的重建，不能称为官方亲和力训练版 AIMNet2(Score)。

| 工作 | 本次独立执行的范围 | 状态 |
| --- | --- | --- |
| KIN66 / PLA15 参考能量 | 81 体系、两套几何、七个成员或计算变体，共 1,134 个记录 | 完成 |
| FEP 新版固定受体 Eint | 16 请求体系，14 支持体系，403 成功配体；重新准备口袋并优化 | 完成 |
| CASF LigUnity retrieval | 重新生成 embedding；24 个支持口袋 × 285 分子，6,840 个分数对 | 完成 |
| CASF 2025 能量重排 | 24 口袋、480 配体、47,896 个可用 pose；另有 2,000-pose 示例 | 完成 |
| 原 T3 新版流程 | 从准备口袋和 fresh docking 重跑；另固定公开 pose 单独复算能量 | 两条均完成，41 体系 / 403 配体 |
| 原 T3 早期 composite | 旧 wB97M 与 2025 各 93 体系 / 917 配体，使用对应归档科学代码 | 完成 |
| 非随机 smoke set | 250 分子，58 active / 192 decoy；旧版和 2025 均独立计算 | 完成 |
| 我们的 T3 数据 | 固定抽样 92 个靶点层组合 / 920 活性配体 | 完成，67 体系 / 547 配体成功 |

旧版和 2025 的归档任务各 134 个，共 268 个执行案例，输入与权重校验、分子 ID 和组分恒等式检查均完成。2025 首轮有一个非有限结果，按相同参数重试一次得到有限值；旧版没有评分错误。没有按与参考结果的接近程度替换有限分数。

## 2. 原实验具体怎样“对上”

### FEP：确实有亲和力排序信号

原流程中的 FEP 是数据集名称；我们计算的是固定受体相互作用能，没有执行自由能微扰模拟。指标为每个体系内 `Spearman(−Eint, pAffinity)`，再对体系等权平均。

| 同一批 14 体系 / 403 配体 | 原公开结果 | 本次复算 |
| --- | --- | --- |
| 优化前平均 Spearman | 0.2713 | 0.2713 |
| 优化后平均 Spearman | 0.4992 | 0.5002 |

403 个成功配体逐项对应，标签一致，无缺失或新增；优化后能量平均绝对差 0.0320 kcal/mol，最大差 4.3828 kcal/mol。这里的“有信号”指模型排序与实验结合强弱有一定一致性，不代表准确预测具体 Kd / Ki，也不代表每个体系都有效。参见 [14 个体系的对照表](../results/aimnet2-reproduction-20261007/fep_system_comparison.csv) 和[原 FEP 说明](https://github.com/xianyang123-bit/aimnet_score_pipelines/blob/e6a598bf80f5e284e8a6515b507a2b3192546b62/work/FEP-fixed-receptor-eint-20260921/README.md)。

同一批配体与已有 Boltz 预测严格配对后，Boltz 平均相关性为 0.6391。AIMNet 减 Boltz 的体系均值差为 −0.1389，体系 bootstrap 95% 区间为 [−0.2914, 0.0079]；这个样本不支持宣称已证明普遍优劣。Boltz 预测本次未重新推理。

### CASF：候选集上的 reranking 没有收益

| 24 个口袋的平均 shortlist EF1 | 原报告 | 本次复算 |
| --- | --- | --- |
| LigUnity retrieval | 3.2024 | 3.2024 |
| 优化前 interaction | 1.6389 | 1.6389 |
| 优化后 interaction | 0.9722 | 0.9722 |
| Composite | 0.9722 | 0.9722 |

每个口袋仅有 retrieval 选出的 20 个候选，EF1 截断实际取第 1 名，按该候选集的活性比例归一化。它不是全库 screening EF1。优化后 interaction 的平均 top-10 活性数为 2.6667，原报告为 2.7083，说明不是所有指标完全一致。旧版 interaction / composite EF1 的 0.7579 / 0.9722 也复现了。

独立 retrieval 分数最大差为 0.00114；23 个口袋的 top-20 完全一致，另一个有一个边界候选不同。能量重排固定使用原候选集，保持对照公平。全部 47,896 个 pose 重新打分后，480 个最终 pose 选择全部与原报告一致；包含独立示例的 49,896 个单点能量，平均绝对差为 0.00238 kcal/mol。

因此，支持的是“**能量重排没有改善这批 retrieval 候选**”，不能写成“retrieval 整体不好”。参见 [CASF 对照表](../results/aimnet2-reproduction-20261007/casf_metrics_comparison.csv) 和[原 CASF / composite 说明](https://github.com/xianyang123-bit/aimnet_score_pipelines/blob/e6a598bf80f5e284e8a6515b507a2b3192546b62/work/aimnet2025-benchmarks-20260906/README.md)。

### 原 T3：指标趋势接近，端到端构象不完全相同

新版原数据的 96 个请求体系：41 个成功准备，25 个化学不支持，20 个源未恢复，7 个准备失败，3 个原输入为空，与原覆盖分类一致。41 个准备口袋的重原子坐标、原子顺序和净电荷一致；39 个所有坐标一致，另两个只有氢坐标差异。

| 新版 Eint 平均 Spearman | 原公开结果 | 固定公开 pose 复算 | Fresh docking 从头重跑 |
| --- | --- | --- | --- |
| L1 | −0.1161 | −0.1161 | −0.1931 |
| L2 | 0.0759 | 0.0825 | 0.1124 |
| L3 | 0.0633 | 0.0579 | 0.1006 |
| L4 | 0.0647 | 0.0770 | 0.0493 |

固定公开输入时，优化前能量平均绝对差仅 0.00139 kcal/mol；优化后平均差 0.05463，最大差 8.08770。Fresh docking 的重原子 RMSD 中位数为 1.188 Å，最大 30.956 Å；优化后能量相对原记录平均差 8.346 kcal/mol。因此没有宣称端到端构象和所有能量精确复现。参见 [41 个体系逐项对照](../results/aimnet2-reproduction-20261007/latest_t3_per_target_comparison.csv) 和[原新版 T3 说明](https://github.com/xianyang123-bit/aimnet_score_pipelines/blob/e6a598bf80f5e284e8a6515b507a2b3192546b62/work/L1234-fixed-receptor-eint-20260921/README.md)。

早期 T3 的两个 composite 实验也完成了独立重算，每个均为相同的 93 体系 / 917 配体：

| 层 | 2025 原 composite ρ | 2025 复算 ρ | 旧版原 composite ρ | 旧版复算 ρ |
| --- | --- | --- | --- | --- |
| L1 | −0.0877 | −0.0939 | −0.0165 | 0.0018 |
| L2 | 0.1018 | 0.1058 | 0.0441 | 0.0466 |
| L3 | 0.2174 | 0.2121 | 0.0698 | 0.0650 |
| L4 | 0.1169 | 0.1186 | 0.0177 | 0.0160 |

早期 2025 的 L3 确实有一些正相关，不能说所有 T3 结果都完全没有信号。不同版本的口袋化学、样本范围和 pose 选择有变化；这些表不能用于单独归因某个模型或流程改动。参见 [186 个体系版本记录](../results/aimnet2-reproduction-20261007/composite_t3_per_target_comparison.csv)。

### Smoke 与量子参考：也已实际重算

| 非随机 250 分子 smoke 的 AUROC | 原报告 | 本次复算 |
| --- | --- | --- |
| 2025 优化前 interaction | 0.5796 | 0.5796 |
| 2025 优化后 interaction | 0.5422 | 0.5427 |
| 2025 composite | 0.5216 | 0.5224 |
| 旧版优化后 interaction | 0.4543 | 0.4497 |
| 旧版 composite | 0.4472 | 0.4486 |

该集合非随机选取，结果接近随机水平，不能代表整个 T3 库。旧版少数优化终点的能量差异仍很大；指标接近不等于逐分子能量全部一致。参见 [smoke 对照表](../results/aimnet2-reproduction-20261007/smoke_metrics_comparison.csv)。

KIN66 / PLA15 的两套几何、七个变体均重新计算。下面固定使用 `released_xyz` 几何、`int_b973c` 参考，RMSE 单位为 kcal/mol：

| 模型 | KIN66 RMSE | PLA15 RMSE |
| --- | --- | --- |
| 旧 wB97M | 62.6472 | 31.0367 |
| 2025 member 0 | 3.4937 | 5.3966 |
| 2025 四成员平均 | 3.2245 | 4.7693 |

1,134 个逐模型预测全部对应，最大的逐项复算能量差小于 0.009 kcal/mol。量子相互作用能准确性改善已复现，但这不等于实验亲和力或 screening 排序一定改善。参见[全部参考指标及几何定义](../results/aimnet2-reproduction-20261007/kin_reference_metrics.csv)。

## 3. 在我们冻结的 T3 数据上测试

从 2026-10-06 冻结的 normalized T3 最终 quota 子集中，按 strict L1–L4 层每层最多抽 24 个靶点，每个随机抽 10 个活性配体，种子为 `20260904`。L3 只有 20 个可用靶点，总计 92 个靶点层组合、90 个不同 UniProt、920 个请求案例；失败或化学排除后不补抽样。

沿用原科学准备和评分模块：恢复精确残基源结构，一个残基 padding，ACE/NME 封端，Amber14 / pH 7.4 补氢；fresh ETKDGv3 / MMFF94s 构象；SMINA exhaustiveness 8、seed 1、cpu 4、一个 mode；固定受体，对配体做 FIRE，最多 1,000 步，fmax 0.002 eV/Å。路径通过适配脚本替换；逐配体调用原 docking 阶段以保留其他合法分子的运行机会，分子身份校验拒绝仍保留。

| 层 | 成功体系 / 有效相关体系 | 成功配体 | AIMNet Eint 平均 ρ | SMINA 平均 ρ |
| --- | --- | --- | --- | --- |
| L1 | 16 / 15 | 129 | −0.0204 | −0.0106 |
| L2 | 17 / 17 | 139 | −0.1445 | −0.0948 |
| L3 | 14 / 14 | 122 | 0.0667 | 0.2165 |
| L4 | 20 / 19 | 157 | −0.0363 | 0.1162 |

20 个体系化学不支持，5 个准备失败；67 个成功准备体系的 670 个配体中，123 个被 docking 或身份校验拒绝。最终 547 / 920 = 59.5% 成功，502 个优化收敛，45 个有限但未收敛，没有评分或体系执行错误。相关性只反映成功评分的样本，不能外推到被排除的样本。仅保留收敛分子时，L1–L4 的平均 ρ 为 −0.0197、−0.1369、0.0462、0.0488，趋势仍然弱。

另与 HypSeek `_rk`、LigUnity protein / pocket、DrugCLIP、LiTENCLIP、ConGLUDe 的冻结分数严格配对：每个比较使用同一靶点的相同有效配体，至少五个配体，按体系做 10,000 次 bootstrap。24 个模型层比较中，AIMNet 减已有模型的均值差 13 个为正、11 个为负，**全部 95% 区间包含 0**，没有看到跨层稳定优势。已有六个模型本次没有重新推理，NPZ 哈希、模型分子顺序与标签对应已经核对。

这些是活性分子的亲和力排序，不能拿它报告全库 EF，也没有重排我们完整的 active / decoy 检索库。参见 [547 个新分数](../results/aimnet2-reproduction-20261007/own_t3_scores.csv)、[67 个体系指标](../results/aimnet2-reproduction-20261007/own_t3_per_target_metrics.csv)、[六模型配对区间](../results/aimnet2-reproduction-20261007/own_t3_paired_baselines.csv)。

## 4. “没什么问题”的准确范围

本次证据支持原工作的主要指标与结论，不能写成“所有输入、构象和数值完全一致”或“AIMNet2 在所有任务上都没有信号”。

- 新版和 FEP 均使用原安全邻居表；legacy 保留全范围 Coulomb 邻居。独立 NumPy 邻居矩阵直接调用 TorchScript 的三个体系校验通过，最大 interaction 差为 0.00158 kcal/mol。
- 旧版和 2025 的 268 个归档案例，组分恒等式误差均小于 1e−6 kcal/mol；所有有限结果的 complex 优化后能量不高于初值。
- 早期 PDB 电荷读取沿用原规则，缺失字段时为 0；新版 T3 / FEP 使用准备后的净电荷。保留这些协议差异用于各自复现，不把协议差异当成单一权重效果。
- CASF 有少数优化终点的大能量差，最大 interaction 差约 289 kcal/mol。双方保存结构重新做单点计算接近各自记录，支持优化轨迹 / 终点差异的解释，但不证明所有偏差都来自同一原因。
- 原始非有限结果、准备失败、化学排除和身份拒绝都保留。首个有限重试仅用于原来失败的分子，没有按参考相似度挑选优化结果。
- 原仓库八个历史探索 CSV 的精确参数或输入来源未完整记录，HiQBind 外部参考准备与权重来源也未确证，未列为严格复现完成项。

原记录 CUDA runtime 为 cu130，本次为 cu126；原完整 docking / RDKit 环境未完全锁定。实际运行使用 Python 3.11、AIMNet 0.2、PyTorch 2.14、OpenMM 8.6.1 Reference、PDBFixer 1.12；[汇总记录](../results/aimnet2-reproduction-20261007/summary.json) 保留实际版本、提交、权重身份和误差。部分早期 case receipt 只记录模型别名，后续同时记录解析路径和哈希；汇总区分两种证据，没有事后补写成早期已有逐案例哈希。

## 5. 结果记录与复核方式

本页附带[轻量结果目录](../results/aimnet2-reproduction-20261007)，包含逐体系原结果 / 复算对照、我们的新分数、配对区间、量子参考指标和协议汇总。它不是安装好环境的完整运行包，也没有托管权重、全部分子原子坐标或完整 T3 库。

在本仓库根目录运行：

```bash
python3 physics/verify_aimnet_reproduction_summary.py
```

脚本仅用 Python 标准库，从所附逐体系记录重新核对原结果和复算结果的等权均值，并从 547 个新分数计算 T3 的带 ties Spearman、能量分解与收敛计数；不重新执行分子能量计算，也不重新计算未附原始分数的 screening AUROC。

## 可以发给合作方的简短说明

> 我们把你 GitHub 里主要可追溯的实验独立跑了一遍：FEP 平均相关性从原报告的 0.4992 复算到 0.5002；CASF 优化后 interaction / composite 的 EF1 都复算到 0.9722，能量重排没有改善 retrieval 的候选。新旧 T3 和 smoke 的主要趋势也一致。我们另用自己的冻结 T3 抽样测试，亲和力排序仍偏弱。少数 fresh docking 构象和优化终点有差异，因此说的是主要指标与结论复现，没有宣称所有数值完全一致。
