# Checkpoint 来源与官方下载地址

更新日期：2026-10-06

本项目最终主结果使用作者发布、作者提供或官方仓库随附的权重进行统一推理与评测，
**没有使用我们从头训练的权重作为主表结果**。项目中保留的 HypSeek 训练实验只用于诊断
复现差异，其权重未进入最终主表。

服务器完整归档保存了实际运行时使用的精确文件；轻量 Google Drive 交付则让接收方
从下列官方地址下载公开权重，只随包保留没有稳定公开地址的 LigUnity-pocket v2 和
LiTENCLIP v1。下载后请以交付包内 `SHA256SUMS` 为准核对；文件名相同不代表内容一定相同。

## 最终评测权重

| 模型 | 包内文件 | 来源与下载地址 | 与最终结果的关系 |
|---|---|---|---|
| DrugCLIP | `03-checkpoints/ckpt/drugclip/checkpoint_best.pt` | [作者 Google Drive（含训练数据和 checkpoint）](https://drive.google.com/drive/folders/1zW1MGpgunynFxTKXC2Q4RgWxZmg6CInV?usp=sharing)；[直接查看 checkpoint](https://drive.google.com/file/d/1i87thnbNk8qeLF_tLx_BzelTukWbHaTR/view) | 官方公开权重；包内文件大小与公开文件一致。 |
| BindCLIP-randneg | `03-checkpoints/ckpt/bindclip/BindCLIP_randneg.pt` | [作者 checkpoint 文件夹](https://drive.google.com/drive/folders/1KKID5DU_hh2e5sE5Xmem10lfSy_qWwIE?usp=sharing)；公开文件为 `BindCLIP_randneg.7z` | 官方公开压缩包解出的权重。 |
| BindCLIP-hardneg | `03-checkpoints/ckpt/bindclip/BindCLIP_hardneg.pt` | [作者 checkpoint 文件夹](https://drive.google.com/drive/folders/1KKID5DU_hh2e5sE5Xmem10lfSy_qWwIE?usp=sharing)；公开文件为 `BindCLIP_hardneg.7z` | 官方公开压缩包解出的权重。 |
| LigUnity-pocket | `03-checkpoints/ckpt/ligunity/LigUnity_VS/pocket_ranking_vs_v2/checkpoint_avg_41_50.pt` | [LigUnity_VS 官方 Hugging Face](https://huggingface.co/fengb/LigUnity_VS) 提供第一版；本项目最终 350-target 结果使用作者后续单独提供的第二版 `v2` | **最终使用的 v2 不是 Hugging Face 上同名第一版**；精确 v2 以交付包为准，SHA-256 为 `ee1c56b7ade2b0e842051138b9cf65ee7afb5c22a6756f11adec884d9dfa478e`。 |
| LigUnity-protein | `03-checkpoints/ckpt/ligunity/LigUnity_VS/protein_ranking_vs/checkpoint_avg_41-50.pt` | [LigUnity_VS 官方 Hugging Face](https://huggingface.co/fengb/LigUnity_VS) | 官方公开权重。FEP 专用多次重复权重另见 [LigUnity_protein_ranking](https://huggingface.co/fengb/LigUnity_protein_ranking)，不是本路径中的 VS 权重。 |
| LigUnity HGNN | `03-checkpoints/ckpt/ligunity/LigUnity_VS/*/HGNN_*_model.pt` | [LigUnity_VS 官方 Hugging Face](https://huggingface.co/fengb/LigUnity_VS) | 官方公开组合模型组件。 |
| LiTENCLIP v1 | `03-checkpoints/ckpt/litenclip/checkpoint.best_valid_bedroc_0.50.pt` | 当前项目记录中没有可核验的稳定公开下载 URL | 历史 v1 权重，交付包内保底；**LiTENCLIP v2 不参与本次最终主比较，也未打包。** |
| HypSeek VS | `03-checkpoints/ckpt/hypseek/official_checkpoint_avg_41-50_vs.pt` | [作者在 GitHub issue #4 提供的 Google Drive](https://drive.google.com/drive/folders/1O1oT4y9gK_ntnHoTlE4C2vlTk2vNo3O7?usp=drive_link)；[直接查看 VS 文件](https://drive.google.com/file/d/1OEpEsGLQ3m9OeEcuzy9q1nhyabK-2cOF/view) | 官方作者提供；用于最终筛选主表。 |
| HypSeek RK | `03-checkpoints/ckpt/hypseek/official_checkpoint_avg_41-50_rk.pt` | [同一作者 Google Drive](https://drive.google.com/drive/folders/1O1oT4y9gK_ntnHoTlE4C2vlTk2vNo3O7?usp=drive_link)；[直接查看 RK 文件](https://drive.google.com/file/d/1gzDxTfaHfNQa9LHfi6tQY96b6fDjMTUY/view) | 官方作者提供；用于亲和力排序及文档中明确标注的派生分析。 |
| DrugJEPA | `03-checkpoints/ckpt/drugjepa/checkpoint_best.pt` | [作者 checkpoint 文件夹](https://drive.google.com/drive/folders/1tb7LA_AwRmYxPaxMzuLwP5dO2TGqY5ug?usp=sharing)；[直接查看 checkpoint](https://drive.google.com/file/d/1pbrd7Tw6oxn1c3zFVdbn_uOMHQ7DL6R7/view) | 官方公开权重；包内文件大小与公开文件一致。 |
| ConPLex | `03-checkpoints/ckpt/conplex/BindingDB_ExperimentalValidModel.pt` | [官方直接下载](https://cb.csail.mit.edu/cb/conplex/data/models/BindingDB_ExperimentalValidModel.pt)；也可运行 `conplex-dti download --to . --models ConPLex_v1_BindingDB` | 官方公开权重。 |
| SPRINT | `03-checkpoints/ckpt/sprint/sprint.ckpt` | [作者 Google Drive](https://drive.google.com/file/d/1uojdSn1otFKi-DBJyTKoOA6OZbwOQX6U/view?usp=sharing) | 官方公开权重。 |
| SPRINT-ProtBert | `03-checkpoints/ckpt/sprint/sprint-protbert.ckpt` | [作者 Google Drive](https://drive.google.com/file/d/10EgPNsn4U1hLEOHa7Wg7gfou0Droa9Qa/view?usp=sharing) | 官方公开权重；作为随附对照保留。 |
| ConGLUDe | `03-checkpoints/code/conglude/checkpoints/best_model/` | [官方 GitHub 仓库（权重随仓库提供）](https://github.com/ml-jku/conglude/tree/main/checkpoints/best_model) | 官方仓库自带的四个编码器/图网络组件。 |
| Boltz-2 | 由 Boltz 缓存自动管理，不在 `03-checkpoints` 分卷内 | [官方 GitHub](https://github.com/jwohlwend/boltz)；默认安装后由 `boltz predict` 自动获取官方模型 | 本项目没有自行训练 Boltz-2。复现时应固定项目记录的 Boltz 版本和运行参数。 |

## 预训练依赖

| 组件 | 包内位置 | 官方下载地址 |
|---|---|---|
| Uni-Mol molecule encoder | `03-checkpoints/ckpt/hypseek/pretrain/mol_pre_no_h_220816.pt` | [mol_pre_no_h_220816.pt](https://github.com/deepmodeling/Uni-Mol/releases/download/v0.1/mol_pre_no_h_220816.pt) |
| Uni-Mol pocket encoder | `03-checkpoints/ckpt/hypseek/pretrain/pocket_pre_220816.pt` | [pocket_pre_220816.pt](https://github.com/deepmodeling/Uni-Mol/releases/download/v0.1/pocket_pre_220816.pt) |
| ESM-2 35M | `03-checkpoints/ckpt/esm/esm2_t12_35M_UR50D/` | [facebook/esm2_t12_35M_UR50D](https://huggingface.co/facebook/esm2_t12_35M_UR50D) |

## 重要说明

1. 外部链接用于说明来源并提供备用下载；**复现本项目数字时优先使用交付包内权重**。
2. 上游仓库可能更新文件。下载后必须与包内 `SHA256SUMS` 对照，尤其是
   LigUnity-pocket v2、LiTENCLIP v1 和作者后续补发的 HypSeek 权重。
3. 训练脚本和训练数据随包保留，是为了审计与扩展；最终主表没有混用我们自己的
   训练权重。
