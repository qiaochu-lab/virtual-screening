"""修 HypSeek / LiTENCLIP 的 CASF 分支：它调 model.forward 的方式和自家模型对不上。

症状
----
LiTENCLIP: TypeError: forward() missing 1 required positional argument: 'mol_src_coord'
HypSeek  : ValueError: not enough values to unpack (expected 4, got 3)

原因
----
两个 fork 的 `inference_pdbbind` 都是从 LigUnity 抄来的，写成
    mol_emb, pocket_emb, _, _ = model.forward(**net_input, protein_sequences=seq)
    mol_emb = mol_emb[0]...
但它们自家的 forward 签名/返回值都改过了——
LiTENCLIP 的 forward 多要一个 mol_src_coord（CASF 数据集不提供），
HypSeek 的 forward 返回三个值 (h_prot, h_poc, h_mol)。
这条代码路径作者显然没跑过。

改法
----
不碰模型，改成用两个 fork 自己在别处一直用的写法（DUD-E/DEKOIS/LIT-PCBA 分支）：
    mol_emb    = model.mol_forward(**net_input)
    pocket_emb = model.pocket_forward(protein_sequences=seq, **net_input)
两个塔分开前向，返回就是 [B, D] 张量，不需要 [0] 取第一个元素——
原来的 `mol_emb[0]` 在返回值是张量时会取走第一行，即使不报错也是错的。
"""
import shutil

OLD = """            mol_emb, pocket_emb, _, _ = model.forward(**sample["net_input"], protein_sequences=seq)
            mol_emb = mol_emb[0].detach().cpu().numpy()
            mol_reps.append(mol_emb)
            pocket_emb = pocket_emb[0].detach().cpu().numpy()
            pocket_reps.append(pocket_emb)"""

NEW = """            # 改用两个塔各自的前向：本仓库其它分支（DUD-E/DEKOIS/LIT-PCBA）
            # 一直是这么调的，而 forward() 的签名与返回值和这里的假设对不上。
            mol_emb = model.mol_forward(**sample["net_input"])
            mol_reps.append(mol_emb.detach().cpu().numpy())
            pocket_emb = model.pocket_forward(protein_sequences=seq, **sample["net_input"])
            pocket_reps.append(pocket_emb.detach().cpu().numpy())"""

B = "/data/work/vs-benchmark"
for name in ["HypSeek", "LiTENCLIP"]:
    p = f"{B}/code/{name}/unimol/tasks/test_task.py"
    s = open(p).read()
    if OLD not in s:
        print(f"{name}: 没找到目标代码块（可能已改过）")
        continue
    shutil.copy(p, p + ".casf.bak")
    open(p, "w").write(s.replace(OLD, NEW))
    print(f"{name}: CASF 分支已改（备份 .casf.bak）")
