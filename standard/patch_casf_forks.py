"""Fix HypSeek / LiTENCLIP's CASF branch: the way it calls model.forward
doesn't match their own model.

Symptoms
----
LiTENCLIP: TypeError: forward() missing 1 required positional argument: 'mol_src_coord'
HypSeek  : ValueError: not enough values to unpack (expected 4, got 3)

Cause
----
Both forks' `inference_pdbbind` was copied from LigUnity as-is:
    mol_emb, pocket_emb, _, _ = model.forward(**net_input, protein_sequences=seq)
    mol_emb = mol_emb[0]...
but each fork changed its own forward's signature and/or return value --
LiTENCLIP's forward needs an extra mol_src_coord (which the CASF dataset
does not provide), and HypSeek's forward returns three values
(h_prot, h_poc, h_mol). This code path was evidently never exercised by
either author.

Fix
----
Leave the model alone; switch to the call pattern each fork already uses
everywhere else (the DUD-E/DEKOIS/LIT-PCBA branch):
    mol_emb    = model.mol_forward(**net_input)
    pocket_emb = model.pocket_forward(protein_sequences=seq, **net_input)
Forwarding the two towers separately returns [B, D] tensors directly, with
no need to index [0] -- the original `mol_emb[0]` silently takes the first
row whenever the return value is a tensor, which is wrong even when it
doesn't raise.
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
