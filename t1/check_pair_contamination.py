"""查 T3 里有多少 protein–ligand pair 其实已经在训练集里。

背景
----
现在的 T3 只做了**时间切分**（取 2025+ 入库的记录），没有对训练集做**内容层面的差集**。
问题在于：一条 2025 年的数据库记录，不代表这个 pair 第一次出现——
可能 2023 年就有了、模型训练过；2025 年只是被重新测了一遍，或才录入 BindingDB。

这类污染会落在 L1/L2（靶点见过的层），让对照层的数字偏高，
从而**高估** L1→L4 的衰减。

这里按 (UniProt, InChIKey) 逐对核对，量化污染比例。
"""
import json

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
B = "/data/work/vs-benchmark"
TD = f"{B}/code/LigUnity/test_datasets"


def train_pairs():
    """LigUnity 训练集里的 (uniprot, inchikey) 对，以及出现过的分子集合。"""
    lab = json.load(open(f"{TD}/train_label_blend_seq_full.json"))
    pairs, mols, n_lig, bad = set(), set(), 0, 0
    for a in lab:
        up = a.get("uniprot")
        for l in a.get("ligands", []):
            smi = l.get("smi") if isinstance(l, dict) else None
            if not smi:
                continue
            n_lig += 1
            m = Chem.MolFromSmiles(smi)
            if m is None:
                bad += 1
                continue
            try:
                ik = Chem.MolToInchiKey(m)
            except Exception:
                bad += 1
                continue
            mols.add(ik)
            if up:
                pairs.add((up, ik))
    print(f"训练集: {len(pairs):,} 个 (靶点,分子) 对，{len(mols):,} 个唯一分子")
    print(f"        （来自 {n_lig:,} 条配体记录，{bad:,} 条解析失败）\n")
    return pairs, mols


def main():
    tp, tm = train_pairs()
    print("%-4s %10s %18s %10s %18s %10s" %
          ("层", "记录数", "pair 训练集已有", "污染率", "分子训练集已有", "占比"))
    print("-" * 78)
    for L in ["L1", "L2", "L3", "L4"]:
        n = pair_hit = mol_hit = 0
        for line in open(f"{B}/data/t3/layers/{L}.jsonl"):
            d = json.loads(line)
            n += 1
            if (d["uniprot"], d["inchikey"]) in tp:
                pair_hit += 1
            if d["inchikey"] in tm:
                mol_hit += 1
        print("%-4s %10s %18s %9.2f%% %18s %9.1f%%" %
              (L, f"{n:,}", f"{pair_hit:,}", pair_hit / n * 100,
               f"{mol_hit:,}", mol_hit / n * 100))

    print("\n判读：")
    print("· pair 污染 = 这个(靶点,分子)组合训练集里已有，模型确实见过 → 必须剔除")
    print("· 分子污染 = 分子见过但配的是别的靶点 → 不算泄漏，但说明化学空间重叠")
    print("· 预期 L3/L4 的 pair 污染应为 0（靶点本身就是新的）")


if __name__ == "__main__":
    main()
