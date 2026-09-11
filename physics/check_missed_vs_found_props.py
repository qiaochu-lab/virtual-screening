"""missed / found 两组活性的分子性质对照——支撑 export_rerank_sub.py 的主分析解读。

**为什么需要这个。** 主分析比的是 Boltz 在「检索漏掉的活性 vs 诱饵」和
「检索找到的活性 vs 诱饵」上的 AUROC。如果 missed 明显低于 found，有两个解释：

    (a) 物理和检索的盲区重合——级联加物理这一级补不上什么
    (b) 补回的那批活性本身对任何方法都更难（更大、更新、更怪）

两个数分不开这个歧义，所以要单独查分子性质。

## ⚠️ 看效应量，不要看 p——这是这个脚本最容易读错的地方

第一版我把「p 从 3e-80 掉到 0.312」读成「效应消失了」，**这是错的**。
样本量从 1,813 个分子掉到 5 个靶点，p 必然崩掉；而且 n=5 时 Wilcoxon
**双侧 p 的下界就是 2/2^5 = 0.0625**，从一开始就够不到 0.05。
**这个检验只能读方向，不能读显著性。**

分得清「Simpson 构成效应」和「功效不足」的唯一办法是**在同一批靶点上
比 pooled 效应量和 paired 效应量**（所以下表两列都只用 found>=10 的 5 个靶点，
和 AUROC 比较用的是同一批；第一版拿 12 个靶点的 pooled 去比 5 个靶点的 paired，
那个比较本身就是混淆的）。

    收缩接近 100% 或符号翻转  → 靶点构成效应，同靶点内部没有差异
    收缩很小                → 效应在同靶点内部真实存在，只是检验没功效

## 结论：两类维度，结论相反

                      pooled差    paired差   收缩
    MW                 +53.5      +13.0     76%   ← 构成效应
    重原子                +3.0       +0.0    100%   ← 构成效应
    可旋转键              -1.0       +0.0    100%   ← 构成效应
    logP / TPSA         小        小       —     ← 本来就没差
    新颖度(亲和半)        -0.081     -0.075    8%   ← ⚠️ **没有收缩**
    新颖度(结构半)        -0.070     -0.031   56%

**尺寸和理化性质上，(b) 不成立**——混合起来看差别大得吓人（MW p=3.3e-23、
重原子 p=5.4e-19），但那是靶点构成效应：检索失败的靶点碰巧整体配体更大。
同一个靶点内部分不开。典型的 Simpson 悖论，和项目里「均值会骗人」那条同形。

**但新颖度上，(b) 没有被排除。** 对 LigUnity-protein 自己的训练配体
（PocketAffDB 亲和力半，428,767 个），效应量几乎不收缩：**同一个靶点内部，
检索漏掉的活性对训练化学确实更陌生约 0.075 Tanimoto**，5 个靶点里 4 个同向。

这在机制上几乎是同义反复——**检索就是按「与训练化学像不像」排序的**，
所以它漏掉的当然是离训练化学更远的那批。写出来是因为它决定了主分析怎么读。

## 主分析该怎么读（先写下来，再看数）

Boltz 的 missed vs found 有两个方向，**两个方向都能写**：

**missed ≈ found** → 物理方法不吃「对训练化学的熟悉度」这一套，
确实能补上检索因化学不熟而漏掉的活性。**这是级联值得做的正面结果。**

**missed ≪ found** → **不能**说成「两组活性没有系统差异，所以只能是盲区重合」，
因为新颖度这一维确实有差异。能说的是：两组在尺寸和理化性质上分不开，
差异集中在**对训练化学的相似度**上——而 **Boltz-2 本身也是训练出来的模型**。
那么两个方法共享的可能不是「口袋识别的盲区」，而是**对训练化学的依赖**。
这比「盲区重合」更强、也更有意思，而且和全文主线（模型靠化学系列记忆）是同一条。

## ⚠️ 两条限制，必须和数字一起写

1. **n=5，Wilcoxon 下界 0.0625**，这个检验读方向不读显著性。
2. 能做配对的靶点**按构造是检索部分成功的那批**（两组都要有样本）。
   检索完全失败的 3 个靶点（found 精确为 0）上，missed 活性有没有系统差异，
   这个方法测不了。

## 顺带：检索的失败主要是靶点级的

尺寸维度全部塌成构成效应，说明检索不是「在每个靶点上挑走了大分子」，
而是**在某些靶点上基本能用、在另一些靶点上几乎全废**——和 recall@200 的
双峰分布（5 个靶点 49.8%–84.5%，7 个靶点 0%–6.8%，3 个精确为 0）一致。
决定成败的主要是**落在哪个靶点上**。新颖度那 0.075 是叠在上面的分子级残差，
方向和检索的工作原理一致，但量级远小于靶点之间的差距。
"""
import json

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors
from scipy import stats

RDLogger.DisableLog("rdApp.*")
B = "/data/work/vs-benchmark"
MIN_FOUND = 10          # 和 export_rerank_sub.py 的 found AUROC 门限保持一致

PROPS = {"MW": Descriptors.MolWt, "重原子": Descriptors.HeavyAtomCount,
         "logP": Crippen.MolLogP, "TPSA": Descriptors.TPSA,
         "可旋转键": Descriptors.NumRotatableBonds}
# 新颖度 = 对训练配体的最大 Tanimoto（ECFP4, r=2, 2048 位）。两套都看：
# 粗筛用的检索模型是 ligunity_protein_ranking，它读两个标签文件——
# 亲和力半（PocketAffDB，428,767 个配体）和结构半（= DrugCLIP 的集合，13,590 个）。
# 给一个模型划新颖度档必须用它自己的训练集，几套缓存的配体空间差两个数量级。
NOVELTY = {"新颖度(亲和半)": "ligand_novelty.json",
           "新颖度(结构半)": "ligand_novelty_drugclip.json"}


def main():
    man = json.load(open(f"{B}/data/t3/rerank_manifest.json"))
    tn = man["topn"]
    nov = {k: json.load(open(f"{B}/data/t3/{f}")) for k, f in NOVELTY.items()}

    rows = []
    for e in man["entries"]:
        if e["label"] != 1:
            continue
        mol = Chem.MolFromSmiles(e["smi"])
        if mol is None:
            continue
        d = {k: f(mol) for k, f in PROPS.items()}
        for k, cache in nov.items():
            if e["smi"] in cache:
                d[k] = cache[e["smi"]]
        # rank >= topn 就是被 --inject-actives 补回来的，即检索漏掉的那批
        rows.append((e["uniprot"], e["rank"] >= tn, d))
    keys = list(PROPS) + list(NOVELTY)
    print(f"可解析活性 {len(rows):,}")

    n_found = {}
    for up, missed, _ in rows:
        n_found.setdefault(up, 0)
        if not missed:
            n_found[up] += 1
    use = sorted(u for u, n in n_found.items() if n >= MIN_FOUND)
    skip = sorted(u for u, n in n_found.items() if n < MIN_FOUND)
    print(f"可用靶点（found>={MIN_FOUND}，和 AUROC 同一批）{len(use)}: {use}")
    print(f"排除 {len(skip)}: {skip}\n")
    sub = [r for r in rows if r[0] in use]

    print("⚠️ 两列都只用上面这 5 个靶点——pooled 和 paired 必须同批才比得出构成效应")
    print("%-16s%10s%10s%9s%10s%9s" % ("", "pooled差", "paired差", "收缩", "paired p", "p下界"))
    print("-" * 66)
    for k in keys:
        a = [r[2][k] for r in sub if r[1] and k in r[2]]
        b = [r[2][k] for r in sub if not r[1] and k in r[2]]
        if not a or not b:
            continue
        pooled = float(np.median(a) - np.median(b))
        d = []
        for up in use:
            x = [r[2][k] for r in sub if r[0] == up and r[1] and k in r[2]]
            y = [r[2][k] for r in sub if r[0] == up and not r[1] and k in r[2]]
            if x and y:
                d.append(float(np.median(x) - np.median(y)))
        paired = float(np.median(d))
        pw = stats.wilcoxon(d).pvalue if not np.allclose(d, 0) else float("nan")
        if pooled == 0:
            sh = "n/a"
        elif paired != 0 and np.sign(paired) != np.sign(pooled):
            sh = "符号翻转"
        else:
            sh = f"{100 * (1 - abs(paired) / abs(pooled)):.0f}%"
        print("%-16s%10.3f%10.3f%9s%10.3f%9.4f"
              % (k, pooled, paired, sh, pw, 2 / 2 ** len(d)))
        print("%-16s  逐靶点: %s" % ("", ", ".join(f"{x:+.3f}" for x in d)))
    print("\n收缩接近 100% 或符号翻转 → 靶点构成效应（同靶点内没差异）")
    print("收缩很小               → 效应在同靶点内真实存在，只是 n=5 没功效")
    print(f"⚠️ n={len(use)} 时 Wilcoxon 双侧 p 下界 {2 / 2 ** len(use):.4f}，"
          "这个检验读方向不读显著性。")


if __name__ == "__main__":
    main()
