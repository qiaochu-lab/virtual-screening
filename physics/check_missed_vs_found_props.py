"""missed / found 两组活性的分子性质对照——支撑 export_rerank_sub.py 的主分析解读。

**为什么需要这个。** 主分析比的是 Boltz 在「检索漏掉的活性 vs 诱饵」和
「检索找到的活性 vs 诱饵」上的 AUROC。如果 missed 明显低于 found，有两个解释：

    (a) 物理和检索的盲区重合——级联加物理这一级补不上什么
    (b) 补回的那批活性本身对任何方法都更难（更大、更新、更怪）

两个数分不开这个歧义，所以要单独查分子性质。

## ⚠️ 看效应量不要看 p；但效应量的**比值**同样会骗人

第一版我把「p 从 3e-80 掉到 0.312」读成「效应消失了」，**这是错的**。
样本量从 1,813 个分子掉到 5 个靶点，p 必然崩掉；而且 n=5 时 Wilcoxon
**双侧 p 的下界就是 2/2^5 = 0.0625**，从一开始就够不到 0.05。
**这个检验只能读方向，不能读显著性。**

比 pooled 和 paired 的效应量（同一批靶点上）比比 p 强，但**收缩率本身也不可靠**：
它是个比值，paired 中位数碰巧落在 pooled 附近就会给出「效应真实」的假象，
而完全掩盖靶点之间的异质。**n 小的时候，唯一可信的是把每个靶点的数直接列出来。**

## 结论：混合比较确实被构成效应污染；组内有没有差异，n=5 定不下来

**三层分析，一层比一层能说得少，最后一层才是对的。**

**第一层（错）**：混合起来比，MW p=3.3e-23、新颖度 p=3.2e-80，看着两组差别巨大。

**第二层（还是错）**：逐靶点配对之后 p 全部掉到 0.2–1.0，我写成「效应消失」。
不对——n 从 1,813 个分子掉到 5 个靶点，p 必然崩，n=5 时 Wilcoxon 双侧下界
就是 2/2^5=0.0625。**这个检验读方向不读显著性。**

**第三层（还是错，但错得更隐蔽）**：改比 pooled 和 paired 的**效应量收缩率**，
得到「MW 收缩 76%/重原子 100% = 构成效应；新颖度只收缩 8% = 组内真实存在」。
**收缩率是个比值，它和本项目退役掉的 EF 比值是同一个坑**，而且这里它骗人的方式
更隐蔽：paired 中位数只是**碰巧**落在 pooled 附近，掩盖了靶点之间的巨大异质。

**第四层（对的）：直接看五个靶点。**
逐靶点 AUC = P(missed 的值 > found 的值)，0.5 表示该靶点内两组分不开：

    靶点        missed found      MW    重原子    logP    TPSA  可旋转键  新颖度(亲和半)
    O14578        36   116   0.619   0.608   0.452   0.743   0.396     0.094
    O42275        27    91   0.217   0.162   0.471   0.241   0.187     0.116
    P20648        35    77   0.500   0.478   0.627   0.228   0.356     0.648
    Q8N1C3       101   100   0.848   0.845   0.558   0.706   0.692     0.508
    Q96DB2        11    60   0.448   0.400   0.803   0.165   0.864     0.470

**七个维度没有一个在五个靶点上同向**，新颖度也不例外（0.09 到 0.65，
两个靶点强烈「missed 更新颖」、一个反向、两个居中）。

所以能说的只有两句：

1. **混合比较那些极小的 p 确实被靶点构成效应污染了**——逐靶点符号是散的，
   把所有靶点倒进一个池子比较，测的主要是「哪些靶点的配体整体更大/更陌生」。
2. **同一靶点内部两组有没有系统差异，n=5 且异质这么大，定不下来。**
   既不能说「已排除」，也不能说「确实存在」。

⚠️ **所以 (b) 既没被证实也没被排除**，主分析的解读必须把它当成**公开的替代解释**
写出来，不能当成已经处理掉的混杂。

## 主分析该怎么读（先写下来，再看数）

    missed ≈ found → 物理不吃「对训练化学的熟悉度」这一套，确实能补上检索因
                     化学不熟而漏掉的活性。级联值得做的正面结果。
    missed ≪ found → 有两个解释分不开：(a) 两个方法共享对训练化学的依赖
                     （Boltz-2 本身也是训练出来的模型），(b) 补回的活性本身更难。
                     **两个都要写，不能只挑一个。**

**出分之后还有一个直接相关的检验**（免费，等分数落地就能做）：
把每个靶点的 **Boltz AUROC 差（missed − found）** 和该靶点的**性质差**
（上表那些 AUC）做相关。如果 O42275 这种「missed 明显更新颖」的靶点
恰好也是 Boltz 差距最大的，那才是 (b) 的正面证据。n=5 证据很弱，
但这是唯一直接对题的检验，比在性质上做边际比较强。

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

    # ---------- 主输出：逐靶点 AUC。n 小的时候只有这个可信 ----------
    print("【主】逐靶点 AUC = P(missed 的值 > found 的值)，0.5 = 该靶点内两组分不开")
    hdr = "%-10s%8s%7s" % ("靶点", "missed", "found")
    for k in keys:
        hdr += "%14s" % k[:12]
    print(hdr)
    print("-" * len(hdr))
    store = {k: [] for k in keys}
    for up in use:
        x = [r for r in sub if r[0] == up and r[1]]
        y = [r for r in sub if r[0] == up and not r[1]]
        line = "%-10s%8d%7d" % (up, len(x), len(y))
        for k in keys:
            a = np.array([r[2][k] for r in x if k in r[2]], float)
            b = np.array([r[2][k] for r in y if k in r[2]], float)
            if len(a) < 2 or len(b) < 2:
                line += "%14s" % "-"
                continue
            auc = stats.mannwhitneyu(a, b).statistic / (len(a) * len(b))
            store[k].append(auc)
            line += "%14.3f" % auc
        print(line)
    print("-" * len(hdr))
    line = "%-10s%8s%7s" % ("五靶点范围", "", "")
    for k in keys:
        v = store[k]
        line += "%14s" % (f"{min(v):.2f}-{max(v):.2f}" if v else "-")
    print(line)
    line = "%-10s%8s%7s" % ("是否同向", "", "")
    for k in keys:
        v = store[k]
        line += "%14s" % ("是" if v and (all(t < 0.5 for t in v) or all(t > 0.5 for t in v))
                          else "否")
    print(line)
    print("\n⚠️ 没有一个维度在五个靶点上同向 → 混合比较那些极小的 p 确实被靶点构成")
    print("   效应污染了；但同靶点内部有没有系统差异，n=5 且异质这么大，定不下来。")
    print("   **(b)「补回的活性本身更难」既没被证实也没被排除**，主分析要把它当成")
    print("   公开的替代解释写出来，不能当成已经处理掉的混杂。")

    # ---------- 次输出：pooled vs paired。留着是为了说明它为什么不够 ----------
    print("\n【次】pooled vs paired 效应量（⚠️ 收缩率是比值，单独看会骗人，")
    print("      paired 中位数碰巧落在 pooled 附近就会掩盖上面那种异质）")
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
    print(f"\n⚠️ n={len(use)} 时 Wilcoxon 双侧 p 下界 {2 / 2 ** len(use):.4f}，"
          "这个检验读方向不读显著性。")
    join_auroc(use, store, keys)


def join_auroc(use, store, keys):
    """把逐靶点的 Boltz AUROC 差（missed−found）和性质 AUC 并排列出来。

    **这是唯一直接对题的检验**：如果「补回的活性本身更难」成立，那么
    missed 活性在性质上越极端的靶点，Boltz 的 missed−found 差就该越负。

    ⚠️ **只列五个点，不报 r 和 p。** 理由不是「n=5 功效不足」，而是内在矛盾：
    **n=5 上的相关系数本身就是个汇总统计量，正是本文件刚证明会骗人的那一类。**
    用它来检验「汇总统计量会骗人」的结论会自相矛盾。让读者看五个点。

    结果文件不存在时（重排还没跑完）跳过，不报错。
    """
    import csv
    import os
    f = f"{B}/results/export/T6_rerank_subset.csv"
    if not os.path.exists(f):
        print("\n[待办] 重排结果还没生成，出分后重跑本脚本会自动并表：")
        print("   逐靶点 Boltz AUROC 差（missed−found） × 上表性质 AUC，只列五个点。")
        return
    diff = {}
    for r in csv.DictReader(open(f)):
        if r.get("method") != "boltz":
            continue
        try:
            m, fo = float(r["auroc_missed_vs_decoy"]), float(r["auroc_found_vs_decoy"])
        except (ValueError, KeyError):
            continue
        if np.isfinite(m) and np.isfinite(fo):
            diff[r["target"]] = m - fo
    rows = [(u, diff[u]) for u in use if u in diff]
    if not rows:
        print("\n[待办] 结果文件里没有可用的逐靶点 AUROC 差，跳过并表。")
        return
    print("\n" + "=" * 72)
    print("并表：Boltz 的 missed−found 差 × 该靶点的性质 AUC（只列点，不报 r 和 p）")
    print("=" * 72)
    hdr = "%-10s%16s" % ("靶点", "Boltz missed−found")
    for k in keys:
        hdr += "%14s" % k[:12]
    print(hdr)
    print("-" * len(hdr))
    order = {u: i for i, u in enumerate(use)}
    for u, d in sorted(rows, key=lambda x: x[1]):
        line = "%-10s%16.4f" % (u, d)
        for k in keys:
            v = store[k]
            line += "%14.3f" % v[order[u]] if order[u] < len(v) else "%14s" % "-"
        print(line)
    print("-" * len(hdr))
    print("读法：若「补回的活性本身更难」成立，性质 AUC 越极端的靶点，")
    print("      Boltz 的 missed−found 差应越负。**五个点，自己看，不给 r 和 p。**")


if __name__ == "__main__":
    main()
