"""Boltz-2 重排在 350 子集 L4 靶点上的结果（T6-RE 的 Boltz 那半）。

读这个结果之前必须先知道这轮的设计，以及它**不能**回答什么。

## 候选池怎么构成的

每个靶点 = 检索模型的 top-200（`rank < 200`）+ **不在 top-200 里的全部活性**
被补回（`rank >= 200`，`prep_rerank.py --inject-actives`）。所以：

    rank < 200   2,388 条：1,934 个诱饵 + 454 个「检索找到的活性」
    rank >= 200  1,359 条：全部是「检索漏掉的活性」

**所有诱饵都来自 top-200，一个诱饵都没补。** 补回活性是为了解掉召回天花板：
子集上 L4 的 recall@200 只有 22.6%，重排一个近八成活性都不在里面的列表，
无论物理方法多准都做不出什么。

## ⚠️ 这个设计让检索臂在数学上不可用

补回的活性按构造排在**所有诱饵之后**（它们本来就不在 top-200 里）。
极端情形：某靶点 top-200 里一个活性都没有 → 每个活性排在每个诱饵之后 →
**AUROC 精确等于 0**，不是「很低」，是数学上的 0。实测 12 个靶点里 5 个如此。

**所以任何「Boltz 赢检索」的数字都是这个设计造成的假象，在任何覆盖率下都不能引用。**
csv 里仍然保留 retrieval / rank_fusion 的行，只为留档；脚本不再打印它们的配对检验。

## 那么能回答什么

**主分析（能进正文的那个）：检索漏掉的活性，物理方法捞不捞得回来。**
只在「补回的活性 + 诱饵」上算 Boltz 的 AUROC。补回的活性正是检索失败的那批，
如果 Boltz 能把它们排到诱饵上面，那就是物理补上了检索的盲区——这是级联的全部
价值。零假设干净（随机 = 0.5），不需要和检索的排序比，绕开了上面那个构造缺陷。

**配套对照：检索找到的活性，物理方法排得动吗。**
同样对诱饵算 AUROC，但用 `rank < 200` 的活性。两者一比就知道 Boltz 的盲区
是不是和检索的重合：

  · missed ≈ found   → 物理对「检索觉得像」和「检索觉得不像」一视同仁
  · missed ≪ found   → 两者盲区重合，级联加物理这一级补不上什么
  · missed > found   → 物理确实互补，级联有价值

**次分析：整池对随机。** 只回答「Boltz 有没有任何信号」，门槛很低，作参考。

## ⚠️ found 这一列只有 5 个靶点能算

逐靶点的 found 活性数不是「倾斜」，是**双峰**——检索在一个 L4 靶点上要么基本管用，
要么几乎全废，中间几乎没有：

    O14578 116  Q8N1C3 100  O42275 91  P20648 77  Q96DB2 60   ← recall@200 49.8%–84.5%
    P14060   5  Q08828   3  Q13233  1  Q13574  1              ← recall@200 0.2%–6.8%
    O88634   0  O60427   0  P52429  0                         ← 精确为 0，AUROC 无定义

所以：**missed 那一列 12 个靶点全报**（每个靶点都有 ≥11 个 missed 活性和 ≥84 个诱饵）；
**found/missed 的对比只在 found>=10 的 5 个靶点上做**，并且 n=5 的 Wilcoxon
**双侧 p 的下界是 2/2^5 = 0.0625**，本来就够不到 0.05——所以那 5 对直接逐个列出来，
不靠一个够不到显著的 p 值说事。

总体 recall@200 = 25.0%（454 / 1,813），但这个数是两群靶点的混合，单独看没有意义。
**「一半以上的 L4 靶点，检索 top-200 里的活性少于 5 个」比任何 AUROC 都更直白地
说明召回天花板。**

## missed 比 found 低，能不能用「补回的活性本身更难」解释

**定不下来，必须当成公开的替代解释写出来**（check_missed_vs_found_props.py）。

混合起来比，两组在 MW（p=3.3e-23）和新颖度（p=3.2e-80）上差别巨大。但那些 p
被靶点构成效应污染了——逐靶点看，**七个性质维度没有一个在五个靶点上同向**：

    逐靶点 AUC = P(missed 的值 > found 的值)，0.5 = 该靶点内分不开
    靶点          MW    重原子   新颖度(亲和半)
    O14578     0.619   0.608      0.094
    O42275     0.217   0.162      0.116
    P20648     0.500   0.478      0.648
    Q8N1C3     0.848   0.845      0.508
    Q96DB2     0.448   0.400      0.470

新颖度也不例外（0.09–0.65）。所以只能说两句：混合比较确实被构成效应污染；
**同靶点内部有没有系统差异，n=5 且异质这么大，定不下来**——既没排除也没证实。

⚠️ 中途我用过两个更弱的判据，都会骗人，记在这里免得再犯：
**(1) 看配对后 p 变大** —— n 从 1,813 个分子掉到 5 个靶点，p 必然崩，
Wilcoxon 双侧下界就是 2/2^5=0.0625，这个检验读方向不读显著性。
**(2) 看 pooled/paired 的收缩率** —— 收缩率是比值，和本项目退役掉的 EF 比值
同一个坑；paired 中位数碰巧落在 pooled 附近就会给出「效应真实」的假象，
完全掩盖上表那种异质。**n 小的时候唯一可信的是把每个靶点的数列出来。**

## 主分析该怎么读（先写下来，再看数）

    missed ≈ found → 物理不吃「对训练化学的熟悉度」这一套，确实能补上检索因
                     化学不熟而漏掉的活性。**级联值得做的正面结果。**
    missed ≪ found → 有两个解释分不开，**两个都要写，不能只挑一个**：
                     (a) 两个方法共享对训练化学的依赖——Boltz-2 本身也是训练
                         出来的模型，这比「口袋识别的盲区重合」更强，
                         也和全文主线（模型靠化学系列记忆）是同一条；
                     (b) 补回的活性本身更难，上面那张表没能排除它。

**出分之后要做的直接检验**：把逐靶点的 Boltz AUROC 差（missed − found）和
上表的性质 AUC 放在一起看。**只列那五个点，不报 r 和 p**——n=5 上相关系数
本身就是个汇总统计量，正是上面刚证明会骗人的那一类。让读者看五个点。

## ⚠️ 绝对指标不可与前四轮或全库 EF 比

补回活性把整池的活性占比抬到约 48%，P@5/P@10 的随机基线也是 0.48。
这是构造集上的**排序**测试，不是富集测量。
"""
import argparse
import glob
import json
import os
import random

import numpy as np
from scipy import stats

B = "/data/work/vs-benchmark"
# prep_rerank.py 的 manifest 路径是写死的，不跟随 --out
MAN = f"{B}/data/t3/rerank_manifest.json"


def load(out_root):
    aff = {}
    for p in glob.glob(f"{out_root}/shard_*/*/predictions/*/affinity_*.json"):
        n = os.path.basename(p)[9:-5]
        try:
            aff[n] = json.load(open(p))["affinity_pred_value"]
        except Exception:
            pass
    return aff


def auroc(pos, neg):
    """分数越大越靠前。pos/neg 都非空才有定义。"""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    u = stats.mannwhitneyu(pos, neg, alternative="greater").statistic
    return float(u / (len(pos) * len(neg)))


def metrics(lab, sc):
    """P@5 / P@10 / 活性平均名次 / AUROC。分数越大越靠前。"""
    o = np.argsort(-sc)
    lo = lab[o]
    ranks = np.where(lo == 1)[0] + 1
    return (float(lo[:5].mean()), float(lo[:10].mean()), float(ranks.mean()),
            auroc(sc[lab == 1], sc[lab == 0]))


def sign_test(vals, null):
    """逐单位方向一致性：符号检验（双侧二项，p=0.5），平局剔除。

    ⚠️ 这是这轮最大教训的执行件（PATCHES.md「n 小的时候三个汇总统计量各给一个
    答案」）。七个性质维度上，五个靶点没有一个同向——**没有共同效应可合并**，
    三个汇总统计量于是互相打架，因为它们都在回答一个前提不成立的问题。
    不是 n 小的问题，是异质的问题；n 小只是让异质无法被检出。

    **为什么不是二值标记。** 第一版只标「同向/不同向」，队友指出那在纯随机数据上
    必然触发——12 个靶点、中等效应，期望也就 8/12 或 9/12，**对真效应和无效应
    都会亮，读者没法用它区分任何东西**。二值化丢掉了底层计数携带的信息，
    和比值丢掉两个原始数是同一个形状的错。改成报 k/n 加它的精确二项 p。

    **为什么数的是「高于零假设的个数」而不是「与均值同向的个数」。**
    后者让数据自己挑方向，再检验这个方向，是轻度的双重浸渍（k 会被系统性抬高：
    均值方向通常就是多数方向）。数「高于零假设」是方向固定的，
    这就是标准的符号检验，零分布干净地是 Binomial(n, 0.5)。
    代价是它测的是「多数在哪边」而不是「均值方向可不可信」——但正是前者
    才回答「有没有共同效应」。
    """
    v = np.array([x for x in vals if np.isfinite(x)], dtype=float)
    v = v[v != null]                       # 平局剔除，符号检验的标准做法
    n = len(v)
    if n == 0:
        return 0, 0, float("nan")
    k = int((v > null).sum())
    p = float(stats.binomtest(k, n, 0.5).pvalue)   # 双侧
    return k, n, p


def sign_note(k, n, p):
    """一行：k/n + 二项 p + 该不该信这一行的汇总。

    ⚠️ n 很小时符号检验**根本达不到 0.05**（n=5 时即使 5/5 也只有 0.0625）。
    这种情况下「仅供参考」是**检验本身没功效**，不是「逐靶点异质」的证据——
    两者读起来一样但含义完全相反，所以必须分开写，否则会把「测不了」
    误读成「测了，是散的」。
    """
    if n == 0:
        return "逐靶点方向：无可用单位"
    best = 2 / 2 ** n
    if best >= 0.05:
        return (f"逐靶点高于零假设 {k}/{n}   二项 p={p:.4f}"
                f"  ⚠️ n={n} 时符号检验最好也只有 p={best:.4f}，**够不到 0.05**——"
                f"这一行仅供参考是因为**检验没功效**，不是因为方向散")
    tag = ("  方向一致" if p < 0.05 else
           "  ⚠️ **与抛硬币不可区分——这一行仅供参考，结论以逐靶点表为准**")
    return f"逐靶点高于零假设 {k}/{n}   二项 p={p:.4f}{tag}"


def floor_note(n):
    """n 个单位时 Wilcoxon 双侧 p 的下界。够不到 0.05 就必须说明。"""
    f = 2 / 2 ** n if n > 0 else float("nan")
    tail = "，够不到 0.05" if f > 0.05 else ""
    return f"n={n}，双侧 p 下界 {f:.4f}{tail}"


def wilcoxon_vs(vals, null):
    """逐靶点对一个常数零假设做 Wilcoxon。"""
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if len(v) < 5 or np.allclose(v, null):
        return float("nan"), int((v > null).sum()), len(v)
    return float(stats.wilcoxon(v - null).pvalue), int((v > null).sum()), len(v)


def coverage_gate(man, aff):
    """逐靶点完成率。**这是这个脚本存在过的最大教训。**

    上一版把「出分 2,562/3,747」当成一句覆盖率脚注就发了数字。错在：这轮按
    **复合物**切 shard，四个 shard 每个都覆盖全部 12 个靶点，所以崩掉的 shard
    不是拿走几个完整靶点，而是**每个靶点都被咬掉三分之一**（实测逐靶点完成率
    中位 68.7%，0/12 完整）。在一个靶点三分之二的候选上算的 AUROC 不是全集值
    加宽误差棒，是另一个量；缺三分之一成员的 P@5 根本没法解释。

    所以：**任何靶点不满就拒绝出主结论**，不提供「按现有数据凑合」的路径。
    """
    by = {}
    for e in man["entries"]:
        by.setdefault(e["uniprot"], [0, 0])
        by[e["uniprot"]][0] += 1
        by[e["uniprot"]][1] += int(e["name"] in aff)
    print("逐靶点完成率")
    bad = []
    for up, (tot, got) in sorted(by.items()):
        f = got / tot if tot else 0.0
        flag = "" if f >= 0.999 else "  ← 不完整"
        if f < 0.999:
            bad.append((up, got, tot))
        print(f"  {up:10} {got:5d} / {tot:5d}  {f:6.1%}{flag}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="烟雾测试：跳过完成度闸门，把缺的记录用随机分填满，"
                         "只测所有输出路径跑不跑得通。**数字全无意义**，"
                         "csv 改写到 /tmp，绝不碰真实结果路径。"
                         "用途是别等跑完才发现汇总脚本崩了——"
                         "上一轮就白等过半天。")
    args = ap.parse_args()

    man = json.load(open(MAN))
    aff = load(f"{B}/boltz_rerank_sub_out")
    if args.smoke:
        random.seed(0)
        for e in man["entries"]:
            aff.setdefault(e["name"], random.uniform(-3, 3))
        print("[烟雾测试] 已用随机分填满缺的记录——**数字无意义，只测代码路径**")
    tn = man["topn"]
    print(f"Boltz-2 出分 {len(aff):,} / {len(man['entries']):,}\n")

    bad = coverage_gate(man, aff)
    if bad and args.smoke:
        print(f"\n[烟雾测试] 跳过 {len(bad)} 个靶点的完成度闸门")
        bad = []
    if bad:
        print(f"\n⛔ {len(bad)} 个靶点不完整，拒绝出主结论。")
        print("   原因见 coverage_gate() 的注释：按复合物切片时，缺失是每个靶点都缺，")
        print("   不是缺掉整个靶点；部分候选上的 AUROC / P@k 是另一个量，不可报。")
        return
    print("\n✅ 12 个靶点全部完整，可以出结论。\n")

    by = {}
    for e in man["entries"]:
        by.setdefault(e["uniprot"], []).append(e)
    n_found = {up: sum(1 for e in v if e["label"] == 1 and e["rank"] < tn)
               for up, v in by.items()}
    n_missed = {up: sum(1 for e in v if e["label"] == 1 and e["rank"] >= tn)
                for up, v in by.items()}
    n_decoy = {up: sum(1 for e in v if e["label"] == 0) for up, v in by.items()}

    rows = ["target,n_shortlist,n_decoy,n_active_found,n_active_missed,"
            "method,p_at_5,p_at_10,mean_active_rank,auroc,"
            "auroc_missed_vs_decoy,auroc_found_vs_decoy"]
    per = {}
    a_missed, a_found, a_all = [], [], []
    for up, items in sorted(by.items()):
        lab = np.array([e["label"] for e in items], dtype=int)
        rnk = np.array([e["rank"] for e in items], dtype=int)
        if lab.sum() < 3 or lab.sum() == len(lab):
            continue
        # 检索分数越大越好；Boltz 的 affinity_pred_value 越小越好（预测的 log Kd）
        ret = np.array([e["pred"] for e in items], dtype=float)
        bol = -np.array([aff[e["name"]] for e in items], dtype=float)
        fus = -(stats.rankdata(-ret) + stats.rankdata(-bol))

        is_dec = lab == 0
        is_found = (lab == 1) & (rnk < tn)
        is_missed = (lab == 1) & (rnk >= tn)
        am = auroc(bol[is_missed], bol[is_dec])
        af = auroc(bol[is_found], bol[is_dec])
        aa = auroc(bol[lab == 1], bol[is_dec])
        a_missed.append(am)
        a_found.append(af)
        a_all.append(aa)

        for name, sc in (("retrieval", ret), ("boltz", bol), ("rank_fusion", fus)):
            m = metrics(lab, sc)
            per.setdefault(name, []).append(m)
            extra = (am, af) if name == "boltz" else (float("nan"), float("nan"))
            rows.append("%s,%d,%d,%d,%d,%s,%.3f,%.3f,%.2f,%.4f,%.4f,%.4f"
                        % (up, len(items), int(is_dec.sum()), int(is_found.sum()),
                           int(is_missed.sum()), name, *m, *extra))

    n_t = len(a_missed)
    ups = sorted(by)

    print("=" * 72)
    print("召回结构：检索在 L4 上是接近「全有或全无」的")
    print("=" * 72)
    print("%-10s%8s%8s%9s%12s" % ("靶点", "found", "missed", "活性总数", "recall@200"))
    for up in sorted(ups, key=lambda u: -n_found[u]):
        a = n_found[up] + n_missed[up]
        print("%-10s%8d%8d%9d%11.1f%%"
              % (up, n_found[up], n_missed[up], a, 100 * n_found[up] / a if a else 0))
    tf, tm = sum(n_found.values()), sum(n_missed.values())
    print(f"\n  总体 recall@200 = {tf / (tf + tm):.1%}（{tf} / {tf + tm}）"
          "——但这是两群靶点的混合，单独看没有意义")
    print(f"  top-200 里活性 <5 个的靶点：{sum(v < 5 for v in n_found.values())}/{n_t}"
          f"；精确为 0 的：{sum(v == 0 for v in n_found.values())}/{n_t}")
    print("  ⚠️ 这比任何 AUROC 都更直白地说明召回天花板。")

    print("\n" + "=" * 72)
    print("主分析：检索漏掉的活性，Boltz 捞不捞得回来（对诱饵算 AUROC，零假设 0.5）")
    print("=" * 72)
    # ⚠️ 逐靶点表在汇总行**之前**，不是附录。汇总和逐靶点打架时以逐靶点为准。
    print("【逐靶点】这是主表。下面的汇总行只是它的摘要。")
    print("%-10s%8s%8s%8s%14s%14s%13s"
          % ("靶点", "诱饵", "missed", "found", "AUROC missed", "AUROC found", "差"))
    print("-" * 75)
    for i, up in enumerate(ups):
        fa = f"{a_found[i]:14.4f}" if np.isfinite(a_found[i]) else f"{'无定义':>13}"
        df = (f"{a_missed[i] - a_found[i]:+13.4f}"
              if np.isfinite(a_found[i]) else f"{'-':>13}")
        print("%-10s%8d%8d%8d%14.4f%s%s"
              % (up, n_decoy[up], n_missed[up], n_found[up], a_missed[i], fa, df))
    print("-" * 75)

    p, w, n = wilcoxon_vs(a_missed, 0.5)
    v = np.array([x for x in a_missed if np.isfinite(x)])
    k_, n_, pb = sign_test(a_missed, 0.5)
    print(f"\n【汇总】missed vs 诱饵   AUROC 均值 {v.mean():.4f}   "
          f"高于 0.5 的 {w}/{n}   Wilcoxon p={p:.4f}")
    print(f"  {floor_note(n)}")
    print(f"  {sign_note(k_, n_, pb)}")
    print(f"  （{n} 个靶点全可算：每个都有 >=11 个 missed 活性和 >=84 个诱饵）")

    print("\n" + "-" * 72)
    print("对照：检索找到的活性，Boltz 排得动吗——**只有 found>=10 的靶点能算**")
    print("-" * 72)
    idx = [i for i, up in enumerate(ups) if n_found[up] >= 10]
    skip = [up for up in ups if n_found[up] < 10]
    print(f"  可算的 {len(idx)}/{n_t} 个：{[ups[i] for i in idx]}")
    print(f"  排除的 {len(skip)} 个（top-200 里活性 <10，found AUROC 估不出来）：{skip}")
    if idx:
        m_ = np.array([a_missed[i] for i in idx])
        f_ = np.array([a_found[i] for i in idx])
        print(f"\n  {'靶点':10}{'found 数':>9}{'AUROC found':>13}{'AUROC missed':>14}{'差':>9}")
        for j, i in enumerate(idx):
            print("  %-10s%9d%13.4f%14.4f%+9.4f"
                  % (ups[i], n_found[ups[i]], f_[j], m_[j], m_[j] - f_[j]))
        d = m_ - f_
        print(f"\n  配对差 missed − found 中位 {np.median(d):+.4f}   "
              f"missed 更高的 {(d > 0).sum()}/{len(d)}")
        # ⚠️ n=5 的 Wilcoxon 双侧 p 下界是 2/2^5=0.0625，够不到 0.05。
        #    报一个注定不显著的 p 值会误导，所以把下界一起写出来。
        k_, n_, pb = sign_test(d, 0.0)
        print(f"  {sign_note(k_, n_, pb)}")
        if len(d) >= 2:
            pw = stats.wilcoxon(d).pvalue if not np.allclose(d, 0) else float("nan")
            print(f"  Wilcoxon p={pw:.4f}（⚠️ {floor_note(len(d))}；看上表的逐靶点差）")
        print("  （missed ≪ found ⇒ 物理和检索的盲区重合，级联加这一级补不上什么）")
        print("  ⚠️ 解读前先看本文件开头：混合比较的性质差异被靶点构成效应污染了，")
        print("     但**「补回的活性本身更难」并没有被排除**（n=5，逐靶点异质极大）。")
        print("     missed ≪ found 时两个解释都要写，不能只挑一个。")

    print("\n" + "=" * 72)
    print("次分析：整池排序对随机（门槛很低，只答「有没有信号」）")
    print("=" * 72)
    print("%-14s %8s %8s %14s %8s" % ("排序", "P@5", "P@10", "活性平均名次", "AUROC"))
    print("-" * 58)
    for name in ("retrieval", "boltz", "rank_fusion"):
        v = np.array(per.get(name, []))
        if len(v):
            print("%-14s %8.3f %8.3f %14.1f %8.4f"
                  % (name, v[:, 0].mean(), v[:, 1].mean(), v[:, 2].mean(), v[:, 3].mean()))
    print("-" * 58)
    V = np.array(per["boltz"])
    fa = np.array([np.mean([e["label"] for e in by[up]]) for up in sorted(by)])
    for i, lab_, null in ((3, "AUROC", np.full(n_t, 0.5)), (0, "P@5", fa), (1, "P@10", fa)):
        d = V[:, i] - null
        p = stats.wilcoxon(d).pvalue if not np.allclose(d, 0) else float("nan")
        k_, n_, pb = sign_test(d, 0.0)
        print(f"  boltz {lab_:6} {V[:, i].mean():.3f} 对随机 {null.mean():.3f}   "
              f"赢 {(d > 0).sum()}/{n_t}   Wilcoxon p={p:.4f}   （{floor_note(n_t)}）")
        print(f"    {sign_note(k_, n_, pb)}")

    print("\n⚠️ 不打印「对 retrieval」的配对检验：补回的活性按构造排在所有诱饵之后，")
    print("   12 个靶点里 5 个的检索 AUROC 是精确的 0。那个比较在任何覆盖率下都是假象。")

    out = ("/tmp/_smoke_rerank.csv" if args.smoke
           else f"{B}/results/export/T6_rerank_subset.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write("\n".join(rows) + "\n")
    print(f"\n写入 {out}")


if __name__ == "__main__":
    main()
