"""Molecular-property control comparing the missed vs. found active sets --
supports the interpretation of export_rerank_sub.py's main analysis.

**Why this is needed.** The main analysis compares Boltz's AUROC on "actives
retrieval missed vs. decoys" against "actives retrieval found vs. decoys". If
missed scores clearly below found, there are two explanations:

    (a) physics and retrieval share the same blind spot -- adding a physics
        stage to the cascade recovers little
    (b) the recovered actives are themselves harder for any method (bigger,
        newer, weirder)

The two numbers alone cannot resolve this ambiguity, so molecular properties
need to be checked separately.

## Warning: read the effect size, not the p-value -- but the **ratio** of effect sizes can mislead too

In the first version I read "p drops from 3e-80 to 0.312" as "the effect
disappeared" -- **that was wrong**. The sample shrinks from 1,813 molecules to
5 targets, so p necessarily collapses; and at n=5 the Wilcoxon **two-sided
p floor is 2/2^5 = 0.0625**, which never reaches 0.05 to begin with.
**This test can only tell direction, not significance.**

Comparing the pooled and paired effect sizes (on the same set of targets) is
stronger than comparing p-values, but **the shrinkage ratio itself is also
unreliable**: it is a ratio, and a paired median that happens to land near the
pooled one gives a false impression that "the effect is real" while completely
masking heterogeneity across targets. **With n this small, the only thing that
can be trusted is listing each target's numbers directly.**

## Conclusion: the pooled comparison is genuinely contaminated by a composition
effect; whether there is a within-group difference cannot be settled at n=5

**Three layers of analysis, each saying less than the last -- only the final
layer is correct.**

**Layer 1 (wrong)**: pooling everything together, MW p=3.3e-23, novelty
p=3.2e-80 -- the two groups look enormously different.

**Layer 2 (still wrong)**: after pairing per target, all p-values drop to
0.2-1.0, which I wrote up as "the effect disappeared". Not so -- n drops from
1,813 molecules to 5 targets, so p necessarily collapses; at n=5 the Wilcoxon
two-sided floor is 2/2^5=0.0625. **This test reads direction, not
significance.**

**Layer 3 (still wrong, but more subtly so)**: switching to compare the
**shrinkage ratio** between the pooled and paired effect sizes gives "MW
shrinks 76% / heavy atoms 100% = composition effect; novelty shrinks only 8% =
genuinely present within groups." **The shrinkage ratio is itself a ratio, the
same trap as the EF ratio this project retired elsewhere**, and here it
misleads more subtly: the paired median only **happens** to land near the
pooled one, masking huge heterogeneity across targets.

**Layer 4 (correct): look at the five targets directly.**
Per-target AUC = P(missed value > found value); 0.5 means the two groups are
indistinguishable within that target:

    target       missed found      MW  heavy atoms   logP    TPSA  rot. bonds  novelty (affinity half)
    O14578        36   116   0.619   0.608   0.452   0.743   0.396     0.094
    O42275        27    91   0.217   0.162   0.471   0.241   0.187     0.116
    P20648        35    77   0.500   0.478   0.627   0.228   0.356     0.648
    Q8N1C3       101   100   0.848   0.845   0.558   0.706   0.692     0.508
    Q96DB2        11    60   0.448   0.400   0.803   0.165   0.864     0.470

**Not one of the seven dimensions points the same way across all five
targets**, novelty included (0.09 to 0.65 -- two targets strongly "missed is
more novel", one reversed, two in between).

So only two statements are supportable:

1. **The pooled comparison's tiny p-values are genuinely contaminated by a
   target composition effect** -- the per-target sign is scattered, and
   pooling all targets into one comparison mostly measures "which targets'
   ligands are bigger/more unusual overall."
2. **Whether there is a systematic within-target difference between the two
   groups cannot be settled, at n=5 with this much heterogeneity.** It can
   neither be called "ruled out" nor "confirmed."

Warning: **so (b) is neither confirmed nor ruled out**, and the main
analysis's interpretation must present it as an **open alternative
explanation**, not as an already-handled confound.

## How to read the main analysis (write this down before looking at the numbers)

    missed ~ found  -> physics does not buy into "familiarity with training
                        chemistry" the way retrieval does, and genuinely
                        recovers actives retrieval missed for chemistry
                        reasons. A positive result worth the cascade.
    missed << found -> two explanations that cannot be told apart: (a) the two
                        methods share the same dependence on training
                        chemistry (Boltz-2 is itself a trained model too),
                        (b) the recovered actives are themselves harder.
                        **Both must be written up, not just one picked.**

**Once scores land there is one more directly relevant test** (free, doable as
soon as the scores exist): correlate each target's **Boltz AUROC gap
(missed - found)** with that target's **property gap** (the AUCs in the table
above). If a target like O42275, where missed is clearly more novel, is also
where the Boltz gap is largest, that would be positive evidence for (b). n=5
is weak evidence, but it is the only test that bears directly on the question,
stronger than a marginal comparison on properties.

## Warning: two caveats that must travel with the numbers

1. **n=5, Wilcoxon floor 0.0625** -- this test reads direction, not
   significance.
2. The targets that can be paired **are, by construction, the ones where
   retrieval partially succeeded** (both groups need samples). Whether the
   missed actives differ systematically on the 3 targets where retrieval
   failed completely (found is exactly 0) cannot be measured by this method.

## Aside: retrieval's failure is mostly at the target level

All of the size-related dimensions collapse into a composition effect, which
means retrieval is not "picking off the bigger molecules within every
target" but rather **works reasonably well on some targets and is nearly
useless on others** -- consistent with recall@200's bimodal distribution (5
targets at 49.8%-84.5%, 7 targets at 0%-6.8%, 3 exactly 0%). Which target a
molecule falls on is the main determinant of success or failure. The 0.075 on
novelty is a molecule-level residual riding on top of that, pointing the same
direction as how retrieval works, but far smaller in magnitude than the
gap between targets.
"""
import json

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors
from scipy import stats

RDLogger.DisableLog("rdApp.*")
B = "/data/work/vs-benchmark"
MIN_FOUND = 10          # matches the found-AUROC threshold in export_rerank_sub.py

PROPS = {"MW": Descriptors.MolWt, "重原子": Descriptors.HeavyAtomCount,
         "logP": Crippen.MolLogP, "TPSA": Descriptors.TPSA,
         "可旋转键": Descriptors.NumRotatableBonds}
# Novelty = max Tanimoto to the training ligands (ECFP4, r=2, 2048 bits). Check both:
# the retrieval model used for the shortlist is ligunity_protein_ranking, which reads
# two label files -- the affinity half (PocketAffDB, 428,767 ligands) and the structure
# half (= DrugCLIP's set, 13,590 ligands).
# Novelty tiers for a given model must use that model's own training set -- the cached
# ligand spaces differ by two orders of magnitude between the two.
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
        # rank >= topn means it was injected back by --inject-actives, i.e. the batch retrieval missed
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

    # ---------- Main output: per-target AUC. With n this small, only this is trustworthy ----------
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

    # ---------- Secondary output: pooled vs paired. Kept to show why it falls short ----------
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
    """List, per target, the Boltz AUROC gap (missed - found) side by side with the property AUCs.

    **This is the only test that directly bears on the question**: if "the
    recovered actives are themselves harder" holds, then targets where the
    missed actives are more extreme on a property should show a more
    negative Boltz missed-found gap.

    Warning: **list only the five points, do not report r or p.** Not because
    n=5 lacks power, but because of an internal contradiction: **a
    correlation coefficient over n=5 is itself a summary statistic -- exactly
    the kind this file just showed can mislead.** Using one to test the
    conclusion that summary statistics mislead would be self-contradictory.
    Let the reader look at the five points.

    Skip without erroring if the results file does not exist yet (rerank
    still running).
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
