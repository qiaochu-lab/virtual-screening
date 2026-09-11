"""Boltz-2 rerank results on the 350-subset L4 targets (the Boltz half of T6-RE).

Read this result only after understanding how this run was designed, and what
it **cannot** answer.

## How the candidate pool was built

Each target = the retrieval model's top-200 (`rank < 200`) + **every active
not in the top-200** injected back (`rank >= 200`, via
`prep_rerank.py --inject-actives`). So:

    rank < 200   2,388 records: 1,934 decoys + 454 "actives retrieval found"
    rank >= 200  1,359 records: all "actives retrieval missed"

**Every decoy comes from the top-200 -- not one decoy was injected.** Actives
were injected back to remove the recall ceiling: on the subset, L4's
recall@200 is only 22.6%, and reranking a list missing nearly 80% of the
actives cannot demonstrate anything, no matter how accurate the physics
method is.

## Warning: this design makes the retrieval arm mathematically unusable

The injected actives are, by construction, ranked **after every decoy** (they
were not in the top-200 to begin with). In the extreme case: a target with
zero actives in its top-200 -> every active ranks below every decoy ->
**AUROC exactly equals 0**, not "very low" but mathematically 0. Measured: 5
of the 12 targets hit this exactly.

**So any "Boltz beats retrieval" number is an artefact of this design and
cannot be quoted at any coverage level.** The retrieval / rank_fusion rows
are still kept in the csv for the record; the script no longer prints their
paired test.

## So what can it answer

**Main analysis (the one that belongs in the writeup): can physics recover
the actives retrieval missed.** Compute Boltz's AUROC only on "injected
actives + decoys". The injected actives are exactly the batch retrieval
failed on; if Boltz can rank them above the decoys, that is physics
recovering retrieval's blind spot -- the entire value case for the cascade.
The null hypothesis is clean (random = 0.5) and needs no comparison against
retrieval's own order, sidestepping the construction flaw above.

**Companion control: can physics move the actives retrieval already found.**
Compute AUROC against decoys the same way, but using the `rank < 200`
actives. Comparing the two tells you whether Boltz's blind spot overlaps
retrieval's:

  . missed ~ found  -> physics treats "retrieval thinks it's similar" and
                        "retrieval thinks it's not" the same way
  . missed << found -> the two blind spots overlap; adding a physics stage
                        to the cascade recovers little
  . missed > found  -> physics is genuinely complementary; the cascade has
                        value

**Secondary analysis: whole pool vs. random.** Only answers "does Boltz carry
any signal at all" -- a very low bar, kept for reference.

## Warning: the found column can only be computed for 5 targets

The per-target found-active count is not "skewed", it is **bimodal** --
retrieval on an L4 target is either basically working or nearly worthless,
with almost nothing in between:

    O14578 116  Q8N1C3 100  O42275 91  P20648 77  Q96DB2 60   <- recall@200 49.8%-84.5%
    P14060   5  Q08828   3  Q13233  1  Q13574  1              <- recall@200 0.2%-6.8%
    O88634   0  O60427   0  P52429  0                         <- exactly 0, AUROC undefined

So: **the missed column is reported for all 12 targets** (each has >=11
missed actives and >=84 decoys); **the found/missed comparison is done only
on the 5 targets with found>=10**, and at n=5 the Wilcoxon **two-sided p
floor is 2/2^5 = 0.0625**, which never reaches 0.05 to begin with -- so those
5 pairs are listed individually rather than leaning on a p-value that can
never be significant.

Overall recall@200 = 25.0% (454 / 1,813), but this number mixes two distinct
groups of targets and is meaningless on its own. **"More than half the L4
targets have fewer than 5 actives in retrieval's top-200" states the recall
ceiling more plainly than any AUROC.**

## Can "the recovered actives are themselves harder" explain missed < found

**Cannot be settled -- must be written up as an open alternative explanation**
(check_missed_vs_found_props.py).

Pooled, the two groups differ hugely on MW (p=3.3e-23) and novelty
(p=3.2e-80). But those p-values are contaminated by a target composition
effect -- per target, **not one of the seven property dimensions points the
same way across all five targets**:

    per-target AUC = P(missed value > found value); 0.5 = indistinguishable within that target
    target        MW    heavy atoms   novelty (affinity half)
    O14578     0.619   0.608      0.094
    O42275     0.217   0.162      0.116
    P20648     0.500   0.478      0.648
    Q8N1C3     0.848   0.845      0.508
    Q96DB2     0.448   0.400      0.470

Novelty is no exception (0.09-0.65). So only two statements hold: the pooled
comparison is genuinely contaminated by a composition effect;
**whether there's a systematic within-target difference cannot be settled at
n=5 with this much heterogeneity** -- neither ruled out nor confirmed.

Warning: along the way I used two weaker criteria, both misleading -- noted
here so they aren't repeated:
**(1) watching p grow after pairing** -- n drops from 1,813 molecules to 5
targets, so p necessarily collapses; the Wilcoxon two-sided floor is
2/2^5=0.0625, and this test reads direction, not significance.
**(2) watching the pooled/paired shrinkage ratio** -- the shrinkage ratio is
itself a ratio, the same trap as the EF ratio this project retired
elsewhere; a paired median that happens to land near the pooled one gives a
false impression that "the effect is real", completely masking the
heterogeneity in the table above. **With n this small, the only trustworthy
thing is listing each target's numbers.**

## How to read the main analysis (write this down before looking at the numbers)

    missed ~ found  -> physics does not buy into "familiarity with training
                        chemistry", and genuinely recovers actives retrieval
                        missed for chemistry reasons. **A positive result
                        worth the cascade.**
    missed << found -> two explanations that cannot be told apart, **both
                        must be written up, not just one picked**:
                        (a) the two methods share the same dependence on
                            training chemistry -- Boltz-2 is itself a
                            trained model too, which is a stronger claim than
                            "the pocket-recognition blind spots overlap", and
                            follows the same throughline as the rest of the
                            project (models rely on chemical-series memory);
                        (b) the recovered actives are themselves harder, and
                            the table above failed to rule this out.

**A direct test to run once scores land**: put the per-target Boltz AUROC gap
(missed - found) side by side with the property AUCs from the table above.
**List only those five points, do not report r or p** -- a correlation
coefficient over n=5 is itself a summary statistic, exactly the kind just
shown above to mislead. Let the reader look at the five points.

## Warning: absolute metrics are not comparable to the first four runs or the full-pool EF

Injecting actives back raises the pool's active fraction to about 48%, so the
random baseline for P@5/P@10 is also 0.48. This is a **ranking** test on a
constructed set, not an enrichment measurement.
"""
import argparse
import glob
import json
import os
import random

import numpy as np
from scipy import stats

B = "/data/work/vs-benchmark"
# prep_rerank.py hardcodes the manifest path; it does not follow --out
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
    """Higher score ranks first. Defined only when both pos and neg are non-empty."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    u = stats.mannwhitneyu(pos, neg, alternative="greater").statistic
    return float(u / (len(pos) * len(neg)))


def metrics(lab, sc):
    """P@5 / P@10 / mean active rank / AUROC. Higher score ranks first."""
    o = np.argsort(-sc)
    lo = lab[o]
    ranks = np.where(lo == 1)[0] + 1
    return (float(lo[:5].mean()), float(lo[:10].mean()), float(ranks.mean()),
            auroc(sc[lab == 1], sc[lab == 0]))


def sign_test(vals, null):
    """Per-unit directional agreement: a sign test (two-sided binomial, p=0.5), ties dropped.

    Warning: this is the executable form of this round's biggest lesson
    (PATCHES.md, "with n this small, the three summary statistics each give a
    different answer"). Across the seven property dimensions, not one of the
    five targets agrees in direction -- **there is no common effect to pool**,
    so the three summary statistics fight each other because they are all
    answering a question whose premise does not hold. It is not a small-n
    problem, it is a heterogeneity problem; small n just keeps the
    heterogeneity from being detected.

    **Why not a binary flag.** The first version only flagged "same
    direction / not", and a teammate pointed out that on purely random data
    this necessarily fires -- with 12 targets and a moderate effect, the
    expectation is already 8/12 or 9/12, **so it lights up for both a real
    effect and no effect at all, and a reader cannot use it to distinguish
    anything**. Binarising throws away the information carried by the
    underlying counts, the same shape of mistake as a ratio throwing away
    the two raw numbers. Switched to reporting k/n plus its exact binomial p.

    **Why it counts "how many are above the null" rather than "how many
    agree with the mean's direction".** The latter lets the data pick its own
    direction and then tests that direction, which is a mild double dip (k
    gets systematically inflated: the mean's direction is usually already
    the majority direction). Counting "above the null" fixes the direction in
    advance, which is the standard sign test, with a clean null distribution
    of Binomial(n, 0.5). The cost is that it measures "which side the
    majority is on" rather than "is the mean's direction trustworthy" -- but
    it is exactly the former that answers "is there a common effect".
    """
    v = np.array([x for x in vals if np.isfinite(x)], dtype=float)
    v = v[v != null]                       # ties dropped, the standard sign-test convention
    n = len(v)
    if n == 0:
        return 0, 0, float("nan")
    k = int((v > null).sum())
    p = float(stats.binomtest(k, n, 0.5).pvalue)   # two-sided
    return k, n, p


def sign_note(k, n, p):
    """One line: k/n + binomial p + whether this line's summary should be trusted.

    Warning: **the sign test deliberately discards magnitude**: an AUROC of
    0.95 and one of 0.51 each count as one vote. So it can only decide
    "whether this line's summary can be quoted on its own", **it cannot be
    used the other way around as evidence of 'no effect'**. Counter-example:
    per-target values 0.72/0.68/0.65/0.61/0.58/0.55/0.53/0.51/0.47/0.45/0.42/0.38
    are 8/12, p=0.39, which the sign test calls noise, but the positive half
    is clearly stronger and the per-target table shows at a glance that it is
    not noise. **Judging the effect always defers to the per-target table**,
    which is why that table is printed before the summary line.

    Warning: when n is very small the sign test **cannot reach 0.05 at all**
    (at n=5, even 5/5 only gives 0.0625). In that case "for reference only"
    means **the test itself has no power**, not "evidence of per-target
    heterogeneity" -- the two read the same but mean the opposite, so they
    must be stated separately, otherwise "cannot be tested" gets misread as
    "was tested, and came out scattered".
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
    """The Wilcoxon two-sided p floor at n units. Must be stated if it never reaches 0.05."""
    f = 2 / 2 ** n if n > 0 else float("nan")
    tail = "，够不到 0.05" if f > 0.05 else ""
    return f"n={n}，双侧 p 下界 {f:.4f}{tail}"


def wilcoxon_vs(vals, null):
    """Per-target Wilcoxon against a constant null hypothesis.

    Warning: the returned n is **after ties are dropped**. scipy's wilcoxon
    already drops zero differences by default (zero_method="wilcox"), so the
    effective sample size is already the post-drop one; using the pre-drop n
    to compute the p floor 2/2^n would make the floor look smaller -- i.e.
    more powerful than it actually is. Two of the 12 targets tie, so n is
    really 10, and the floor moves from 0.0005 to 0.002.
    """
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    k = int((v > null).sum())
    v = v[v != null]                       # ties dropped, matching sign_test's convention
    if len(v) < 5 or np.allclose(v, null):
        return float("nan"), k, len(v)
    return float(stats.wilcoxon(v - null).pvalue), k, len(v)


def quarantined():
    """Records that failed the structure stage and were moved out of the input directory (the list is read from disk, not hardcoded).

    When Boltz's affinity stage hits a record with no `pre_affinity_*.npz`, it
    does not skip it -- it **exits entirely**, and a single bad record can
    take down an entire shard -- that is how three shards crashed last round.
    So these records get moved into a quarantine directory before resuming.
    """
    d = f"{B}/boltz_rerank_sub_quarantine"
    if not os.path.isdir(d):
        return set()
    return {f[:-5] for f in os.listdir(d) if f.endswith(".yaml")}


def coverage_gate(man, aff, excl):
    """Per-target completion rate. **This is the biggest lesson this script ever taught.**

    The previous version reported "scored 2,562/3,747" as a one-line coverage
    footnote and shipped the numbers. The mistake: this round shards by
    **complex**, and all four shards each cover all 12 targets, so a crashed
    shard does not remove a few complete targets -- it **shaves a third off
    every single target** (measured per-target completion rate: median 68.7%,
    0/12 complete). AUROC computed on two-thirds of a target's candidates is
    not the full-set value with a wider error bar -- it is a different
    quantity; P@5 with a third of the members missing cannot be interpreted
    at all.

    So: **reject the main conclusion if any target is incomplete** -- there is
    no "make do with what's there" path offered.

    Warning: **the only exception is the named records in excl, and it is a
    list, not a threshold.** Never relax the bar from 100% to "98% is good
    enough" -- that would just reinstall the "make do" path just removed.
    Instead: list the structure-stage failures **record by record**, state
    clearly what they are, and then require **every remaining record to have
    a score**. The list is read from the quarantine directory on disk, not
    hardcoded.
    """
    by = {}
    for e in man["entries"]:
        d = by.setdefault(e["uniprot"], {"tot": 0, "got": 0, "ex": 0, "ex_act": 0})
        if e["name"] in excl:
            d["ex"] += 1
            d["ex_act"] += int(e["label"] == 1)
            continue
        d["tot"] += 1
        d["got"] += int(e["name"] in aff)
    print("逐靶点完成率（分母已剔除结构阶段失败的具名记录）")
    bad = []
    for up, d in sorted(by.items()):
        f = d["got"] / d["tot"] if d["tot"] else 0.0
        ex = f"  (另有 {d['ex']} 条结构失败已剔除)" if d["ex"] else ""
        flag = "" if f >= 0.999 else "  ← 不完整"
        if f < 0.999:
            bad.append((up, d["got"], d["tot"]))
        print(f"  {up:10} {d['got']:5d} / {d['tot']:5d}  {f:6.1%}{ex}{flag}")
    n_ex = sum(d["ex"] for d in by.values())
    n_ex_act = sum(d["ex_act"] for d in by.values())
    if n_ex:
        print(f"\n⚠️ 结构阶段失败、已剔除：{n_ex} 条（占 manifest 的 "
              f"{n_ex / len(man['entries']):.2%}），其中活性 {n_ex_act} 条。")
        if n_ex_act == 0:
            print("   **全部是诱饵，活性一条没丢**——所以 missed/found 两组的分子是完整的，")
            print("   受影响的只有 AUROC 负类的分母（任一靶点最多丢 3 个诱饵，<1.5%）。")
            print("   这和「每个靶点被咬掉三分之一」是两回事：那个换掉了要测的量，")
            print("   这个在任一靶点上最多移动 AUROC 约 1/200。**但它仍是限制，要写进文档。**")
        else:
            print(f"   ⚠️ 其中有 {n_ex_act} 条活性——这会直接影响 missed/found 的分子，")
            print("   必须逐条检查它们属于哪一组，不能当成可忽略的损失。")
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

    excl = quarantined()
    bad = coverage_gate(man, aff, excl)
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
        if e["name"] in excl:
            continue
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
        items = [e for e in items if e["name"] not in excl]
        lab = np.array([e["label"] for e in items], dtype=int)
        rnk = np.array([e["rank"] for e in items], dtype=int)
        if lab.sum() < 3 or lab.sum() == len(lab):
            continue
        # higher is better for the retrieval score; lower is better for Boltz's affinity_pred_value (predicted log Kd)
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
    # warning: the per-target table comes **before** the summary line, not as an appendix. When summary and per-target disagree, defer to per-target.
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
    print("⚠️ 判断效应以上表为准。下面的汇总行和符号检验只决定「这一行能不能单独")
    print("   引用」——符号检验只看方向不看幅度，不能反过来当「没有效应」的证据。")

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
        # warning: at n=5 the Wilcoxon two-sided p floor is 2/2^5=0.0625, which never reaches 0.05.
        #    Reporting a p-value that can never be significant would mislead, so the floor is printed alongside it.
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
