"""Make the T1 branch save its scores, for LigUnity and LiTENCLIP.

The gap
-------
Seven of the ten models in T1_main.csv ship `saved_preds.npy` per target, so
their DUD-E / DEKOIS / LIT-PCBA numbers can be recomputed from the release by
anyone. Three cannot: LigUnity-pocket, LigUnity-protein and LiTENCLIP. Upstream
stores the two embedding matrices and the labels, but not the score -- the same
omission `patch_ligunity_t3.py` already works around on the T3 path ("the
official code only stores the embedding, not the score"). The T1 path never got
the same treatment.

Why not just recompute the score from the saved embeddings
----------------------------------------------------------
Because it does not reproduce EF@1%, which is T1's headline metric. The score
is `(pocket_reps @ mol_reps.T).max(axis=0)`, and both matrices are on disk --
but they are stored as **float16**. Reconstructing from them reproduces AUROC,
BEDROC and EF@5% to four decimals exactly, while EF@1% lands ~0.03 away: at
k = ceil(0.01 * 1240) = 13, the metric is decided by a handful of molecules at
the boundary, and float16 is too coarse to place them the same way. Neither
float16 nor float32 matmul, under any of ceil/floor/round for k, reproduces the
published mean. `timesplit/analysis/rebuild_t1_preds.py` is the script that
establishes this, and it is worth keeping as a consistency check on the
embeddings; it is not a substitute for the real scores.

What this patch does
--------------------
Inserts one `np.save(.../saved_preds.npy, res_single)` into each of
test_dude_target / test_dekois_target / test_pcba_target, next to the saves
already there. Nothing else is touched -- no forward logic, no metric call --
so a re-run reproduces the published table rather than redefining it.

⚠️ The three functions do not agree on ordering: DUD-E and DEKOIS compute
`res_single` *before* writing their files, PCBA writes first and computes
after. So the insertion point is "after whichever of the two comes later",
not a fixed offset. Getting this wrong yields a NameError at run time on PCBA
only -- i.e. after DUD-E and DEKOIS have already burned GPU time.
"""
import argparse
import os
import re

B = "/data/work/vs-benchmark"

TARGETS = {
    "test_dude_target": "DUDE",
    "test_dekois_target": "DEKOIS",
    "test_pcba_target": "PCBA",
}
RES_LINE = re.compile(r"^(\s*)res_single = res\.max\(axis=0\)\s*$", re.M)
LABELS_LINE = re.compile(
    r'^(\s*)np\.save\(f"\{self\.args\.results_path\}/(\w+)/\{(\w+)\}/saved_labels\.npy".*$',
    re.M)


def body_of(src, fname):
    """Span of one method: its def line to the next def at the same indent."""
    m = re.search(rf"\n(\s*)def {fname}\(", src)
    if not m:
        return None
    start = m.start() + 1
    nxt = re.search(rf"\n{m.group(1)}def ", src[m.end():])
    end = m.end() + nxt.start() + 1 if nxt else len(src)
    return start, end


def patch_one(src, fname):
    span = body_of(src, fname)
    if span is None:
        return src, f"{fname}: 找不到该方法"
    a, b = span
    body = src[a:b]
    if "saved_preds.npy" in body:
        return src, f"{fname}: 已有 saved_preds，跳过"

    mres = RES_LINE.search(body)
    mlab = LABELS_LINE.search(body)
    if not mres:
        return src, f"{fname}: 找不到 res_single 计算行"
    if not mlab:
        return src, f"{fname}: 找不到 saved_labels 存盘行"

    indent, bm, var = mlab.group(1), mlab.group(2), mlab.group(3)
    # Insert after whichever comes later: the score must exist, and the
    # directory must already have been created by the block above.
    ins_at = max(mres.end(), mlab.end())
    line = (f'\n{indent}np.save(f"{{self.args.results_path}}/{bm}/{{{var}}}/'
            f'saved_preds.npy", res_single)'
            f'  # 本项目新增：官方只存嵌入不存打分')
    body = body[:ins_at] + line + body[ins_at:]
    which = "res_single 之后" if mres.end() > mlab.end() else "saved_labels 之后"
    return src[:a] + body + src[b:], f"{fname}: 已插入（{which}，{bm}）"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="+", default=["LigUnity", "LiTENCLIP"])
    ap.add_argument("--dry-run", action="store_true",
                    help="只报告会改什么，不落盘")
    args = ap.parse_args()

    for r in args.repos:
        p = f"{B}/code/{r}/unimol/tasks/test_task.py"
        print(f"[{r}] {p}")
        if not os.path.exists(p):
            print("  文件不存在")
            continue
        src = open(p).read()
        orig = src
        for fn in TARGETS:
            src, msg = patch_one(src, fn)
            print(f"  {msg}")
        if src == orig:
            print("  无改动")
            continue
        if args.dry_run:
            print("  [dry-run] 未写入")
            continue
        bak = p + ".bak_t1preds"
        if not os.path.exists(bak):
            open(bak, "w").write(orig)
            print(f"  备份 -> {bak}")
        open(p, "w").write(src)
        print("  已写入")


if __name__ == "__main__":
    main()
