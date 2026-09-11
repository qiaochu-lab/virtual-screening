"""Fix the hardcoded bsz=64 in test_t3_target.

Copying code over from DEKOIS also copied `bsz = 64` along with it, so
the command line's --batch-size had no effect at all (this is what
surfaced the bug: setting it to 8 didn't change the OOM allocation size
one bit). Changed to read args.batch_size instead.
"""
B = "/data/work/vs-benchmark"
for repo in ["DrugCLIP", "BindCLIP"]:
    p = f"{B}/code/{repo}/unimol/tasks/drugclip.py"
    s = open(p).read()
    i = s.find("def test_t3_target")
    if i < 0:
        print(f"{repo}: 没有 test_t3_target"); continue
    j = s.find("def test_t3(", i)
    seg = s[i:j]
    if "bsz = 64" not in seg:
        print(f"{repo}: 段内没有 bsz = 64（可能已改）"); continue
    seg2 = seg.replace("        bsz = 64\n",
        "        # 必须读命令行参数：T3 的分子最大 336 个原子（DEKOIS 才 50），\n"
        "        # UniMol 注意力是 O(n^2)，沿用 DEKOIS 写死的 64 会 CUDA OOM\n"
        "        bsz = int(getattr(self.args, \"batch_size\", 8) or 8)\n")
    s = s[:i] + seg2 + s[j:]
    open(p, "w").write(s)
    print(f"{repo}: bsz 已改为读 args.batch_size")
