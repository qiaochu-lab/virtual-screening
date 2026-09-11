"""Let run_t3_conplex.py also run T1 (the three standard benchmarks),
instead of writing a separate runner.

Only two changes:
  · the eval-set path and sequence file become CLI-configurable
    (--eval_dir / --seqs), still defaulting to T3
  · both sequence-file shapes are accepted: T3's is {up: {"seq": ...}},
    T1's is {up: "sequence"}
Everything else (writing the tsv, calling conplex-dti, splitting output by
target) is left untouched, so T1 and T3 stay on exactly the same scoring
convention -- which is the whole premise of a unified eval layer.
"""
import shutil

P = "/data/work/vs-benchmark/run_t3_conplex.py"
s = open(P).read()
subs = [
    ('def run_layer(layer, out_dir, work_dir, seqs, max_len, limit=None):\n'
     '    recs = [json.loads(l) for l in open(f"{B}/data/t3/eval/{layer}.jsonl")]',
     'def run_layer(layer, out_dir, work_dir, seqs, max_len, limit=None, eval_dir=None):\n'
     '    eval_dir = eval_dir or f"{B}/data/t3/eval"\n'
     '    recs = [json.loads(l) for l in open(f"{eval_dir}/{layer}.jsonl")]'),
    ('        s = (seqs.get(r["uniprot"]) or {}).get("seq")',
     '        v = seqs.get(r["uniprot"])\n'
     '        # T3 的序列表是 {up: {"seq": ...}}，T1 的是 {up: "序列"}，两种都认\n'
     '        s = v.get("seq") if isinstance(v, dict) else v'),
    ('    ap.add_argument("--limit", type=int, default=None, help="每层只跑前 N 个靶点（调试用）")',
     '    ap.add_argument("--limit", type=int, default=None, help="每层只跑前 N 个靶点（调试用）")\n'
     '    ap.add_argument("--eval_dir", default=None, help="评测集目录，默认 data/t3/eval")\n'
     '    ap.add_argument("--seqs", default=None, help="序列 json，默认 data/t3/sequences.json")'),
    ('    seqs = json.load(open(f"{B}/data/t3/sequences.json"))',
     '    seqs = json.load(open(args.seqs or f"{B}/data/t3/sequences.json"))\n'
     '    eval_dir = args.eval_dir or f"{B}/data/t3/eval"'),
    ('        if not os.path.exists(f"{B}/data/t3/eval/{L}.jsonl"):',
     '        if not os.path.exists(f"{eval_dir}/{L}.jsonl"):'),
    ('        run_layer(L, args.out_dir, args.work_dir, seqs, args.max_len, args.limit)',
     '        run_layer(L, args.out_dir, args.work_dir, seqs, args.max_len, args.limit,\n'
     '                  eval_dir=eval_dir)'),
]
ok = True
for a, b in subs:
    if a not in s:
        print(f"⚠️ 没匹配上: {a.splitlines()[0][:60]}")
        ok = False
        continue
    s = s.replace(a, b, 1)
if ok:
    shutil.copy(P, P + ".t1.bak")
    open(P, "w").write(s)
    print("run_t3_conplex.py 已支持 --eval_dir / --seqs")
