"""让 run_t3_conplex.py 也能跑 T1（三个标准基准），不必再写一个 runner。

改动只有两处：
  · 评测集路径与序列文件改成可传参（--eval_dir / --seqs），默认仍是 T3
  · 序列文件两种结构都认：T3 的是 {up: {"seq": ...}}，T1 的是 {up: "序列"}
其余逻辑（写 tsv、调 conplex-dti、按靶点切分落盘）一个字不动，
保证 T1 和 T3 的打分口径完全一致——这正是统一评测的前提。
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
