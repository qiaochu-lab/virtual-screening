"""Let run_t3_conglude.py also run T1, reusing the same scoring and
disk-write logic.

On the T3 side, structures are fetched live from RCSB and fall back to a
Boltz-2 prediction when unavailable; on the T1 side, all 149 targets'
.pdb files are already prepared (data/t1/structures/{uniprot}.pdb), so a
--struct_dir flag is added: when given, structures are taken straight from
that directory, with no download and no fallback.

Everything else (writing protein_ids.txt / smiles.txt, calling predict.py,
splitting output by target) is left untouched, so T1 and T3 stay on the
same convention.
"""
import shutil

P = "/data/work/vs-benchmark/run_t3_conglude.py"
s = open(P).read()

subs = [
    ("def prepare(layer, recs, ds_dir, boltz_idx, pdb_choice, workers):",
     "def prepare(layer, recs, ds_dir, boltz_idx, pdb_choice, workers, struct_dir=None):"),
    ("""    # 先并发下载实验结构，拿不到的再回退预测结构""",
     """    # struct_dir：结构已经在本地备齐（T1 走这条），直接拷，不下载也不回退
    if struct_dir:
        used, missing = {}, []
        for r in recs:
            up = r["uniprot"]
            src = f"{struct_dir}/{up}.pdb"
            if os.path.exists(src) and os.path.getsize(src) > 1000:
                shutil.copyfile(src, f"{pdbdir}/{up}.pdb")
                used[up] = {"kind": "local_pdb"}
            else:
                missing.append(up)
        return _finish(layer, recs, info, used, missing)

    # 先并发下载实验结构，拿不到的再回退预测结构"""),
    ("""    ups = [r["uniprot"] for r in recs if r["uniprot"] in used]
    open(f"{info}/protein_ids.txt", "w").write("\\n".join(ups) + "\\n")""",
     """    return _finish(layer, recs, info, used, missing)


def _finish(layer, recs, info, used, missing):
    \"\"\"两条取结构的路径汇合到这里：写清单、收集唯一分子、回填标签。\"\"\"
    ups = [r["uniprot"] for r in recs if r["uniprot"] in used]
    open(f"{info}/protein_ids.txt", "w").write("\\n".join(ups) + "\\n")"""),
    ("""    ap.add_argument("--limit", type=int, default=None)""",
     """    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--eval_dir", default=None, help="评测集目录，默认 data/t3/eval")
    ap.add_argument("--struct_dir", default=None,
                    help="现成 .pdb 目录（{uniprot}.pdb）；给了就不下载不回退")
    ap.add_argument("--tag", default="t3", help="数据集目录前缀，避免 T1/T3 相互覆盖")"""),
    ("""        p = f"{B}/data/t3/eval/{L}.jsonl\"""",
     """        p = f"{args.eval_dir or (B + '/data/t3/eval')}/{L}.jsonl\""""),
    ("""        ds_rel = f"./data/datasets/predict_datasets/t3_{L}"
        ds = f"{CG}/data/datasets/predict_datasets/t3_{L}\"""",
     """        ds_rel = f"./data/datasets/predict_datasets/{args.tag}_{L}"
        ds = f"{CG}/data/datasets/predict_datasets/{args.tag}_{L}\""""),
    ("""        ups, per_target, used, missing = prepare(L, recs, ds, boltz_idx, pdb_choice, args.workers)""",
     """        ups, per_target, used, missing = prepare(L, recs, ds, boltz_idx, pdb_choice,
                                                 args.workers, args.struct_dir)"""),
    ("""    man = json.load(open(f"{B}/data/t3/pockets/pdb_pocket_manifest.json"))["manifest"]
    pdb_choice = {u: v["pdb_id"] for u, v in man.items()}
    boltz_idx = build_boltz_index()""",
     """    if args.struct_dir:      # T1：结构现成，不需要这两张索引
        pdb_choice, boltz_idx = {}, {}
    else:
        man = json.load(open(f"{B}/data/t3/pockets/pdb_pocket_manifest.json"))["manifest"]
        pdb_choice = {u: v["pdb_id"] for u, v in man.items()}
        boltz_idx = build_boltz_index()"""),
]
ok = True
for a, b in subs:
    if a not in s:
        print(f"⚠️ 没匹配: {a.splitlines()[0][:70]}")
        ok = False
        continue
    s = s.replace(a, b, 1)
if ok:
    shutil.copy(P, P + ".t1.bak")
    open(P, "w").write(s)
    print("run_t3_conglude.py 已支持 --eval_dir / --struct_dir / --tag")
