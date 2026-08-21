"""T1 靶点身份改用 target_id（基准里的名字），因为有五处 UniProt 撞车。

撞车实例（后一个会覆盖前一个，DEKOIS 81 个只落盘 78 个就是这么来的）
  DUD-E    P11362 → CSF1R, FGFR1        （dude.json 把 CSF1R 标错，应是 P07333）
  DEKOIS   P03366 → HIV1PR, HIV1RT      （同一条 pol 多聚蛋白切出的两个酶）
           P19793 → PPARG, RXR          （PPARG 应是 P37231，也是标错）
           P06737 → PYGL-IN, PYGL-OUT   （同一蛋白的两个不同结合位点）
  LIT-PCBA P03372 → ESR1_ago, ESR1_ant  （同一蛋白，激动/拮抗两种构象）

原则：**身份用 target_id，查序列/结构仍用 uniprot**——
同一个蛋白的两个位点本来就该用同一条序列和同一个结构。
另外 PYGL-IN / PYGL-OUT 这种情况，纯序列模型原理上就分不开，
这是基准自身的性质，不是模型的问题，报告时要写明。
"""
import json
import shutil

B = "/data/work/vs-benchmark"

for bench in ["DUDE", "DEKOIS", "PCBA"]:
    p = f"{B}/data/t1/{bench}.jsonl"
    rows = [json.loads(l) for l in open(p)]
    for r in rows:
        r["target_id"] = r["name"]
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{bench}: {len(rows)} 条已补 target_id")

# ---- ConPLex：TSV 里的行标识与落盘目录都改成 target_id ----
P = f"{B}/run_t3_conplex.py"
s = open(P).read()
subs = [
    ('''        for r in usable:
            up = r["uniprot"]
            v = seqs[up]''',
     '''        for r in usable:
            up = r["uniprot"]
            # 身份用 target_id（T1 有多个靶点共用一个 UniProt），查序列仍用 uniprot
            tid = r.get("target_id") or up
            v = seqs[up]'''),
    ('''                    mid = f"{up}_{kind[0]}{i}"
                    f.write(f"{up}\\t{mid}\\t{seq}\\t{m['smiles']}\\n")
                    index.append((up, mid, lab))''',
     '''                    mid = f"{tid}_{kind[0]}{i}"
                    f.write(f"{tid}\\t{mid}\\t{seq}\\t{m['smiles']}\\n")
                    index.append((tid, mid, lab))'''),
]
n = 0
for a, b in subs:
    if a in s:
        s = s.replace(a, b, 1)
        n += 1
    else:
        print(f"⚠️ conplex 没匹配: {a.splitlines()[0][:50]}")
if n:
    shutil.copy(P, P + ".tid.bak")
    open(P, "w").write(s)
print(f"conplex: 改了 {n} 处")

# ---- ConGLUDe：protein_ids 清单与落盘也用 target_id，结构文件名仍按 uniprot 找 ----
P = f"{B}/run_t3_conglude.py"
s = open(P).read()
subs = [
    ('''        for r in recs:
            up = r["uniprot"]
            src = f"{struct_dir}/{up}.pdb"
            if os.path.exists(src) and os.path.getsize(src) > 1000:
                shutil.copyfile(src, f"{pdbdir}/{up}.pdb")
                used[up] = {"kind": "local_pdb"}
            else:
                missing.append(up)''',
     '''        for r in recs:
            up, tid = r["uniprot"], (r.get("target_id") or r["uniprot"])
            src = f"{struct_dir}/{up}.pdb"
            if os.path.exists(src) and os.path.getsize(src) > 1000:
                # 结构按 UniProt 存，但拷成 target_id 命名：
                # 同一蛋白的两个位点共用结构，身份却必须分开
                shutil.copyfile(src, f"{pdbdir}/{tid}.pdb")
                used[tid] = {"kind": "local_pdb", "uniprot": up}
            else:
                missing.append(tid)'''),
    ('''    ups = [r["uniprot"] for r in recs if r["uniprot"] in used]''',
     '''    ups = [(r.get("target_id") or r["uniprot"]) for r in recs
           if (r.get("target_id") or r["uniprot"]) in used]'''),
    ('''    smi_set, per_target = {}, {}
    for r in recs:
        if r["uniprot"] not in used:
            continue''',
     '''    smi_set, per_target = {}, {}
    for r in recs:
        tid = r.get("target_id") or r["uniprot"]
        if tid not in used:
            continue'''),
    ('''        per_target[r["uniprot"]] = lab''', '''        per_target[tid] = lab'''),
]
n = 0
for a, b in subs:
    if a in s:
        s = s.replace(a, b, 1)
        n += 1
    else:
        print(f"⚠️ conglude 没匹配: {a.splitlines()[0][:50]}")
if n:
    shutil.copy(P, P + ".tid.bak")
    open(P, "w").write(s)
print(f"conglude: 改了 {n} 处")
