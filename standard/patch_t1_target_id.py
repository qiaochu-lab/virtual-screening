"""Switch T1 target identity to target_id (the benchmark's own name),
because five UniProt IDs collide across targets.

Collision instances (the later one overwrites the earlier one -- this is
why DEKOIS's 81 targets only ever write 78 to disk)
  DUD-E    P11362 -> CSF1R, FGFR1        (dude.json mislabels CSF1R; should be P07333)
  DEKOIS   P03366 -> HIV1PR, HIV1RT      (two enzymes cleaved from the same pol polyprotein)
           P19793 -> PPARG, RXR          (PPARG should be P37231, also mislabeled)
           P06737 -> PYGL-IN, PYGL-OUT   (two different binding sites on the same protein)
  LIT-PCBA P03372 -> ESR1_ago, ESR1_ant  (same protein, agonist/antagonist conformations)

Principle: **identity uses target_id; sequence/structure lookup still uses
uniprot** -- two sites on the same protein should legitimately share the
same sequence and the same structure. Also, a case like PYGL-IN / PYGL-OUT
is something a sequence-only model cannot distinguish in principle -- that
is a property of the benchmark itself, not of the model, and must be stated
when reporting.
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

# ---- ConPLex: switch both the TSV row identifier and the output directory to target_id ----
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

# ---- ConGLUDe: use target_id for the protein_ids manifest and output too; structure filenames are still looked up by uniprot ----
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
