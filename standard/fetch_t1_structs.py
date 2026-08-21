"""给 T1 三个基准准备 .pdb 结构 —— ConGLUDe 要，SPRINT 的 3Di 也要。

来源分两种
----------
· DUD-E：每个靶点目录里本来就有 receptor.pdb，直接软链，不下载
· DEKOIS / LIT-PCBA：只有 lmdb，没有结构文件，按 dekois.json / PCBA.json
  里的 PDB ID 从 RCSB 下

命名统一成 {UniProt}.pdb —— 和 T3 那边一致，保证同一个靶点在两套评测里
用的是同一个标识，后面合表不会错位。

⚠️ RCSB 的传统 PDB 格式对超大结构不提供（只有 mmCIF），拿不到的会记下来；
ConGLUDe 只认 .pdb，这类靶点只能缺席，和 T3 的处理一致。
"""
import json
import os
import time
import urllib.request

B = "/data/work/vs-benchmark"
TD = f"{B}/code/LigUnity/test_datasets"
OUT = f"{B}/data/t1/structures"


def dude_local(name):
    p = f"{TD}/DUD-E/{name.lower()}/receptor.pdb"
    return p if os.path.exists(p) else None


def rcsb(pdb, dest, retry=3):
    url = f"https://files.rcsb.org/download/{pdb.upper()}.pdb"
    for i in range(retry):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            if len(data) < 1000:
                return False
            open(dest, "wb").write(data)
            return True
        except Exception:
            if i == retry - 1:
                return False
            time.sleep(2 * (i + 1))


def main():
    os.makedirs(OUT, exist_ok=True)
    stats = {}
    for bench, jf in [("DUDE", "dude.json"), ("DEKOIS", "dekois.json"), ("PCBA", "PCBA.json")]:
        rows = json.load(open(f"{TD}/{jf}"))
        ok = local = fetched = miss = 0
        missing = []
        for up, pdb, name in rows:
            dest = f"{OUT}/{up}.pdb"
            if os.path.exists(dest) and os.path.getsize(dest) > 1000:
                ok += 1
                continue
            src = dude_local(name) if bench == "DUDE" else None
            if src:
                # 复制而不是软链：SPRINT 那边 foldseek 对软链偶尔挑剔
                open(dest, "wb").write(open(src, "rb").read())
                local += 1
                ok += 1
                continue
            if rcsb(pdb, dest):
                fetched += 1
                ok += 1
                time.sleep(0.3)
            else:
                miss += 1
                missing.append((up, pdb, name))
        stats[bench] = (len(rows), ok, local, fetched, miss, missing)
        print(f"{bench}: {ok}/{len(rows)} 有结构"
              f"（本地 receptor.pdb {local}，RCSB 下载 {fetched}，缺 {miss}）")
        for up, pdb, name in missing[:5]:
            print(f"    缺: {name} {up} {pdb}")

    json.dump({b: {"total": v[0], "ok": v[1], "local": v[2], "fetched": v[3],
                   "missing": v[5]} for b, v in stats.items()},
              open(f"{B}/data/t1/structure_manifest.json", "w"), indent=1)
    print(f"\n结构目录: {OUT}")


if __name__ == "__main__":
    main()
