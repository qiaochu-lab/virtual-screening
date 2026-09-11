"""Prepare .pdb structures for T1's three benchmarks -- needed by ConGLUDe,
and by SPRINT's 3Di.

Two sources
----------
· DUD-E: every target directory already has a receptor.pdb; symlink it
  directly, no download needed
· DEKOIS / LIT-PCBA: only lmdb is available, no structure file; fetch from
  RCSB using the PDB ID in dekois.json / PCBA.json

Naming is unified to {UniProt}.pdb -- matching T3's convention, so the same
target uses the same identifier across both eval sets and tables can be
joined later without misalignment.

⚠️ RCSB does not serve the legacy PDB format for very large structures
(mmCIF only); unavailable ones are recorded. ConGLUDe only accepts .pdb, so
these targets are simply absent, the same treatment T3 uses.
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
                # copy rather than symlink: SPRINT's foldseek is occasionally picky about symlinks
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
