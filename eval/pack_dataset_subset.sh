#!/bin/bash
# Pack the 350-quota subset's pocket + ligand data for a public deposit.
#
# Split by layer, because L2 alone is most of the bulk and few people need all
# four. Each archive is self-describing: the manifest records every record the
# subset list asks for, including the ones that have no directory, so a consumer
# can tell "absent from the deposit" from "absent from the subset".
#
# ⚠️ 13 of the 328 records have no directory under data/T3_6A. That is the known
# pocket-extraction gap (95.4% coverage), not something this script introduces.
# They are listed in the manifest as missing rather than silently dropped.
set -u
B=/data/work/vs-benchmark
SUBSET=$B/results/export/T3_vsds_matched.csv
OUT=${1:?usage: $0 <out_dir>}
mkdir -p "$OUT"

python3 - "$SUBSET" "$B" "$OUT" <<'PY'
import csv, json, os, subprocess, sys, hashlib
subset, B, out = sys.argv[1], sys.argv[2], sys.argv[3]
rows = list(csv.DictReader(open(subset)))
by_layer = {}
manifest = {"subset_file": os.path.basename(subset), "n_records": len(rows),
            "layers": {}, "missing": []}
for r in rows:
    by_layer.setdefault(r["layer"], []).append(r["uniprot"])

for layer, ups in sorted(by_layer.items()):
    present, missing = [], []
    for up in sorted(ups):
        (present if os.path.isdir(f"{B}/data/T3_6A/{layer}/{up}") else missing).append(up)
    manifest["missing"] += [f"{layer}/{u}" for u in missing]
    if not present:
        continue
    lst = f"{out}/.{layer}.files"
    with open(lst, "w") as f:
        for up in present:
            f.write(f"{layer}/{up}\n")
    tar = f"{out}/T3_6A_{layer}.tar.gz"
    print(f"  {layer}: {len(present)} targets -> {os.path.basename(tar)}", flush=True)
    subprocess.run(["tar", "-czf", tar, "-C", f"{B}/data/T3_6A",
                    "-T", lst], check=True)
    os.remove(lst)
    h = hashlib.sha256()
    with open(tar, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    manifest["layers"][layer] = {
        "file": os.path.basename(tar), "targets": len(present),
        "size_mb": round(os.path.getsize(tar) / 1e6, 1), "sha256": h.hexdigest()}

json.dump(manifest, open(f"{out}/manifest_data.json", "w"), indent=1)
tot = sum(v["size_mb"] for v in manifest["layers"].values())
print(f"\n{len(manifest['layers'])} archives, {tot:.0f} MB")
print(f"{len(manifest['missing'])} records in the subset have no directory: "
      f"{manifest['missing']}")
PY
