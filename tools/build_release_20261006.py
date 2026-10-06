#!/usr/bin/env python3
"""Build the canonical 2026-10-06 result set from committed result tables.

The repository contains supporting and historical analyses whose model panels
are intentionally broader.  This script applies the task-specific checkpoint
policy used by the final release and writes only the tables that may be cited as
the canonical result set.
"""

from __future__ import annotations

import csv
import collections
import hashlib
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "release-2026-10-06"

SCREENING_T1 = {
    "bindclip_hardneg",
    "bindclip_randneg",
    "conglude",
    "conplex",
    "drugclip",
    "drugjepa",
    "hypseek_official_vs",
    "ligunity_pocket",
    "ligunity_protein",
    "litenclip",
    "sprint",
}

SCREENING_T3 = {
    "bindclip_hardneg",
    "bindclip_randneg",
    "conglude",
    "conplex",
    "drugclip",
    "drugjepa",
    "hypseek_official_vs",
    "ligunity_pocket_ranking",
    "ligunity_protein_ranking",
    "litenclip",
    "sprint",
}

RANKING_T3 = (SCREENING_T3 - {"hypseek_official_vs"}) | {"hypseek_rk"}
TARGET_SWAP_T3 = RANKING_T3


def select(source: str, destination: str, models: set[str], *, layer: str | None = None) -> None:
    src = RESULTS / source
    dst = OUT / destination
    with src.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = [
            row
            for row in reader
            if row["model"] in models and (layer is None or row.get("layering") == layer)
        ]
        fieldnames = reader.fieldnames
    if not fieldnames or not rows:
        raise RuntimeError(f"selection produced no rows: {source}")
    with dst.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def copy(source: str, destination: str | None = None) -> None:
    src = RESULTS / source
    dst = OUT / (destination or source)
    if src.suffix == ".csv":
        dst.write_text(src.read_text(encoding="utf-8-sig"), encoding="utf-8")
    else:
        shutil.copyfile(src, dst)


def build_strict_targets() -> None:
    """Write the strict target list with the final layer labels, not legacy labels."""
    with (RESULTS / "T3_target_exposure.csv").open(newline="", encoding="utf-8-sig") as handle:
        exposure_rows = list(csv.DictReader(handle))
    strict_layer = {
        (row["uniprot"], row["layer_original"]): row["layer_strict"]
        for row in exposure_rows
    }

    with (RESULTS / "T3_vsds_matched.csv").open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = []
        for row in reader:
            layer = strict_layer[(row["uniprot"], row["layer"])]
            if layer not in {"L1", "L2", "L3", "L4"}:
                continue
            row["layer"] = layer
            rows.append(row)

    counts = collections.Counter(row["layer"] for row in rows)
    expected = {"L1": 56, "L2": 178, "L3": 20, "L4": 41}
    if counts != expected:
        raise RuntimeError(f"strict target counts changed: {dict(counts)} != {expected}")

    with (OUT / "T3_targets_strict.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.iterdir():
        if old.is_file():
            old.unlink()

    select("T1_main.csv", "T1_main.csv", SCREENING_T1)
    select(
        "T3_main_vsds_subset.csv",
        "T3_main_strict.csv",
        SCREENING_T3,
        layer="strict",
    )
    select("T2_on_T3_subset_strict.csv", "T2_on_T3_strict.csv", RANKING_T3)
    select("T3_recall_at_k_strict.csv", "T3_recall_at_k_strict.csv", SCREENING_T3)
    select(
        "T3_novelty_tiered_ef_strict.csv",
        "T3_novelty_tiered_ef_strict.csv",
        SCREENING_T3,
    )
    select(
        "T3_chemical_memory_pref_strict.csv",
        "T3_chemical_memory_pref_strict.csv",
        SCREENING_T3,
    )
    select("T3_target_swap_strict.csv", "T3_target_swap_strict.csv", TARGET_SWAP_T3)
    select(
        "T3_target_swap_family_strict.csv",
        "T3_target_swap_family_strict.csv",
        TARGET_SWAP_T3,
    )
    select("T5_structure_source_strict.csv", "T5_structure_source_strict.csv", SCREENING_T3)
    build_strict_targets()
    copy("T3_target_exposure.csv")

    files = sorted(path for path in OUT.iterdir() if path.is_file())
    with (OUT / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in files:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            handle.write(f"{digest}  {path.name}\n")


if __name__ == "__main__":
    main()
