#!/usr/bin/env python3
"""Inventory the preprocessed image tree so two copies can be compared.

Standard library only — runs with the system python3 on a Mac, on rikyu's
login node and on raiden alike (no uv, no repo needed).

  # on each machine: one line per file, "<relpath>\t<bytes>", plus a summary
  python3 scripts/data_inventory.py inventory <data_dir> -o inventory_mac.tsv
  python3 scripts/data_inventory.py inventory /data1/rkp00078/satoc/Glioma/images/v1.0.0 -o inventory_rikyu.tsv

  # then, with both files on one machine:
  python3 scripts/data_inventory.py compare inventory_mac.tsv inventory_rikyu.tsv --files-from missing.txt
  rsync -av --files-from=missing.txt <data_dir>/ rikyu:/data1/rkp00078/satoc/Glioma/images/v1.0.0/

Only the cohorts a job can read are inventoried (COHORTS); _backup_* and egd
are ignored on both sides.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

COHORTS = ("ucsf-pdgm", "upenn-gbm", "utsw-glioma", "_global")


def inventory(data_dir: Path, out: Path | None) -> dict:
    rows: list[tuple[str, int]] = []
    for c in COHORTS:
        base = data_dir / c
        if not base.is_dir():
            print(f"WARNING: cohort dir missing: {base}", file=sys.stderr)
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames.sort()
            for f in sorted(filenames):
                p = Path(dirpath) / f
                rows.append((str(p.relative_to(data_dir)), p.stat().st_size))
    rows.sort()
    if out:
        with out.open("w") as fh:
            fh.writelines(f"{r}\t{s}\n" for r, s in rows)
    summary = defaultdict(lambda: {"subjects": set(), "files": 0, "bytes": 0, "nii": 0})
    for r, s in rows:
        parts = r.split("/")
        d = summary[parts[0]]
        if len(parts) > 2:
            d["subjects"].add(parts[1])
        d["files"] += 1; d["bytes"] += s; d["nii"] += r.endswith(".nii.gz")
    print(f"{'cohort':13s} {'subj_dirs':>9s} {'files':>7s} {'nii.gz':>7s} {'GB':>8s}")
    for c in COHORTS:
        if c in summary:
            d = summary[c]
            print(f"{c:13s} {len(d['subjects']):9d} {d['files']:7d} {d['nii']:7d} {d['bytes'] / 1e9:8.2f}")
    tot = sum(d["bytes"] for d in summary.values())
    print(f"{'TOTAL':13s} {'':9s} {len(rows):7d} {'':7s} {tot / 1e9:8.2f}")
    return dict(rows)


def compare(a: Path, b: Path, files_from: Path | None) -> int:
    def load(p):
        with p.open() as fh:
            return {l.split("\t")[0]: int(l.rstrip("\n").split("\t")[1]) for l in fh if "\t" in l}
    A, B = load(a), load(b)
    missing = sorted(k for k in A if k not in B)
    extra = sorted(k for k in B if k not in A)
    size = sorted(k for k in A if k in B and A[k] != B[k])
    print(f"{a.name}: {len(A)} files   {b.name}: {len(B)} files")
    print(f"missing in {b.name}: {len(missing)}   extra in {b.name}: {len(extra)}   size mismatch: {len(size)}")
    for label, lst in (("MISSING", missing), ("SIZE", size), ("EXTRA", extra)):
        for k in lst[:20]:
            print(f"  {label:7s} {k}")
        if len(lst) > 20:
            print(f"  ... {len(lst) - 20} more")
    if files_from:
        with files_from.open("w") as fh:
            fh.writelines(f"{k}\n" for k in missing + size)
        print(f"wrote {files_from} ({len(missing) + len(size)} paths for rsync --files-from)")
    return 0 if not (missing or size) else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("inventory"); i.add_argument("data_dir", type=Path); i.add_argument("-o", "--out", type=Path)
    c = sub.add_parser("compare"); c.add_argument("a", type=Path); c.add_argument("b", type=Path)
    c.add_argument("--files-from", type=Path, default=None)
    a = ap.parse_args()
    if a.cmd == "inventory":
        inventory(a.data_dir, a.out)
    else:
        sys.exit(compare(a.a, a.b, a.files_from))


if __name__ == "__main__":
    main()
