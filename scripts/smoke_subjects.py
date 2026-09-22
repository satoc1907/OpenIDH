"""List the subject directories a config will actually load (for /tmp staging).

train.smoke.yaml keeps `data.subset_n` subjects: `_balanced_subset` takes the first
k Mut + first n-k WT rows of each split role, in split-CSV order, so the set is
fixed by the frozen CSV alone. Prints one "<cohort>/<subject_dir>" per line,
relative to data_dir — the format `stage_data` (slurm/env.sh) feeds to rsync.

    uv run python scripts/smoke_subjects.py --train-yaml train.smoke.yaml > list.txt
    uv run python scripts/smoke_subjects.py --train-yaml train.yaml        # = every subject
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from openidh_model.data.dataset import GliomaDataset  # noqa: E402
from openidh_model.data.metadata import resolve_subject_dir  # noqa: E402
from openidh_model.train.loop import _balanced_subset  # noqa: E402
from openidh_model.utils.config import load_config  # noqa: E402
from openidh_model.utils.paths import load_paths  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-file", default="splits_loso_foldA.csv")
    ap.add_argument("--fold-id", default=None)
    ap.add_argument("--train-yaml", default="train.smoke.yaml")
    ap.add_argument("--paths-file", default=None, help="defaults to $OPENIDH_PATHS")
    a = ap.parse_args()

    paths = load_paths(a.paths_file)
    cfg = load_config(paths.root / "configs", a.train_yaml)
    n = cfg["data"].get("subset_n")
    out: list[str] = []
    for role in ("train", "val", "test"):
        # skip_missing_dirs=False: the list must not depend on what is staged already
        ds = GliomaDataset(paths.splits_dir / a.split_file, paths, role=role,
                           fold_id=a.fold_id, skip_missing_dirs=False)
        rows = ds.rows
        if n:  # same rule as train_fold / run_fold
            rows = _balanced_subset(rows, n if role != "val" else max(4, n // 2))
        for r in rows:
            d = resolve_subject_dir(paths.data_dir, r["site"], r["subject_id"])
            out.append(str(d.relative_to(paths.data_dir)))
    for line in dict.fromkeys(out):   # dedupe, keep order
        print(line)


if __name__ == "__main__":
    main()
