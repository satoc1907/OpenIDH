"""Path resolution — the single place environment differences live (spec §1.2).

Which YAML is loaded is controlled by the ``OPENIDH_PATHS`` env var; it defaults
to ``configs/paths.local.yaml``. ``${...}`` references between keys are resolved,
and a ``root: AUTO`` value is replaced by the auto-detected repo root (the nearest
ancestor directory containing ``data/metadata``).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

_DEFAULT = "configs/paths.local.yaml"
_VAR = re.compile(r"\$\{(\w+)\}")


def _find_repo_root(start: Path) -> Path:
    for d in [start, *start.parents]:
        if (d / "data" / "metadata").is_dir():
            return d
    # fall back to the package's grandparent (…/openidh_model/utils/paths.py -> repo)
    return Path(__file__).resolve().parents[2]


def _resolve(raw: dict) -> dict:
    """Resolve ${key} references iteratively (data_dir -> root, splits_dir -> data_dir)."""
    out = dict(raw)
    for _ in range(10):
        changed = False
        for k, v in out.items():
            if isinstance(v, str) and _VAR.search(v):
                new = _VAR.sub(lambda m: str(out.get(m.group(1), m.group(0))), v)
                if new != v:
                    out[k] = new
                    changed = True
        if not changed:
            break
    return out


@dataclass(frozen=True)
class Paths:
    root: Path
    data_dir: Path
    metadata_dir: Path
    splits_dir: Path
    weights_dir: Path
    output_dir: Path

    def as_dict(self) -> dict:
        return {k: str(v) for k, v in self.__dict__.items()}


def load_paths(paths_file: str | os.PathLike | None = None) -> Paths:
    pf = Path(paths_file or os.environ.get("OPENIDH_PATHS", _DEFAULT))
    if not pf.is_absolute():
        pf = _find_repo_root(Path.cwd()) / pf
    raw = yaml.safe_load(pf.read_text())

    if str(raw.get("root", "")).strip().upper() == "AUTO":
        raw["root"] = str(_find_repo_root(pf.parent))

    r = _resolve(raw)
    return Paths(
        root=Path(r["root"]),
        data_dir=Path(r["data_dir"]),
        metadata_dir=Path(r["metadata_dir"]),
        splits_dir=Path(r["splits_dir"]),
        weights_dir=Path(r["weights_dir"]),
        output_dir=Path(r["output_dir"]),
    )
