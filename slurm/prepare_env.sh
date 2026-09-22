#!/bin/bash
# One-time setup on the rikyu LOGIN node (free), from the repo root:
#
#   bash slurm/prepare_env.sh
#
# Does what a compute job must not / cannot do for itself:
#   0. connectivity check          -> online or offline mode
#   1. uv + uv-managed CPython 3.13 -> $UV_PYTHON_INSTALL_DIR (group area)
#   2. venv from uv.lock           -> ./.venv  (torch 2.11.0+cu128, aarch64 wheels)
#   3. pretrained ViT weights      -> weights/  (jobs run HF_HUB_OFFLINE=1)
#   4. verification                -> versions, resolved paths, CPU-only pytest
#
# Offline fallback (if step 0 fails): copy the aarch64 uv cache + uv binary +
# weights from a machine that has them (the Claude workspace is aarch64 too):
#   rsync -a <src>/.cache/uv/ rikyu:/data1/rkp00078/satoc/uv/cache/
#   rsync -a <src>/.local/bin/uv rikyu:~/.local/bin/
#   rsync -a <src>/OpenIDH/weights/ rikyu:~/OpenIDH/weights/
# then re-run with OFFLINE=1.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root
source slurm/env.sh
mkdir -p slurm/logs        # sbatch --output needs the directory to exist (gitignored: logs/)
PY_VERSION="${PY_VERSION:-3.13}"   # raiden used 3.13; lock allows >=3.12

# ── 0. connectivity ───────────────────────────────────────────────────────────
# expected (confirmed 2026-09-22): all four print "HTTP/2 200"
OFFLINE="${OFFLINE:-0}"
if [ "${OFFLINE}" != "1" ]; then
  for u in https://astral.sh https://pypi.org/simple/ https://download.pytorch.org/whl/cu128/ https://huggingface.co; do
    code=$(curl -sI -m 10 -o /dev/null -w '%{http_code}' "$u" || echo 000)
    echo "[prepare] $u -> $code"
    [ "$code" = "000" ] && { echo "[prepare] no route to $u — re-run with OFFLINE=1 after staging the cache"; exit 2; }
  done
fi

# ── storage layout ────────────────────────────────────────────────────────────
mkdir -p "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}"
echo "[prepare] uv cache   : ${UV_CACHE_DIR}"
echo "[prepare] uv python  : ${UV_PYTHON_INSTALL_DIR}"
echo "[prepare] venv       : ${UV_PROJECT_ENVIRONMENT:-$(pwd)/.venv}"

# ── 1. uv + interpreter ───────────────────────────────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
  [ "${OFFLINE}" = "1" ] && { echo "[prepare] uv missing and OFFLINE=1 — stage ~/.local/bin/uv first"; exit 2; }
  echo "[prepare] installing uv to ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv --version                                   # expected: uv 0.9.x (aarch64)
UV_OFF=(); [ "${OFFLINE}" = "1" ] && UV_OFF=(--offline)
uv python install "${UV_OFF[@]}" "${PY_VERSION}"   # expected: "Installed Python 3.13.x" (or already present)
uv python find "${PY_VERSION}"                 # expected: a path under $UV_PYTHON_INSTALL_DIR

# ── 2. venv from the lock (same lock as raiden / Mac / dev box) ───────────────
# expected: ~7GB into the cache the first time (torch cu128 aarch64 + nvidia libs),
# then "Installed N packages". Anything about "no wheels for linux_aarch64" is a stop.
echo "[prepare] uv sync --frozen (python ${PY_VERSION})"
uv sync --frozen "${UV_OFF[@]}" --python "${PY_VERSION}"

# ── 3. pretrained weights (jobs are offline) ──────────────────────────────────
W="weights/vit_small_patch16_224.augreg_in21k.safetensors"
if [ -f "${W}" ]; then
  echo "[prepare] weights present: ${W} ($(du -h "${W}" | cut -f1))"
elif [ "${OFFLINE}" = "1" ]; then
  echo "[prepare] weights missing and OFFLINE=1 — rsync weights/ from a networked machine"; exit 2
else
  echo "[prepare] downloading weights from HF (once)"
  HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 uv run --frozen python scripts/download_weights.py
fi
# expected: 115MB file

# ── 4. verification (CPU only — the login node has no GPU for us) ─────────────
echo "[prepare] torch / platform"
uv run --frozen python - <<'PY'
import platform, sys, torch, timm
print(f"  python {sys.version.split()[0]}  {platform.machine()}")
print(f"  torch {torch.__version__}  cuda-build {torch.version.cuda}  timm {timm.__version__}")
assert platform.machine() == "aarch64" and torch.version.cuda == "12.8", "unexpected wheel"
PY
# expected: python 3.13.x aarch64 / torch 2.11.0+cu128 cuda-build 12.8

echo "[prepare] resolved paths (${OPENIDH_PATHS})"
uv run --frozen python - <<'PY'
from pathlib import Path
from openidh_model.utils.paths import load_paths
p = load_paths()
for k, v in p.as_dict().items():
    ok = Path(v).is_dir() or k == "output_dir"
    print(f"  {k:13s} {v}  {'ok' if ok else 'MISSING'}")
    assert ok, f"{k} missing: {v}"
for c in ("ucsf-pdgm", "upenn-gbm", "utsw-glioma"):
    n = sum(1 for d in (p.data_dir / c).iterdir() if d.is_dir())
    print(f"  {c:13s} {n} subject dirs")
PY
# expected: all ok; ucsf-pdgm ~501+ (sessions), upenn-gbm ~611, utsw-glioma ~622 dirs

echo "[prepare] pytest (CPU, no image data needed)"
uv run --frozen pytest tests -q          # expected: 33 passed (param count 44,224,396 inside)

echo "[prepare] home usage: $(du -sh "${HOME}" 2>/dev/null | cut -f1)  group: $(du -sh "${RIKYU_GROUP}/uv" | cut -f1)"
echo "[prepare] done — next: sbatch slurm/tests.sbatch"
