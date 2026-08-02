#!/bin/bash
# One-time setup on the raiden LOGIN node (raiden-l1), from the repo root:
#
#   bash qsub/prepare_env.sh
#
# Does the three things a compute job cannot do for itself:
#   1. install uv (if missing) and build the venv from uv.lock
#   2. download the pretrained ViT weights into weights/  (jobs run HF_HUB_OFFLINE=1)
#   3. print the resolved HPC paths and check they exist
# Set NO_PROXY_SETUP=1 if the login node reaches the internet without the proxy.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

if [ -z "${NO_PROXY_SETUP:-}" ]; then
  export MY_PROXY_URL="http://10.1.10.1:8080/"
  export HTTP_PROXY=$MY_PROXY_URL HTTPS_PROXY=$MY_PROXY_URL FTP_PROXY=$MY_PROXY_URL
  export http_proxy=$MY_PROXY_URL https_proxy=$MY_PROXY_URL ftp_proxy=$MY_PROXY_URL
fi

unset PYTHONPATH PYTHONUSERBASE PREFIX
export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
export UV_LINK_MODE=copy
export UV_CACHE_DIR="${UV_CACHE_DIR:-${HOME}/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${HOME}/.local/share/uv/python}"
export OPENIDH_PATHS="${OPENIDH_PATHS:-configs/paths.hpc.yaml}"

# ── 1. uv + venv ──────────────────────────────────────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
  echo "[prepare] uv not found — installing to ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv --version
echo "[prepare] uv sync (downloads CPython >=3.12 + torch cu128, ~5GB on first run)"
uv sync --frozen

# ── 2. pretrained weights (jobs are offline) ──────────────────────────────────
W="weights/vit_small_patch16_224.augreg_in21k.safetensors"
if [ -f "${W}" ]; then
  echo "[prepare] weights present: ${W}"
else
  echo "[prepare] staging ${W}"
  uv run --frozen python scripts/download_weights.py
fi

# ── 3. path sanity ────────────────────────────────────────────────────────────
uv run --frozen python - <<'PY'
import os, sys
from pathlib import Path
sys.path.insert(0, ".")
from openidh_model.utils.paths import load_paths

p = load_paths(os.environ["OPENIDH_PATHS"])
Path(p.output_dir).mkdir(parents=True, exist_ok=True)   # created, not checked
bad = 0
for k, v in p.as_dict().items():
    ok = Path(v).exists()
    bad += not ok
    print(f"{'OK ' if ok else 'MISSING'}  {k:14s} {v}")
if bad:
    print(f"\n{bad} path(s) missing — fix configs/paths.hpc.yaml or stage the data before submitting.")
    sys.exit(1)
print("\nall paths present.")
PY

echo
echo "[prepare] ready. Submit with:  DRY_RUN=1 bash qsub/submit_all.sh   (drop DRY_RUN to launch)"
