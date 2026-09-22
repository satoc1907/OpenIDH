#!/bin/bash
# Step 1 on rikyu: environment check on a GPU node (no image data needed).
#   sbatch slurm/tests.sbatch          # from the repo root
# Expected: node report, cuda available on "NVIDIA GB200", sm_100 in the arch
# list, a GPU forward of the real model with the right output shapes, and
# "33 passed" from pytest. ~3 minutes -> ~15 yen.
#SBATCH --job-name=openidh_tests
#SBATCH --account=rkp00078
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/%x.%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source slurm/env.sh
node_report

echo "[tests] torch on the GPU"
run_py python - <<'PY'
import torch, time
assert torch.cuda.is_available(), "cuda not available — driver/arch mismatch?"
print("  device      :", torch.cuda.get_device_name(0))
print("  capability  :", torch.cuda.get_device_capability(0))        # expected (10, 0)
print("  arch list   :", torch.cuda.get_arch_list())                 # must contain sm_100
assert "sm_100" in torch.cuda.get_arch_list()
x = torch.randn(4096, 4096, device="cuda"); torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(10): y = x @ x
torch.cuda.synchronize(); print(f"  matmul 4096^2 x10: {time.perf_counter()-t:.3f}s")
print("  tf32 matmul :", torch.backends.cuda.matmul.allow_tf32, " cudnn tf32:", torch.backends.cudnn.allow_tf32)

from openidh_model.utils.paths import load_paths
from openidh_model.models.openidh import OpenIDH, IMG_MODALITIES
m = OpenIDH(weights_dir=load_paths().weights_dir).cuda().eval()
print("  params      :", m.num_parameters())                          # expected 44224396
with torch.no_grad():
    ev = m.raw_evidence({k: torch.randn(2, 3, 224, 224, device="cuda") for k in IMG_MODALITIES},
                        torch.randn(2, 2, device="cuda"))
print("  forward     :", {k: tuple(v.shape) for k, v in ev.items()})   # all (2, 2)
PY

echo "[tests] pytest"
run_py pytest tests -q                                                  # expected: 33 passed
echo "[tests] done"
