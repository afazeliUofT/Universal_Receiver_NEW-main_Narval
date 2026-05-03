#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/rsadve1/scratch/Universal_Receiver_NEW-main_Narval}"
VENV_PATH="${VENV_PATH:-/home/rsadve1/scratch/.venvUPAIR}"
STDENV_MODULE="${STDENV_MODULE:-StdEnv/2023}"
GCC_MODULE="${GCC_MODULE:-gcc/12.3}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11.5}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.2}"

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

module --force purge || true
module load "${STDENV_MODULE}"
module load "${GCC_MODULE}"
module load "${PYTHON_MODULE}"
if [[ -n "${CUDA_MODULE}" ]]; then
  module load "${CUDA_MODULE}"
fi

export PYTHONNOUSERSITE=1
export PIP_NO_USER=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
unset PYTHONPATH

cd "${PROJECT_ROOT}"
# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"

python - <<'PY'
import importlib
import sys

print("[INFO] Python executable:", sys.executable)
print("[INFO] Python version   :", sys.version)

for name in ["numpy", "scipy", "pandas", "matplotlib", "yaml", "tensorflow", "sionna"]:
    mod = importlib.import_module(name)
    print(f"[INFO] {name:10s}: {getattr(mod, '__version__', 'available')}")

import tensorflow as tf
print("[INFO] TensorFlow GPUs :", tf.config.list_physical_devices("GPU"))

from sionna.phy.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from sionna.phy.channel.tr38901 import CDL
from sionna.phy.ofdm import LinearDetector
from sionna.phy.mimo import StreamManagement

print("[INFO] Sionna required symbols: available")
print("[INFO] PUSCHConfig:", PUSCHConfig)
print("[INFO] PUSCHTransmitter:", PUSCHTransmitter)
print("[INFO] PUSCHReceiver:", PUSCHReceiver)
print("[INFO] CDL:", CDL)
print("[INFO] LinearDetector:", LinearDetector)
print("[INFO] StreamManagement:", StreamManagement)
PY

if [[ "${UPAIR_SKIP_DMRS_PROBE:-0}" != "1" ]]; then
  python probe_multiuser_dmrs_orthogonality.py \
    --config configs/twc_comprehensive_mu32_base.yaml \
    --dmrs-case 1dmrs \
    --num-users 4

  python probe_multiuser_dmrs_orthogonality.py \
    --config configs/twc_comprehensive_mu32_base.yaml \
    --dmrs-case 2dmrs \
    --num-users 4
fi

echo "[INFO] Narval venv validation complete."
