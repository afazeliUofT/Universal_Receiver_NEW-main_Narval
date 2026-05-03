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

echo "[INFO] Project root : ${PROJECT_ROOT}"
echo "[INFO] Venv path    : ${VENV_PATH}"
echo "[INFO] StdEnv module: ${STDENV_MODULE}"
echo "[INFO] GCC module   : ${GCC_MODULE}"
echo "[INFO] Python module: ${PYTHON_MODULE}"
echo "[INFO] CUDA module  : ${CUDA_MODULE}"

if [[ ! -d "${PROJECT_ROOT}" ]]; then
  echo "[ERROR] PROJECT_ROOT does not exist: ${PROJECT_ROOT}" >&2
  exit 1
fi
cd "${PROJECT_ROOT}"

python --version

if [[ -d "${VENV_PATH}" && "${UPAIR_RECREATE_VENV:-0}" == "1" ]]; then
  echo "[INFO] Removing existing venv because UPAIR_RECREATE_VENV=1"
  rm -rf "${VENV_PATH:?}"
fi

if [[ ! -d "${VENV_PATH}" ]]; then
  python -m venv "${VENV_PATH}"
fi

# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"

python -m ensurepip --upgrade || true
python -m pip install --upgrade pip setuptools wheel packaging
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps

mkdir -p logs

python - <<'PY'
import sys
print("[INFO] Python executable:", sys.executable)
print("[INFO] Python version   :", sys.version)

try:
    import tensorflow as tf
    print("[INFO] TensorFlow      :", tf.__version__)
    print("[INFO] GPUs            :", tf.config.list_physical_devices("GPU"))
except Exception as e:
    print("[WARN] TensorFlow import issue:", repr(e))

try:
    import sionna
    print("[INFO] Sionna          :", sionna.__version__)
    import sionna.phy.nr
    print("[INFO] Sionna PHY NR   : available")
except Exception as e:
    print("[WARN] Sionna import issue:", repr(e))
PY

echo "[INFO] Setup complete."
echo "[INFO] Activate later with:"
echo "       source ${VENV_PATH}/bin/activate"
