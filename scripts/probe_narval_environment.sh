#!/usr/bin/env bash
set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-/home/rsadve1/scratch}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_ROOT}/Universal_Receiver_NEW-main_Narval}"
VENV_PATH="${VENV_PATH:-${SCRATCH_ROOT}/.venvUPAIR}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH_ROOT}}"

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

echo "=== Narval environment probe: current/default module state ==="
python3 "${PROJECT_ROOT}/probe_narval_environment.py" \
  --label default \
  --scratch-root "${SCRATCH_ROOT}" \
  --project-root "${PROJECT_ROOT}" \
  --venv-path "${VENV_PATH}" \
  --output "${OUTPUT_DIR}/narval_env_probe_default.json"

echo
echo "=== Optional: rerun after a candidate module stack ==="
echo "If you normally load modules before creating venvs, rerun manually after loading them, for example:"
echo "  module purge"
echo "  module load StdEnv/<version> gcc/<version> python/<version> cuda/<version>"
echo "  cd ${PROJECT_ROOT}"
echo "  python3 probe_narval_environment.py --label candidate --output ${OUTPUT_DIR}/narval_env_probe_candidate.json"
echo
echo "Send back:"
echo "  ${OUTPUT_DIR}/narval_env_probe_default.json"
echo "and, if you run the optional candidate stack:"
echo "  ${OUTPUT_DIR}/narval_env_probe_candidate.json"
