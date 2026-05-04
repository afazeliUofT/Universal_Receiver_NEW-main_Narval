#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/rsadve1/scratch/Universal_Receiver_NEW-main_Narval}"
VENV_PATH="${VENV_PATH:-/home/rsadve1/scratch/.venvGRAND}"
ACCOUNT="${UPAIR_SLURM_ACCOUNT:-def-rsadve}"
SEEDS="${UPAIR_PUBLICATION_SEEDS:-7,17,29}"
ARRAY_CONCURRENCY="${UPAIR_PUBLICATION_ARRAY_CONCURRENCY:-6}"
TRAIN_WALLTIME="${UPAIR_PUBLICATION_WALLTIME:-12:00:00}"
MERGE_WALLTIME="${UPAIR_PUBLICATION_MERGE_WALLTIME:-02:00:00}"

cd "${PROJECT_ROOT}"
mkdir -p logs

export PROJECT_ROOT
export VENV_PATH
export UPAIR_PUBLICATION_SEEDS="${SEEDS}"

IFS=',' read -r -a seed_array <<< "${SEEDS}"
num_seeds="${#seed_array[@]}"
num_dmrs_cases=2
num_variants=6
num_tasks=$((num_seeds * num_dmrs_cases * num_variants))
last_task=$((num_tasks - 1))

echo "[SUBMIT] project=${PROJECT_ROOT}"
echo "[SUBMIT] venv=${VENV_PATH}"
echo "[SUBMIT] account=${ACCOUNT}"
echo "[SUBMIT] seeds=${SEEDS}"
echo "[SUBMIT] array=0-${last_task}%${ARRAY_CONCURRENCY}"
echo "[SUBMIT] train_walltime=${TRAIN_WALLTIME}"

validate_job=""
if [[ "${UPAIR_SKIP_VALIDATE:-0}" != "1" ]]; then
  validate_job=$(sbatch --parsable --account="${ACCOUNT}" --export=ALL slurm/00_validate_gpu.sbatch)
  echo "[SUBMIT] validate_job=${validate_job}"
fi

train_args=(
  --parsable
  --account="${ACCOUNT}"
  --time="${TRAIN_WALLTIME}"
  --array="0-${last_task}%${ARRAY_CONCURRENCY}"
  --export=ALL
)
if [[ -n "${validate_job}" ]]; then
  train_args+=(--dependency="afterok:${validate_job}")
fi
train_job=$(sbatch "${train_args[@]}" slurm/10_train_eval_array.sbatch)
echo "[SUBMIT] train_array_job=${train_job}"

merge_job=$(sbatch \
  --parsable \
  --account="${ACCOUNT}" \
  --time="${MERGE_WALLTIME}" \
  --dependency="afterok:${train_job}" \
  --export=ALL \
  slurm/20_merge_plot.sbatch)
echo "[SUBMIT] merge_plot_job=${merge_job}"

cat <<EOF
[SUBMIT] Done.
[SUBMIT] Watch with:
  squeue -u "\$USER"
[SUBMIT] If any array task hits wall time, resubmit the failed task id(s) with the same command shape:
  sbatch --account="${ACCOUNT}" --time="${TRAIN_WALLTIME}" --array=<task_ids> --export=ALL slurm/10_train_eval_array.sbatch
[SUBMIT] After all train/eval tasks finish, rerun merge/plots if needed:
  sbatch --account="${ACCOUNT}" --time="${MERGE_WALLTIME}" --export=ALL slurm/20_merge_plot.sbatch
EOF
