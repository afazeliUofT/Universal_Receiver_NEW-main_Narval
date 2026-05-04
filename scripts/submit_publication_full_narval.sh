#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/rsadve1/scratch/Universal_Receiver_NEW-main_Narval}"
VENV_PATH="${VENV_PATH:-/home/rsadve1/scratch/.venvGRAND}"
ACCOUNT="${UPAIR_SLURM_ACCOUNT:-def-rsadve}"
SEEDS="${UPAIR_PUBLICATION_SEEDS:-7,17,29}"
SEED_ARRAY_CONCURRENCY="${UPAIR_PUBLICATION_ARRAY_CONCURRENCY:-12}"
TRAIN_WALLTIME="${UPAIR_PUBLICATION_WALLTIME:-12:00:00}"
MERGE_WALLTIME="${UPAIR_PUBLICATION_MERGE_WALLTIME:-02:00:00}"
VALIDATE_WALLTIME="${UPAIR_PUBLICATION_VALIDATE_WALLTIME:-01:00:00}"

cd "${PROJECT_ROOT}"

export PROJECT_ROOT
export VENV_PATH
export UPAIR_PUBLICATION_SEEDS="${SEEDS}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"

IFS=',' read -r -a seed_list <<< "${SEEDS}"

echo "[SUBMIT] project=${PROJECT_ROOT}"
echo "[SUBMIT] venv=${VENV_PATH}"
echo "[SUBMIT] account=${ACCOUNT}"
echo "[SUBMIT] seeds=${SEEDS}"
echo "[SUBMIT] train_walltime=${TRAIN_WALLTIME}"
echo "[SUBMIT] merge_walltime=${MERGE_WALLTIME}"
echo "[SUBMIT] per-seed array=0-11%${SEED_ARRAY_CONCURRENCY}"
echo "[SUBMIT] effective GPUs per seed wave: up to ${SEED_ARRAY_CONCURRENCY} GPUs as ${SEED_ARRAY_CONCURRENCY} one-GPU array tasks"
echo "[SUBMIT] workflow: after each seed train/eval array succeeds, run merge/plot before starting the next seed"

mkdir -p logs

validate_job=""
if [[ "${UPAIR_SKIP_VALIDATE:-0}" != "1" ]]; then
  validate_job=$(sbatch --parsable \
    --account="${ACCOUNT}" \
    --time="${VALIDATE_WALLTIME}" \
    --export=ALL \
    slurm/00_validate_gpu.sbatch)
  echo "[SUBMIT] validation job=${validate_job}"
else
  echo "[SUBMIT] validation skipped because UPAIR_SKIP_VALIDATE=1"
fi

prev_dependency=""
if [[ -n "${validate_job}" ]]; then
  prev_dependency="afterok:${validate_job}"
fi

last_merge_job=""
for seed in "${seed_list[@]}"; do
  seed="$(echo "${seed}" | tr -d '[:space:]')"
  if [[ -z "${seed}" ]]; then
    continue
  fi

  seed_args=(
    --parsable
    --account="${ACCOUNT}"
    --time="${TRAIN_WALLTIME}"
    --array="0-11%${SEED_ARRAY_CONCURRENCY}"
    --export="ALL,UPAIR_PUBLICATION_SEEDS=${seed}"
  )

  if [[ -n "${prev_dependency}" ]]; then
    seed_args+=(--dependency="${prev_dependency}")
  fi

  seed_job=$(sbatch "${seed_args[@]}" slurm/10_train_eval_array.sbatch)
  echo "[SUBMIT] seed ${seed} train/eval array job=${seed_job}"
  echo "[SUBMIT] seed ${seed} starts after: ${prev_dependency:-none}"

  merge_job=$(sbatch --parsable \
    --account="${ACCOUNT}" \
    --time="${MERGE_WALLTIME}" \
    --dependency="afterok:${seed_job}" \
    --export=ALL \
    slurm/20_merge_plot.sbatch)

  echo "[SUBMIT] seed ${seed} merge/plot job=${merge_job}"
  echo "[SUBMIT] seed ${seed} merge/plot starts after: afterok:${seed_job}"

  last_merge_job="${merge_job}"
  prev_dependency="afterok:${merge_job}"
done

if [[ -z "${last_merge_job}" ]]; then
  echo "[SUBMIT] ERROR: no seed jobs were submitted. Check UPAIR_PUBLICATION_SEEDS=${SEEDS}" >&2
  exit 2
fi

echo
echo "[SUBMIT] final merge/plot job=${last_merge_job}"
echo "[SUBMIT] monitor:"
echo "  squeue -u \"\$USER\""
echo "  tail -f logs/upair-pub-mu16-<jobid>_<taskid>.out"
echo "  tail -f logs/upair-merge-<jobid>.out"
echo
echo "[SUBMIT] if a seed array hits walltime or fails, its merge and later seed jobs will not start because dependencies use afterok."
echo "[SUBMIT] resubmit failed task ids for that seed, then resume the dependency chain manually."
