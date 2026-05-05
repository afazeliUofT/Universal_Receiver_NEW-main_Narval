#!/usr/bin/env bash
#SBATCH --job-name=upair-mb48-canary
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

cd /home/rsadve1/scratch/Universal_Receiver_NEW-main_Narval

if [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
fi

module --force purge || true
module load StdEnv/2023
module load gcc/12.3
module load python/3.11.5
module load cuda/12.2

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PIP_NO_USER=1
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1
unset PYTHONPATH

source /home/rsadve1/scratch/.venvGRAND/bin/activate

python - <<'PY'
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))

from upair5g.config import load_config, set_cfg
from upair5g.evaluation import evaluate_model

cfg = load_config("configs/twc_comprehensive_mu32_base.yaml")

# Keep output away from publication dirs.
out = tempfile.mkdtemp(prefix="upair_mb48_canary_", dir="/tmp")
set_cfg(cfg, "experiment.output_root", out)
set_cfg(cfg, "experiment.name", "canary_task0_u4")

# Stress the largest user-count for task 0.
set_cfg(cfg, "multiuser.fixed_num_users", 4)
set_cfg(cfg, "multiuser.dmrs.length", 1)
set_cfg(cfg, "multiuser.dmrs.additional_position", 0)
set_cfg(cfg, "model.d_model", 96)
set_cfg(cfg, "model.num_blocks", 4)
set_cfg(cfg, "model.mlp_ratio", 2.0)

# Fast one-point test.
set_cfg(cfg, "system.seed", 1007)
set_cfg(cfg, "system.training_seed", 7)
set_cfg(cfg, "system.evaluation_seed", 1007)
set_cfg(cfg, "system.ebno_db_eval", [0])
set_cfg(cfg, "system.batch_size_eval", 96)
set_cfg(cfg, "evaluation.receiver_microbatch_size", 48)
set_cfg(cfg, "evaluation.min_num_batches_per_point", 1)
set_cfg(cfg, "evaluation.max_num_batches_per_point", 2)
set_cfg(cfg, "evaluation.target_block_errors_per_receiver", 0)
set_cfg(cfg, "evaluation.progress_every_batches", 1)
set_cfg(cfg, "evaluation.save_example_batch", False)
set_cfg(cfg, "evaluation.force", True)

# Keep all publication receivers enabled, including perfect CSI.
set_cfg(cfg, "baselines.covariance_estimation.num_batches", 1)
set_cfg(cfg, "baselines.covariance_estimation.batch_size", 8)
set_cfg(cfg, "baselines.covariance_estimation.reuse_cache", False)

ckpt = Path("TWC_plots_comprehensive/runs_rx16/seed7/1dmrs/main_d96_b4_r2/checkpoints/best.weights.h5")
print("canary output:", out)
print("checkpoint:", ckpt)
print("batch_size_eval=96 receiver_microbatch_size=48 users=4")

result = evaluate_model(cfg, checkpoint_path=str(ckpt), num_users=4)
print("PASS canary:", result)
PY
