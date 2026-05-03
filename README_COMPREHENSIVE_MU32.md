# Comprehensive 1--4 User, 32-Antenna SIMO Study

This revision adds a multi-user SIMO setting with 1--4 simultaneous
single-antenna users and a 32-antenna gNB. Training samples the number of
scheduled users with triangular weights, so 4-user batches occur more often
than 3-user batches, 3 more often than 2, and 2 more often than 1.

## Narval environment setup

Use a dedicated virtual environment on Narval:
`/home/rsadve1/scratch/.venvUPAIR`. Before creating it, probe the Narval
software stack from the project directory:

```bash
cd /home/rsadve1/scratch/Universal_Receiver_NEW-main_Narval
bash scripts/probe_narval_environment.sh
```

Send back `/home/rsadve1/scratch/narval_env_probe_default.json` so the
TensorFlow/Sionna/CUDA-compatible venv recipe can be pinned to the installed
Narval stack.

If possible, also run the GPU-node probe so the driver/GPU CUDA runtime is
captured from an allocated node:

```bash
cd /home/rsadve1/scratch/Universal_Receiver_NEW-main_Narval
sbatch slurm/00_probe_narval_gpu.sbatch
```

The Narval GPU validation passed with Python 3.11.5, `cuda/12.2`,
TensorFlow 2.15.1, and Sionna 1.2.1. Create the venv with the pinned
requirements:

```bash
cd /home/rsadve1/scratch/Universal_Receiver_NEW-main_Narval
bash scripts/setup_venv.sh
```

Validate the environment, preferably inside a GPU allocation, with:

```bash
cd /home/rsadve1/scratch/Universal_Receiver_NEW-main_Narval
bash scripts/validate_venv_narval.sh
```

Before running long jobs on Narval, verify the installed Sionna
version creates orthogonal DMRS streams for both DMRS-density cases:

```bash
python probe_multiuser_dmrs_orthogonality.py \
  --config configs/twc_comprehensive_mu32_base.yaml \
  --dmrs-case 1dmrs \
  --num-users 4

python probe_multiuser_dmrs_orthogonality.py \
  --config configs/twc_comprehensive_mu32_base.yaml \
  --dmrs-case 2dmrs \
  --num-users 4
```

The probe prints per-user DMRS counts, pilot-mask overlaps, and the normalized
DMRS Gram matrix. Continue only if the maximum off-diagonal entry passes the
requested tolerance.

To train and evaluate the default ablation suite for both 1-DMRS and 2-DMRS
cases:

```bash
python scripts/run_comprehensive_mu32_ablation.py \
  --config configs/twc_comprehensive_mu32_base.yaml \
  --plot
```

The runner writes per-variant/per-user results under
`TWC_plots_comprehensive/eval_runs/{1dmrs,2dmrs}`, combined CSVs under
`TWC_plots_comprehensive/csv`, and separate figure sets under
`TWC_plots_comprehensive/1dmrs` and `TWC_plots_comprehensive/2dmrs`.

Useful subsets:

```bash
python scripts/run_comprehensive_mu32_ablation.py \
  --variants main_d96_b4_r2 \
  --plot

python scripts/run_comprehensive_mu32_ablation.py \
  --variants main_d96_b4_r2 \
  --dmrs-cases 1dmrs \
  --plot

python scripts/run_comprehensive_mu32_ablation.py \
  --variants main_d96_b4_r2,narrow_d64_b4_r2,wide_d128_b4_r2
```

To regenerate plots from existing CSVs:

```bash
python scripts/make_comprehensive_mu32_plots.py
```

## Slurm workflow

The Slurm scripts are arranged so that validation runs first, then a short
GPU smoke-training job exercises the largest memory cases. If changing memory
or TensorFlow execution settings, run the longer memory probe before the full
array. Then the 12 independent DMRS/ablation workers run as a parallel job
array, and a final CPU job merges CSVs and generates plots:

```bash
cd /home/rsadve1/scratch/Universal_Receiver_NEW-main_Narval
jid_validate=$(sbatch --parsable slurm/00_validate_gpu.sbatch)
jid_smoke=$(sbatch --parsable --dependency=afterok:${jid_validate} slurm/01_smoke_train.sbatch)
jid_memprobe=$(sbatch --parsable --dependency=afterok:${jid_smoke} slurm/02_training_memory_probe.sbatch)
jid_train=$(sbatch --parsable --dependency=afterok:${jid_memprobe} slurm/10_train_eval_array.sbatch)
sbatch --dependency=afterok:${jid_train} slurm/20_merge_plot.sbatch
```

To empirically find the largest stable training/evaluation micro-batch on one
Narval GPU, run:

```bash
sbatch slurm/03_find_safe_batch.sbatch
```

By default this tests batch sizes `128,96,64,48,32`, starting from the largest
and moving smaller only after a failure. It tests both training and evaluation
memory against the deep, wide, and MLP-wide variants for both DMRS cases.
Results are written under `TWC_plots_comprehensive/batch_probe`. Override the
candidate list without editing files, for example:

```bash
sbatch --export=ALL,UPAIR_BATCH_PROBE_SIZES=160,128,96,64,32 slurm/03_find_safe_batch.sbatch
```

The array limit in `slurm/10_train_eval_array.sbatch` controls how many GPU
workers run at once. Lower or raise that value to match the allocation and
 account limits. The default train/evaluation micro-batch sizes are set to 32
for the 32-antenna, 4-user study on Narval A100 40GB GPUs. The Slurm scripts
also set `TF_GPU_ALLOCATOR=cuda_malloc_async` to reduce TensorFlow GPU memory
fragmentation.

The training array is resumable. Each worker writes a full TensorFlow training
state under
`TWC_plots_comprehensive/runs/{1dmrs,2dmrs}/{variant}/checkpoints/training_state`
and best evaluation weights as `best.weights.h5`. If a worker hits wall time,
resubmit the same array task or the full array; completed training resumes from
the saved step and completed per-user evaluation CSVs are reused unless
`--force` is passed.
