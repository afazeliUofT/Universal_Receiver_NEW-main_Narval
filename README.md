# UPAIR-5G Multi-User SIMO Publication Run

This repository contains the source, configuration, and Slurm entrypoints for
the 1--4 user, 16-antenna gNB, 16-QAM multi-user SIMO study.

The publication configuration is `configs/twc_comprehensive_mu32_base.yaml`.
It uses graph mode, training batch size 96, evaluation batch size 128, three
random seeds (`7,17,29`), 10000 training steps, full evaluation batches, and
LMMSE interpolation without spatial smoothing.

Generated logs, checkpoints, metrics, CSVs, and figures are written under
`logs/` and `TWC_plots_comprehensive/`. These directories are intentionally
ignored by git.

## Narval Run

From `/home/rsadve1/scratch/Universal_Receiver_NEW-main_Narval`:

```bash
git pull --ff-only origin main
bash scripts/submit_publication_full_narval.sh
```

The submission script runs validation, launches the seed/DMRS/ablation training
array, and schedules merge/plot generation after the array completes. If any
array task hits wall time, resubmit the failed task id(s); training resumes from
TensorFlow checkpoints and evaluation resumes from completed Eb/N0 points.

The final combined data are written to:

- `TWC_plots_comprehensive/csv_rx16/comprehensive_curves.csv`
- `TWC_plots_comprehensive/csv_rx16/comprehensive_seed_summary.csv`
- `TWC_plots_comprehensive/csv_rx16/comprehensive_manifest.json`

The final figures are written under `TWC_plots_comprehensive/{1dmrs,2dmrs}/`.
