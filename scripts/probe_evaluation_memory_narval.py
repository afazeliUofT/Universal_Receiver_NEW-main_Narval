from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in [SRC_ROOT, SCRIPT_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_comprehensive_mu32_ablation import DMRS_CASES, VARIANTS, _apply_overrides  # noqa: E402
from upair5g.config import load_config, set_cfg  # noqa: E402
from upair5g.evaluation import evaluate_model  # noqa: E402


def _eval_probe_cfg(
    base_cfg: dict[str, Any],
    dmrs_case: str,
    variant_name: str,
    batch_size: int,
    num_users: int,
    ebno_points: list[float],
    num_batches: int,
    cov_batches: int,
    num_rx_ant: int | None,
    receiver_microbatch_size: int | None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg = _apply_overrides(cfg, DMRS_CASES[dmrs_case]["overrides"])
    cfg = _apply_overrides(cfg, VARIANTS[variant_name]["overrides"])
    rx_tag = f"rx{num_rx_ant}" if num_rx_ant is not None else f"rx{cfg['channel']['num_rx_ant']}"
    set_cfg(cfg, "experiment.output_root", f"TWC_plots_comprehensive/eval_memory_probe/{rx_tag}/bs{batch_size}/{dmrs_case}")
    set_cfg(cfg, "experiment.name", f"{variant_name}_u{num_users}")
    if num_rx_ant is not None:
        set_cfg(cfg, "channel.num_rx_ant", int(num_rx_ant))
    set_cfg(cfg, "multiuser.fixed_num_users", int(num_users))
    set_cfg(cfg, "system.batch_size_eval", int(batch_size))
    set_cfg(cfg, "system.graph_mode", True)
    if receiver_microbatch_size is not None and receiver_microbatch_size > 0:
        set_cfg(cfg, "evaluation.receiver_microbatch_size", int(receiver_microbatch_size))
    set_cfg(cfg, "system.ebno_db_eval", [float(x) for x in ebno_points])
    set_cfg(cfg, "evaluation.resume", False)
    set_cfg(cfg, "evaluation.force", True)
    set_cfg(cfg, "evaluation.save_example_batch", False)
    set_cfg(cfg, "evaluation.min_num_batches_per_point", int(num_batches))
    set_cfg(cfg, "evaluation.max_num_batches_per_point", int(num_batches))
    set_cfg(cfg, "evaluation.target_block_errors_per_receiver", 0)
    set_cfg(cfg, "baselines.covariance_estimation.reuse_cache", False)
    set_cfg(cfg, "baselines.covariance_estimation.num_batches", int(cov_batches))
    set_cfg(cfg, "baselines.covariance_estimation.batch_size", int(batch_size))
    set_cfg(cfg, "baselines.covariance_estimation.order", "f-t")
    set_cfg(cfg, "baselines.covariance_estimation.use_spatial_smoothing", False)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Narval GPU-memory probe for UPAIR evaluation.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "twc_comprehensive_mu32_base.yaml"))
    parser.add_argument("--dmrs-case", choices=sorted(DMRS_CASES), required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-users", type=int, default=4)
    parser.add_argument("--ebno-points", default="0,8")
    parser.add_argument("--num-batches", type=int, default=2)
    parser.add_argument("--cov-batches", type=int, default=2)
    parser.add_argument("--num-rx-ant", type=int, default=None)
    parser.add_argument("--receiver-microbatch-size", type=int, default=0)
    args = parser.parse_args()

    ebno_points = [float(item.strip()) for item in args.ebno_points.split(",") if item.strip()]
    if not ebno_points:
        raise ValueError("At least one Eb/N0 point is required.")
    base_cfg = load_config(args.config)
    print(
        "[EVALPROBE] "
        f"dmrs_case={args.dmrs_case} variant={args.variant} batch_size={args.batch_size} "
        f"num_rx_ant={args.num_rx_ant} num_users={args.num_users} ebno_points={ebno_points} "
        f"num_batches={args.num_batches} cov_batches={args.cov_batches} "
        f"receiver_microbatch_size={args.receiver_microbatch_size}"
    )
    result = evaluate_model(
        _eval_probe_cfg(
            base_cfg=base_cfg,
            dmrs_case=args.dmrs_case,
            variant_name=args.variant,
            batch_size=args.batch_size,
            num_users=args.num_users,
            ebno_points=ebno_points,
            num_batches=args.num_batches,
            cov_batches=args.cov_batches,
            num_rx_ant=args.num_rx_ant,
            receiver_microbatch_size=int(args.receiver_microbatch_size) if int(args.receiver_microbatch_size) > 0 else None,
        ),
        checkpoint_path=None,
        num_users=int(args.num_users),
    )
    print(f"[EVALPROBE] completed successfully: {result}")


if __name__ == "__main__":
    main()
