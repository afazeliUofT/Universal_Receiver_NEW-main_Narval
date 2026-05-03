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
from upair5g.training import train_model  # noqa: E402


PROBE_CASES: list[tuple[str, str]] = [
    ("1dmrs", "deep_d96_b6_r2"),
    ("1dmrs", "shallow_d96_b2_r2"),
    ("2dmrs", "deep_d96_b6_r2"),
    ("2dmrs", "shallow_d96_b2_r2"),
    ("1dmrs", "wide_d128_b4_r2"),
    ("1dmrs", "mlpwide_d96_b4_r4"),
    ("2dmrs", "wide_d128_b4_r2"),
    ("2dmrs", "mlpwide_d96_b4_r4"),
]


def _probe_cfg(
    base_cfg: dict[str, Any],
    dmrs_case: str,
    variant_name: str,
    batch_size: int,
    steps: int,
    eval_every: int,
    val_steps: int,
    num_rx_ant: int | None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg = _apply_overrides(cfg, DMRS_CASES[dmrs_case]["overrides"])
    cfg = _apply_overrides(cfg, VARIANTS[variant_name]["overrides"])
    rx_tag = f"rx{num_rx_ant}" if num_rx_ant is not None else f"rx{cfg['channel']['num_rx_ant']}"
    set_cfg(cfg, "experiment.output_root", f"TWC_plots_comprehensive/memory_probe/{rx_tag}/bs{batch_size}/{dmrs_case}")
    set_cfg(cfg, "experiment.name", variant_name)
    if num_rx_ant is not None:
        set_cfg(cfg, "channel.num_rx_ant", int(num_rx_ant))
    set_cfg(cfg, "system.batch_size_train", int(batch_size))
    set_cfg(cfg, "system.batch_size_eval", int(batch_size))
    set_cfg(cfg, "system.graph_mode", True)
    set_cfg(cfg, "baselines.covariance_estimation.order", "f-t-s")
    set_cfg(cfg, "baselines.covariance_estimation.use_spatial_smoothing", True)
    set_cfg(cfg, "training.steps", int(steps))
    set_cfg(cfg, "training.val_steps", int(val_steps))
    set_cfg(cfg, "training.eval_every", int(eval_every))
    set_cfg(cfg, "training.log_every", 25)
    set_cfg(cfg, "training.checkpoint_every", 25)
    set_cfg(cfg, "training.resume", False)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Long Narval GPU-memory probe for UPAIR training stability.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "twc_comprehensive_mu32_base.yaml"))
    parser.add_argument("--case-index", type=int, default=0, choices=range(len(PROBE_CASES)))
    parser.add_argument("--dmrs-case", choices=sorted(DMRS_CASES), default=None)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--eval-every", type=int, default=175)
    parser.add_argument("--val-steps", type=int, default=16)
    parser.add_argument("--num-rx-ant", type=int, default=None)
    args = parser.parse_args()

    if args.dmrs_case is not None or args.variant is not None:
        if args.dmrs_case is None or args.variant is None:
            raise ValueError("--dmrs-case and --variant must be provided together.")
        dmrs_case, variant_name = args.dmrs_case, args.variant
        case_index = "manual"
    else:
        dmrs_case, variant_name = PROBE_CASES[int(args.case_index)]
        case_index = str(args.case_index)
    base_cfg = load_config(args.config)
    print(
        "[MEMPROBE] "
        f"case_index={case_index} dmrs_case={dmrs_case} variant={variant_name} "
        f"batch_size={args.batch_size} num_rx_ant={args.num_rx_ant} "
        f"steps={args.steps} eval_every={args.eval_every} val_steps={args.val_steps}"
    )
    result = train_model(
        _probe_cfg(
            base_cfg=base_cfg,
            dmrs_case=dmrs_case,
            variant_name=variant_name,
            batch_size=args.batch_size,
            steps=args.steps,
            eval_every=args.eval_every,
            val_steps=args.val_steps,
            num_rx_ant=args.num_rx_ant,
        )
    )
    if not result.get("training_complete", False):
        raise SystemExit(f"Memory probe stopped incomplete: {result}")
    print("[MEMPROBE] completed successfully")


if __name__ == "__main__":
    main()
