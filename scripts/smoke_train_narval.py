from __future__ import annotations

import copy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in [SRC_ROOT, SCRIPT_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_comprehensive_mu32_ablation import DMRS_CASES, VARIANTS, _apply_overrides  # noqa: E402
from upair5g.config import load_config, set_cfg  # noqa: E402
from upair5g.training import train_model  # noqa: E402


def _smoke_cfg(base_cfg: dict, variant_name: str, dmrs_case: str) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg = _apply_overrides(cfg, DMRS_CASES[dmrs_case]["overrides"])
    cfg = _apply_overrides(cfg, VARIANTS[variant_name]["overrides"])
    set_cfg(cfg, "experiment.output_root", f"TWC_plots_comprehensive/smoke/{dmrs_case}")
    set_cfg(cfg, "experiment.name", variant_name)
    set_cfg(cfg, "system.batch_size_train", 8)
    set_cfg(cfg, "system.batch_size_eval", 8)
    set_cfg(cfg, "training.steps", 2)
    set_cfg(cfg, "training.val_steps", 1)
    set_cfg(cfg, "training.eval_every", 2)
    set_cfg(cfg, "training.log_every", 1)
    set_cfg(cfg, "training.checkpoint_every", 1)
    set_cfg(cfg, "training.resume", False)
    return cfg


def main() -> None:
    base_cfg = load_config(PROJECT_ROOT / "configs" / "twc_comprehensive_mu32_base.yaml")
    smoke_cases = [
        ("2dmrs", "deep_d96_b6_r2"),
        ("2dmrs", "wide_d128_b4_r2"),
        ("2dmrs", "mlpwide_d96_b4_r4"),
    ]
    for dmrs_case, variant_name in smoke_cases:
        print(f"[SMOKE] dmrs_case={dmrs_case} variant={variant_name}")
        result = train_model(_smoke_cfg(base_cfg, variant_name, dmrs_case))
        if not result.get("training_complete", False):
            raise SystemExit(f"Smoke training did not complete: {dmrs_case}/{variant_name}")
    print("[SMOKE] Narval smoke training completed.")


if __name__ == "__main__":
    main()
