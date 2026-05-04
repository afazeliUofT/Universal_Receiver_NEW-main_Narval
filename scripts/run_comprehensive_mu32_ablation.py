from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from upair5g.config import get_cfg, load_config, set_cfg  # noqa: E402
from upair5g.evaluation import evaluate_model  # noqa: E402
from upair5g.training import train_model  # noqa: E402


VARIANTS: dict[str, dict[str, Any]] = {
    "main_d96_b4_r2": {
        "label": "d=96, L=4, r=2",
        "overrides": {
            "model.d_model": 96,
            "model.num_blocks": 4,
            "model.mlp_ratio": 2.0,
        },
    },
    "shallow_d96_b2_r2": {
        "label": "d=96, L=2, r=2",
        "overrides": {
            "model.d_model": 96,
            "model.num_blocks": 2,
            "model.mlp_ratio": 2.0,
        },
    },
    "deep_d96_b6_r2": {
        "label": "d=96, L=6, r=2",
        "overrides": {
            "model.d_model": 96,
            "model.num_blocks": 6,
            "model.mlp_ratio": 2.0,
        },
    },
    "narrow_d64_b4_r2": {
        "label": "d=64, L=4, r=2",
        "overrides": {
            "model.d_model": 64,
            "model.num_blocks": 4,
            "model.mlp_ratio": 2.0,
        },
    },
    "wide_d128_b4_r2": {
        "label": "d=128, L=4, r=2",
        "overrides": {
            "model.d_model": 128,
            "model.num_blocks": 4,
            "model.mlp_ratio": 2.0,
        },
    },
    "mlpwide_d96_b4_r4": {
        "label": "d=96, L=4, r=4",
        "overrides": {
            "model.d_model": 96,
            "model.num_blocks": 4,
            "model.mlp_ratio": 4.0,
        },
    },
}


DMRS_CASES: dict[str, dict[str, Any]] = {
    "1dmrs": {
        "label": "1-DMRS",
        "overrides": {
            "multiuser.dmrs.length": 1,
            "multiuser.dmrs.additional_position": 0,
        },
    },
    "2dmrs": {
        "label": "2-DMRS",
        "overrides": {
            "multiuser.dmrs.length": 1,
            "multiuser.dmrs.additional_position": 1,
        },
    },
}


def _apply_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(cfg)
    for path, value in overrides.items():
        set_cfg(result, path, value)
    return result


def _rx_tag(cfg: dict[str, Any]) -> str:
    return f"rx{int(get_cfg(cfg, 'channel.num_rx_ant', 0))}"


def _seed_tag(seed: int) -> str:
    return f"seed{int(seed)}"


def _parse_seed_values(base_cfg: dict[str, Any], seed_arg: str | None) -> list[int]:
    if seed_arg:
        values = [int(x.strip()) for x in seed_arg.split(",") if x.strip()]
    else:
        configured = get_cfg(base_cfg, "system.seeds", None)
        if configured is None:
            configured = [get_cfg(base_cfg, "system.seed", 7)]
        if isinstance(configured, str):
            values = [int(x.strip()) for x in configured.split(",") if x.strip()]
        elif isinstance(configured, (int, float)):
            values = [int(configured)]
        else:
            values = [int(x) for x in configured]
    if not values:
        raise ValueError("At least one seed must be configured.")
    return values


def _case_cfg(base_cfg: dict[str, Any], dmrs_case: str) -> dict[str, Any]:
    if dmrs_case not in DMRS_CASES:
        raise KeyError(f"Unknown DMRS case {dmrs_case}. Available: {sorted(DMRS_CASES)}")
    return _apply_overrides(base_cfg, DMRS_CASES[dmrs_case]["overrides"])


def _variant_cfg(base_cfg: dict[str, Any], variant_name: str, dmrs_case: str, seed: int) -> dict[str, Any]:
    if variant_name not in VARIANTS:
        raise KeyError(f"Unknown variant {variant_name}. Available: {sorted(VARIANTS)}")
    cfg = _case_cfg(base_cfg, dmrs_case)
    cfg = _apply_overrides(cfg, VARIANTS[variant_name]["overrides"])
    set_cfg(cfg, "system.seed", int(seed))
    set_cfg(cfg, "experiment.output_root", f"TWC_plots_comprehensive/runs_{_rx_tag(cfg)}/{_seed_tag(seed)}/{dmrs_case}")
    set_cfg(cfg, "experiment.name", variant_name)
    return cfg


def _eval_cfg(train_cfg: dict[str, Any], variant_name: str, dmrs_case: str, num_users: int) -> dict[str, Any]:
    cfg = copy.deepcopy(train_cfg)
    seed = int(get_cfg(cfg, "system.seed", 0))
    set_cfg(cfg, "experiment.output_root", f"TWC_plots_comprehensive/eval_runs_{_rx_tag(cfg)}/{_seed_tag(seed)}/{dmrs_case}")
    set_cfg(cfg, "experiment.name", f"{variant_name}_u{num_users}")
    set_cfg(cfg, "multiuser.fixed_num_users", int(num_users))
    set_cfg(cfg, "evaluation.save_example_batch", variant_name == "main_d96_b4_r2" and num_users == 4)
    return cfg


def _checkpoint_path(cfg: dict[str, Any]) -> Path:
    output_root = PROJECT_ROOT / str(get_cfg(cfg, "experiment.output_root", "outputs"))
    name = str(get_cfg(cfg, "experiment.name", "experiment"))
    ckpt_name = str(get_cfg(cfg, "training.checkpoint_name", "best.weights.h5"))
    return output_root / name / "checkpoints" / ckpt_name


def _summary_path(cfg: dict[str, Any]) -> Path:
    output_root = PROJECT_ROOT / str(get_cfg(cfg, "experiment.output_root", "outputs"))
    name = str(get_cfg(cfg, "experiment.name", "experiment"))
    return output_root / name / "metrics" / "evaluation_summary.json"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _release_tensorflow_state(label: str) -> None:
    gc.collect()
    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass
    try:
        info = tf.config.experimental.get_memory_info("GPU:0")
        current = float(info.get("current", 0)) / (1024.0**3)
        peak = float(info.get("peak", 0)) / (1024.0**3)
        print(f"[COMPREHENSIVE] cleared TensorFlow state before {label}: gpu_mem={current:.2f}GiB peak={peak:.2f}GiB")
    except Exception:
        print(f"[COMPREHENSIVE] cleared TensorFlow state before {label}")


def _copy_curves(
    result: dict[str, Any],
    out_csv: Path,
    variant_name: str,
    label: str,
    dmrs_case: str,
    dmrs_label: str,
    num_users: int,
    seed: int,
) -> pd.DataFrame:
    df = pd.read_csv(result["curves_path"])
    df["dmrs_case"] = dmrs_case
    df["dmrs_label"] = dmrs_label
    df["variant"] = variant_name
    df["variant_label"] = label
    df["num_users"] = int(num_users)
    df["seed"] = int(seed)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 1--4 user comprehensive UPAIR ablation for the configured gNB antenna count.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "twc_comprehensive_mu32_base.yaml"))
    parser.add_argument("--variants", default="main_d96_b4_r2,shallow_d96_b2_r2,deep_d96_b6_r2,narrow_d64_b4_r2,wide_d128_b4_r2,mlpwide_d96_b4_r4")
    parser.add_argument("--dmrs-cases", default="1dmrs,2dmrs", help="Comma-separated DMRS cases to run. Default: 1dmrs,2dmrs.")
    parser.add_argument("--seeds", default=None, help="Comma-separated random seeds. Defaults to system.seeds in the config.")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and reuse existing checkpoints.")
    parser.add_argument("--no-global-summary", action="store_true", help="Skip shared combined CSV/manifest writes. Use this for parallel Slurm array workers.")
    parser.add_argument("--force", action="store_true", help="Re-run training/evaluation even if resumable outputs already exist.")
    parser.add_argument("--plot", action="store_true", help="Generate TWC_plots_comprehensive figures after evaluation.")
    args = parser.parse_args()
    if args.plot and args.no_global_summary:
        raise ValueError("--plot requires global summary files; omit --no-global-summary or run merge_comprehensive_mu32_results.py first.")

    base_cfg = load_config(args.config)
    variant_names = [name.strip() for name in args.variants.split(",") if name.strip()]
    dmrs_cases = [name.strip() for name in args.dmrs_cases.split(",") if name.strip()]
    seed_values = _parse_seed_values(base_cfg, args.seeds)
    eval_num_users = [int(x) for x in get_cfg(base_cfg, "multiuser.eval_num_users", [1, 2, 3, 4])]

    rx_tag = _rx_tag(base_cfg)
    csv_dir = PROJECT_ROOT / "TWC_plots_comprehensive" / f"csv_{rx_tag}"
    csv_dir.mkdir(parents=True, exist_ok=True)

    all_frames: list[pd.DataFrame] = []
    manifest: dict[str, Any] = {
        "base_config": str(Path(args.config).resolve()),
        "num_rx_ant": int(get_cfg(base_cfg, "channel.num_rx_ant", 0)),
        "dmrs_cases": {},
        "eval_num_users": eval_num_users,
        "seeds": seed_values,
    }

    for seed in seed_values:
        for dmrs_case in dmrs_cases:
            if dmrs_case not in DMRS_CASES:
                raise KeyError(f"Unknown DMRS case {dmrs_case}. Available: {sorted(DMRS_CASES)}")
            dmrs_label = str(DMRS_CASES[dmrs_case]["label"])
            manifest["dmrs_cases"].setdefault(
                dmrs_case,
                {
                    "label": dmrs_label,
                    "overrides": DMRS_CASES[dmrs_case]["overrides"],
                    "variants": {},
                },
            )

            for variant_name in variant_names:
                train_cfg = _variant_cfg(base_cfg, variant_name, dmrs_case, seed)
                label = str(VARIANTS[variant_name]["label"])

                if args.eval_only:
                    checkpoint_path = _checkpoint_path(train_cfg)
                    if not checkpoint_path.exists():
                        raise FileNotFoundError(f"--eval-only requested but checkpoint is missing: {checkpoint_path}")
                    train_result = {
                        "checkpoint_path": str(checkpoint_path),
                        "model_summary_path": str(_checkpoint_path(train_cfg).parents[1] / "metrics" / "model_summary.json"),
                        "training_complete": True,
                    }
                else:
                    if args.force:
                        set_cfg(train_cfg, "training.resume", False)
                    train_result = train_model(train_cfg)
                    if not bool(train_result.get("training_complete", True)):
                        raise SystemExit(
                            "Training stopped after saving resumable state. "
                            "Resubmit the same Slurm array task to continue from the saved checkpoint."
                        )
                    checkpoint_path = Path(train_result["checkpoint_path"])

                model_summary_path = Path(str(train_result.get("model_summary_path", "")))
                model_summary = _read_json(model_summary_path)
                variant_manifest = manifest["dmrs_cases"][dmrs_case]["variants"].setdefault(
                    variant_name,
                    {
                        "label": label,
                        "seed_runs": {},
                    },
                )
                variant_manifest["seed_runs"][str(seed)] = {
                    "checkpoint_path": str(checkpoint_path),
                    "model_summary": model_summary,
                    "curves": {},
                }

                for num_users in eval_num_users:
                    _release_tensorflow_state(f"evaluation seed={seed} {dmrs_case}/{variant_name}/u{num_users}")
                    cfg_eval = _eval_cfg(train_cfg, variant_name, dmrs_case, num_users)
                    out_csv = csv_dir / _seed_tag(seed) / dmrs_case / f"{variant_name}_u{num_users}_curves.csv"
                    summary_path = _summary_path(cfg_eval)
                    if out_csv.exists() and not args.force:
                        print(f"[COMPREHENSIVE] reusing existing evaluation CSV {out_csv}")
                        frame = pd.read_csv(out_csv)
                        if "seed" not in frame.columns:
                            frame["seed"] = int(seed)
                    else:
                        if args.force:
                            set_cfg(cfg_eval, "evaluation.force", True)
                        result = evaluate_model(cfg_eval, checkpoint_path=str(checkpoint_path), num_users=num_users)
                        if not bool(result.get("evaluation_complete", True)):
                            raise SystemExit(
                                "Evaluation stopped after saving resumable state. "
                                "Resubmit the same Slurm array task to continue from the saved evaluation state."
                            )
                        summary_path = Path(str(result["summary_path"]))
                        frame = _copy_curves(result, out_csv, variant_name, label, dmrs_case, dmrs_label, num_users, seed)
                    all_frames.append(frame)
                    variant_manifest["seed_runs"][str(seed)]["curves"][str(num_users)] = {
                        "csv": str(out_csv),
                        "summary": str(summary_path),
                    }

    if args.no_global_summary:
        print("[COMPREHENSIVE] skipped shared combined CSV/manifest writes for parallel worker mode")
    else:
        combined = pd.concat(all_frames, ignore_index=True)
        combined_path = csv_dir / "comprehensive_curves.csv"
        combined.to_csv(combined_path, index=False)
        manifest["combined_csv"] = str(combined_path)
        if "dmrs_case" in combined.columns:
            for dmrs_case, case_df in combined.groupby("dmrs_case"):
                case_path = csv_dir / str(dmrs_case) / "comprehensive_curves.csv"
                case_path.parent.mkdir(parents=True, exist_ok=True)
                case_df.to_csv(case_path, index=False)
                if str(dmrs_case) in manifest["dmrs_cases"]:
                    manifest["dmrs_cases"][str(dmrs_case)]["combined_csv"] = str(case_path)

        manifest_path = csv_dir / "comprehensive_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

        print(f"[COMPREHENSIVE] wrote {combined_path}")
        print(f"[COMPREHENSIVE] wrote {manifest_path}")

    if args.plot:
        from make_comprehensive_mu32_plots import main as plot_main

        plot_main()


if __name__ == "__main__":
    main()
