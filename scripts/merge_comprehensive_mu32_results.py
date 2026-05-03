from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
SRC_ROOT = PROJECT_ROOT / "src"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from run_comprehensive_mu32_ablation import DMRS_CASES, VARIANTS  # noqa: E402
from upair5g.config import get_cfg, load_config  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "configs" / "twc_comprehensive_mu32_base.yaml"
BASE_CFG = load_config(CONFIG_PATH)
RX_TAG = f"rx{int(get_cfg(BASE_CFG, 'channel.num_rx_ant', 0))}"
CSV_ROOT = PROJECT_ROOT / "TWC_plots_comprehensive" / f"csv_{RX_TAG}"
RUN_ROOT = PROJECT_ROOT / "TWC_plots_comprehensive" / f"runs_{RX_TAG}"
EVAL_ROOT = PROJECT_ROOT / "TWC_plots_comprehensive" / f"eval_runs_{RX_TAG}"
CURVE_RE = re.compile(r"(?P<variant>.+)_u(?P<num_users>[1-4])_curves\.csv$")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _curve_files() -> list[Path]:
    files: list[Path] = []
    for dmrs_case in DMRS_CASES:
        files.extend(sorted((CSV_ROOT / dmrs_case).glob("*_u*_curves.csv")))
    return files


def main() -> None:
    frames: list[pd.DataFrame] = []
    manifest: dict[str, Any] = {
        "base_config": str(CONFIG_PATH.resolve()),
        "num_rx_ant": int(get_cfg(BASE_CFG, "channel.num_rx_ant", 0)),
        "dmrs_cases": {},
        "eval_num_users": [1, 2, 3, 4],
    }

    for dmrs_case, dmrs_meta in DMRS_CASES.items():
        manifest["dmrs_cases"][dmrs_case] = {
            "label": dmrs_meta["label"],
            "overrides": dmrs_meta["overrides"],
            "variants": {},
        }

    for path in _curve_files():
        dmrs_case = path.parent.name
        match = CURVE_RE.match(path.name)
        if match is None:
            continue
        variant = match.group("variant")
        num_users = int(match.group("num_users"))
        df = pd.read_csv(path)
        if "dmrs_case" not in df.columns:
            df["dmrs_case"] = dmrs_case
        if "dmrs_label" not in df.columns:
            df["dmrs_label"] = DMRS_CASES.get(dmrs_case, {}).get("label", dmrs_case)
        if "variant" not in df.columns:
            df["variant"] = variant
        if "variant_label" not in df.columns:
            df["variant_label"] = VARIANTS.get(variant, {}).get("label", variant)
        if "num_users" not in df.columns:
            df["num_users"] = num_users
        frames.append(df)

        case_manifest = manifest["dmrs_cases"].setdefault(dmrs_case, {"label": dmrs_case, "variants": {}})
        variant_manifest = case_manifest["variants"].setdefault(
            variant,
            {
                "label": VARIANTS.get(variant, {}).get("label", variant),
                "checkpoint_path": str(RUN_ROOT / dmrs_case / variant / "checkpoints" / "best.weights.h5"),
                "model_summary": _read_json(RUN_ROOT / dmrs_case / variant / "metrics" / "model_summary.json"),
                "curves": {},
            },
        )
        variant_manifest["curves"][str(num_users)] = {
            "csv": str(path),
            "summary": str(EVAL_ROOT / dmrs_case / f"{variant}_u{num_users}" / "metrics" / "evaluation_summary.json"),
        }

    if not frames:
        raise FileNotFoundError(f"No per-worker curve CSVs found under {CSV_ROOT}/{{1dmrs,2dmrs}}")

    combined = pd.concat(frames, ignore_index=True)
    CSV_ROOT.mkdir(parents=True, exist_ok=True)
    combined_path = CSV_ROOT / "comprehensive_curves.csv"
    combined.to_csv(combined_path, index=False)
    manifest["combined_csv"] = str(combined_path)

    for dmrs_case, case_df in combined.groupby("dmrs_case"):
        case_path = CSV_ROOT / str(dmrs_case) / "comprehensive_curves.csv"
        case_path.parent.mkdir(parents=True, exist_ok=True)
        case_df.to_csv(case_path, index=False)
        if str(dmrs_case) in manifest["dmrs_cases"]:
            manifest["dmrs_cases"][str(dmrs_case)]["combined_csv"] = str(case_path)

    manifest_path = CSV_ROOT / "comprehensive_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    print(f"[MERGE] wrote {combined_path}")
    print(f"[MERGE] wrote {manifest_path}")


if __name__ == "__main__":
    main()
