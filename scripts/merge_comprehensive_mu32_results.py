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
METRIC_COLUMNS = ["ber", "bler", "nmse"]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _curve_files() -> list[Path]:
    files: list[Path] = []
    for dmrs_case in DMRS_CASES:
        files.extend(sorted(CSV_ROOT.glob(f"seed*/{dmrs_case}/*_u*_curves.csv")))
        files.extend(sorted((CSV_ROOT / dmrs_case).glob("*_u*_curves.csv")))
    return sorted(set(files))


def _seed_tag(seed: int) -> str:
    return f"seed{int(seed)}"


def _seed_from_path(path: Path) -> int | None:
    try:
        parts = path.relative_to(CSV_ROOT).parts
    except ValueError:
        return None
    for part in parts:
        if part.startswith("seed"):
            try:
                return int(part.removeprefix("seed"))
            except ValueError:
                return None
    return None


def _seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        col
        for col in ["dmrs_case", "dmrs_label", "variant", "variant_label", "num_users", "receiver", "ebno_db"]
        if col in df.columns
    ]
    if not group_cols:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for key, group in df.groupby(group_cols, dropna=False):
        values = key if isinstance(key, tuple) else (key,)
        row = {col: values[i] for i, col in enumerate(group_cols)}
        row["num_seeds"] = int(group["seed"].nunique()) if "seed" in group.columns else 1
        for metric in METRIC_COLUMNS:
            if metric not in group.columns:
                continue
            vals = group[metric].dropna().astype(float)
            count = int(len(vals))
            mean = float(vals.mean()) if count else None
            std = float(vals.std(ddof=1)) if count > 1 else 0.0 if count == 1 else None
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_count"] = count
            row[f"{metric}_ci95"] = float(1.96 * std / (count**0.5)) if count > 1 and std is not None else 0.0 if count == 1 else None
        for metric in ["ber", "bler"]:
            reliable_col = f"reliable_{metric}"
            if reliable_col in group.columns:
                reliable = group[reliable_col].fillna(False).astype(bool)
                row[f"{reliable_col}_all_seeds"] = bool(reliable.all())
                row[f"{reliable_col}_any_seed"] = bool(reliable.any())
        for count_col in ["bit_errors", "num_bits", "block_errors", "num_blocks"]:
            if count_col in group.columns:
                row[f"{count_col}_sum"] = int(group[count_col].fillna(0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    frames: list[pd.DataFrame] = []
    manifest: dict[str, Any] = {
        "base_config": str(CONFIG_PATH.resolve()),
        "num_rx_ant": int(get_cfg(BASE_CFG, "channel.num_rx_ant", 0)),
        "dmrs_cases": {},
        "eval_num_users": [1, 2, 3, 4],
        "seeds": [],
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
        seed = _seed_from_path(path)
        df = pd.read_csv(path)
        if seed is None and "seed" in df.columns and not df["seed"].dropna().empty:
            seed = int(df["seed"].dropna().iloc[0])
        if seed is None:
            seed = int(get_cfg(BASE_CFG, "system.seed", 7))
        df["seed"] = int(seed)
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
                "seed_runs": {},
            },
        )
        seed_manifest = variant_manifest["seed_runs"].setdefault(
            str(seed),
            {
                "checkpoint_path": str(RUN_ROOT / _seed_tag(seed) / dmrs_case / variant / "checkpoints" / "best.weights.h5"),
                "model_summary": _read_json(RUN_ROOT / _seed_tag(seed) / dmrs_case / variant / "metrics" / "model_summary.json"),
                "curves": {},
            },
        )
        seed_manifest["curves"][str(num_users)] = {
            "csv": str(path),
            "summary": str(EVAL_ROOT / _seed_tag(seed) / dmrs_case / f"{variant}_u{num_users}" / "metrics" / "evaluation_summary.json"),
        }

    if not frames:
        raise FileNotFoundError(f"No per-worker curve CSVs found under {CSV_ROOT}/{{1dmrs,2dmrs}}")

    combined = pd.concat(frames, ignore_index=True)
    CSV_ROOT.mkdir(parents=True, exist_ok=True)
    combined_path = CSV_ROOT / "comprehensive_curves.csv"
    combined.to_csv(combined_path, index=False)
    manifest["combined_csv"] = str(combined_path)
    manifest["seeds"] = sorted(int(x) for x in combined["seed"].dropna().unique().tolist()) if "seed" in combined.columns else []

    seed_summary = _seed_summary(combined)
    if not seed_summary.empty:
        seed_summary_path = CSV_ROOT / "comprehensive_seed_summary.csv"
        seed_summary.to_csv(seed_summary_path, index=False)
        manifest["seed_summary_csv"] = str(seed_summary_path)

    for dmrs_case, case_df in combined.groupby("dmrs_case"):
        case_path = CSV_ROOT / str(dmrs_case) / "comprehensive_curves.csv"
        case_path.parent.mkdir(parents=True, exist_ok=True)
        case_df.to_csv(case_path, index=False)
        case_seed_summary = _seed_summary(case_df)
        if not case_seed_summary.empty:
            case_seed_summary.to_csv(CSV_ROOT / str(dmrs_case) / "comprehensive_seed_summary.csv", index=False)
        if str(dmrs_case) in manifest["dmrs_cases"]:
            manifest["dmrs_cases"][str(dmrs_case)]["combined_csv"] = str(case_path)

    manifest_path = CSV_ROOT / "comprehensive_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    print(f"[MERGE] wrote {combined_path}")
    print(f"[MERGE] wrote {manifest_path}")


if __name__ == "__main__":
    main()
