from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_cfg(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    cursor: Any = cfg
    for key in path.split("."):
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


def set_cfg(cfg: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    cursor = cfg
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value


def project_root() -> Path:
    return Path.cwd()


def output_dir(cfg: dict[str, Any]) -> Path:
    root = Path(get_cfg(cfg, "experiment.output_root", "outputs"))
    name = str(get_cfg(cfg, "experiment.name", "experiment"))
    return project_root() / root / name


def ensure_output_tree(cfg: dict[str, Any]) -> dict[str, Path]:
    root = output_dir(cfg)
    paths = {
        "root": root,
        "checkpoints": root / "checkpoints",
        "plots": root / "plots",
        "metrics": root / "metrics",
        "artifacts": root / "artifacts",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
