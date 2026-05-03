from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "TWC_plots_comprehensive"
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from upair5g.config import get_cfg, load_config  # noqa: E402


def _csv_dir() -> Path:
    cfg = load_config(ROOT / "configs" / "twc_comprehensive_mu32_base.yaml")
    return FIG_DIR / f"csv_rx{int(get_cfg(cfg, 'channel.num_rx_ant', 0))}"


CSV_DIR = _csv_dir()
CURVES_PATH = CSV_DIR / "comprehensive_curves.csv"
MANIFEST_PATH = CSV_DIR / "comprehensive_manifest.json"

RECEIVERS = [
    "baseline_ls_lmmse",
    "baseline_ls_2dlmmse_lmmse",
    "baseline_ddcpe_ls_lmmse",
    "upair5g_lmmse",
    "perfect_csi_lmmse",
]
NMSE_RECEIVERS = [
    "baseline_ls_lmmse",
    "baseline_ls_2dlmmse_lmmse",
    "baseline_ddcpe_ls_lmmse",
    "upair5g_lmmse",
]

LABELS = {
    "baseline_ls_lmmse": "LS",
    "baseline_ls_2dlmmse_lmmse": "LS + 2D LMMSE",
    "baseline_ddcpe_ls_lmmse": "DD-CPE + LS",
    "upair5g_lmmse": "UPAIR-5G",
    "perfect_csi_lmmse": "Perfect CSI",
}

STYLES = {
    "baseline_ls_lmmse": ("#1f77b4", "--", "o"),
    "baseline_ls_2dlmmse_lmmse": ("#2ca02c", ":", "D"),
    "baseline_ddcpe_ls_lmmse": ("#ff7f0e", "-.", "s"),
    "upair5g_lmmse": ("#d62728", "-", "^"),
    "perfect_csi_lmmse": ("#9467bd", "-", "v"),
}


def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    with open(MANIFEST_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _plot_receiver(ax: plt.Axes, sub: pd.DataFrame, receiver: str, metric: str) -> None:
    color, linestyle, marker = STYLES[receiver]
    sub = sub.sort_values("ebno_db")
    x = sub["ebno_db"].to_numpy(dtype=float)
    y = sub[metric].to_numpy(dtype=float)
    mask = np.isfinite(y) & (y > 0.0)
    if not np.any(mask):
        return
    ax.semilogy(x[mask], y[mask], color=color, linestyle=linestyle, linewidth=2.0, label=LABELS[receiver])
    reliable_col = f"reliable_{metric}"
    marker_mask = mask
    if reliable_col in sub:
        marker_mask = mask & sub[reliable_col].fillna(False).to_numpy(dtype=bool)
    ax.semilogy(
        x[marker_mask],
        y[marker_mask],
        linestyle="None",
        marker=marker,
        markersize=5.2,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.2,
    )


def _format(ax: plt.Axes, ylabel: str) -> None:
    ax.set_xlabel(r"$E_b/N_0$ (dB)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.25)


def _save(fig: plt.Figure, stem: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=250, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _panel_by_user(df: pd.DataFrame, variant: str, metric: str, receivers: list[str], stem: str, out_dir: Path, title_prefix: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2), sharey=True)
    axes_flat = axes.reshape(-1)
    handles = labels = None
    for ax, num_users in zip(axes_flat, [1, 2, 3, 4]):
        panel = df[(df["variant"] == variant) & (df["num_users"] == num_users)]
        for receiver in receivers:
            sub = panel[panel["receiver"] == receiver]
            if not sub.empty:
                _plot_receiver(ax, sub, receiver, metric)
        ax.set_title(f"{num_users} active user{'s' if num_users > 1 else ''}")
        _format(ax, "BLER" if metric == "bler" else "NMSE")
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    fig.suptitle(title_prefix)
    if handles and labels:
        fig.legend(handles, labels, loc="upper center", ncol=len(receivers), frameon=True, bbox_to_anchor=(0.5, 1.01))
    _save(fig, stem, out_dir)


def _snr_at_target(sub: pd.DataFrame, target: float = 1e-2) -> float | None:
    sub = sub.sort_values("ebno_db")
    x = sub["ebno_db"].to_numpy(dtype=float)
    y = sub["bler"].to_numpy(dtype=float)
    for i in range(len(x) - 1):
        y0, y1 = y[i], y[i + 1]
        if y0 >= target >= y1 and y0 > 0.0 and y1 > 0.0:
            a = (np.log10(target) - np.log10(y0)) / (np.log10(y1) - np.log10(y0))
            return float(x[i] + a * (x[i + 1] - x[i]))
    return None


def _variant_params(manifest: dict, dmrs_case: str, variant: str, fallback: int) -> int:
    try:
        value = manifest["dmrs_cases"][dmrs_case]["variants"][variant]["model_summary"]["num_trainable_params"]
        return int(value)
    except Exception:
        pass
    try:
        value = manifest["variants"][variant]["model_summary"]["num_trainable_params"]
        return int(value)
    except Exception:
        return int(fallback)


def _ablation_summary(df: pd.DataFrame, manifest: dict, dmrs_case: str) -> pd.DataFrame:
    rows = []
    variants = sorted(df["variant"].dropna().unique().tolist())
    for order, variant in enumerate(variants, start=1):
        label = str(df.loc[df["variant"] == variant, "variant_label"].dropna().iloc[0])
        params = _variant_params(manifest, dmrs_case, variant, order)
        for num_users in sorted(df["num_users"].dropna().unique().astype(int).tolist()):
            panel = df[(df["variant"] == variant) & (df["num_users"] == num_users) & (df["receiver"] == "upair5g_lmmse")]
            if panel.empty:
                continue
            rows.append(
                {
                    "dmrs_case": dmrs_case,
                    "dmrs_label": str(panel["dmrs_label"].dropna().iloc[0]) if "dmrs_label" in panel and not panel["dmrs_label"].dropna().empty else dmrs_case,
                    "variant": variant,
                    "label": label,
                    "num_users": num_users,
                    "num_trainable_params": params,
                    "upair_snr_at_1e2": _snr_at_target(panel),
                    "upair_avg_nmse": float(panel["nmse"].dropna().mean()),
                }
            )
    return pd.DataFrame(rows)


def _plot_ablation(summary: pd.DataFrame, metric: str, stem: str, ylabel: str, out_dir: Path, title_prefix: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for num_users in sorted(summary["num_users"].unique().tolist()):
        sub = summary[summary["num_users"] == num_users].sort_values("num_trainable_params")
        y = sub[metric].to_numpy(dtype=float)
        x = sub["num_trainable_params"].to_numpy(dtype=float)
        mask = np.isfinite(y)
        if np.any(mask):
            ax.plot(x[mask], y[mask], marker="o", linewidth=2.0, label=f"{num_users} user{'s' if num_users > 1 else ''}")
    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel(ylabel)
    ax.set_title(title_prefix)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)
    _save(fig, stem, out_dir)


def _dmrs_cases(df: pd.DataFrame, manifest: dict) -> list[tuple[str, str]]:
    if "dmrs_case" not in df.columns:
        return [("default", "Default DMRS")]
    cases = sorted(str(x) for x in df["dmrs_case"].dropna().unique().tolist())
    result = []
    for case in cases:
        label = case
        try:
            label = str(manifest["dmrs_cases"][case]["label"])
        except Exception:
            sub = df[df["dmrs_case"] == case]
            if "dmrs_label" in sub and not sub["dmrs_label"].dropna().empty:
                label = str(sub["dmrs_label"].dropna().iloc[0])
        result.append((case, label))
    return result


def main() -> None:
    if not CURVES_PATH.exists():
        raise FileNotFoundError(f"Missing comprehensive curves CSV: {CURVES_PATH}")
    df = pd.read_csv(CURVES_PATH)
    manifest = _load_manifest()

    main_variant = "main_d96_b4_r2"
    all_summaries = []
    for dmrs_case, dmrs_label in _dmrs_cases(df, manifest):
        case_df = df if dmrs_case == "default" else df[df["dmrs_case"] == dmrs_case].copy()
        if case_df.empty:
            continue
        out_dir = FIG_DIR / dmrs_case
        title_prefix = f"{dmrs_label} case"
        _panel_by_user(case_df, main_variant, "bler", RECEIVERS, "Fig01_main_bler_by_user", out_dir, title_prefix)
        _panel_by_user(case_df, main_variant, "nmse", NMSE_RECEIVERS, "Fig02_main_nmse_by_user", out_dir, title_prefix)

        summary = _ablation_summary(case_df, manifest, dmrs_case)
        case_csv_dir = CSV_DIR / dmrs_case
        case_csv_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(case_csv_dir / "comprehensive_ablation_summary.csv", index=False)
        if not summary.empty:
            all_summaries.append(summary)
            _plot_ablation(
                summary,
                "upair_snr_at_1e2",
                "Fig03_ablation_snr_at_bler1e2",
                r"UPAIR $E_b/N_0$ at BLER $10^{-2}$ (dB)",
                out_dir,
                title_prefix,
            )
            _plot_ablation(
                summary,
                "upair_avg_nmse",
                "Fig04_ablation_average_nmse",
                "UPAIR average NMSE",
                out_dir,
                title_prefix,
            )

    if all_summaries:
        pd.concat(all_summaries, ignore_index=True).to_csv(CSV_DIR / "comprehensive_ablation_summary.csv", index=False)

    print(f"[COMPREHENSIVE] figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
