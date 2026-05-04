from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from .builders import build_ls_estimator, build_receiver, get_resource_grid
from .compat import first_present_attr, instantiate_filtered, resolve_attr
from .config import get_cfg
from .impairments import apply_symbol_phase_impairment
from .phase_aware import DecisionDirectedCPEEstimator
from .utils import call_channel, call_transmitter, infer_num_bits_per_symbol, tensor7_to_btfnu

PROPOSED_RECEIVER = "upair5g_lmmse"
PERFECT_RECEIVER = "perfect_csi_lmmse"
BASELINE_DDCPE_LS_LMMSE = "baseline_ddcpe_ls_lmmse"
DEFAULT_ENABLED_RECEIVERS = [
    "baseline_ls_lmmse",
    PROPOSED_RECEIVER,
    PERFECT_RECEIVER,
]

EMPIRICAL_COVARIANCE_CACHE_VERSION = 3


def enabled_receivers_from_cfg(cfg: dict[str, Any]) -> list[str]:
    configured = get_cfg(cfg, "baselines.enabled_receivers", None)
    if configured is None:
        return list(DEFAULT_ENABLED_RECEIVERS)
    return [str(name) for name in configured]


def classical_receivers_from_cfg(cfg: dict[str, Any]) -> list[str]:
    enabled = enabled_receivers_from_cfg(cfg)
    return [name for name in enabled if name not in {PROPOSED_RECEIVER, PERFECT_RECEIVER}]


def wants_receiver(cfg: dict[str, Any], receiver_name: str) -> bool:
    return receiver_name in enabled_receivers_from_cfg(cfg)


def _covariance_cache_path(paths: dict[str, Path], cfg: dict[str, Any]) -> Path:
    cache_name = str(get_cfg(cfg, "baselines.covariance_estimation.cache_name", "empirical_covariances.npz"))
    return paths["artifacts"] / cache_name


def _use_training_impairments_for_covariance(cfg: dict[str, Any]) -> bool:
    return bool(get_cfg(cfg, "baselines.covariance_estimation.use_training_impairments", False))


def _load_cached_empirical_covariances(
    cache_path: Path,
    cfg: dict[str, Any],
) -> dict[str, tf.Tensor] | None:
    if not cache_path.exists():
        return None

    payload = np.load(cache_path)

    version_arr = np.asarray(payload.get("cache_format_version", np.asarray([0], dtype=np.int32)))
    cache_version = int(version_arr.reshape(-1)[0]) if version_arr.size > 0 else 0
    if cache_version != EMPIRICAL_COVARIANCE_CACHE_VERSION:
        return None

    cached_use_training = np.asarray(payload.get("use_training_impairments", np.asarray([0], dtype=np.int32)))
    cached_use_training_flag = bool(int(cached_use_training.reshape(-1)[0])) if cached_use_training.size > 0 else False
    requested_use_training_flag = _use_training_impairments_for_covariance(cfg)
    if cached_use_training_flag != requested_use_training_flag:
        return None

    return {
        "cov_mat_time": tf.convert_to_tensor(payload["cov_mat_time"], dtype=tf.complex64),
        "cov_mat_freq": tf.convert_to_tensor(payload["cov_mat_freq"], dtype=tf.complex64),
        "cov_mat_space": tf.convert_to_tensor(payload["cov_mat_space"], dtype=tf.complex64),
        "cache_path": tf.convert_to_tensor(str(cache_path)),
    }


def _rows_count(x: tf.Tensor) -> float:
    static_rows = x.shape[0]
    if static_rows is not None:
        return float(static_rows)
    return float(tf.shape(x)[0].numpy())


def _hermitianize_and_regularize(mat: tf.Tensor, eps: float) -> tf.Tensor:
    mat = tf.cast(tf.convert_to_tensor(mat), tf.complex64)
    mat = 0.5 * (mat + tf.linalg.adjoint(mat))
    dim = tf.shape(mat)[0]
    eye = tf.eye(dim, dtype=mat.dtype)
    return mat + tf.cast(float(eps), mat.dtype) * eye


def _normalize_trace(mat: tf.Tensor) -> tf.Tensor:
    mat = tf.cast(tf.convert_to_tensor(mat), tf.complex64)
    dim = tf.cast(tf.shape(mat)[0], tf.float32)
    avg_diag = tf.math.real(tf.linalg.trace(mat)) / tf.maximum(dim, 1.0)
    avg_diag = tf.maximum(avg_diag, 1e-9)
    return mat / tf.cast(avg_diag, mat.dtype)


def _covariance_from_rows(rows: tf.Tensor) -> tuple[tf.Tensor, float]:
    rows = tf.cast(tf.convert_to_tensor(rows), tf.complex64)
    # rows has shape [num_samples, dim]. We want the covariance with entries
    # C[i, j] = E[h_i * conj(h_j)], which matches Sionna's covariance format.
    gram = tf.matmul(tf.transpose(rows), tf.math.conj(rows))
    return gram, _rows_count(rows)


def estimate_empirical_covariances(
    tx: Any,
    channel: Any,
    cfg: dict[str, Any],
    paths: dict[str, Path],
) -> dict[str, tf.Tensor]:
    cache_path = _covariance_cache_path(paths, cfg)
    reuse_cache = bool(get_cfg(cfg, "baselines.covariance_estimation.reuse_cache", True))
    if reuse_cache:
        cached = _load_cached_empirical_covariances(cache_path, cfg)
        if cached is not None:
            return cached

    num_batches = int(get_cfg(cfg, "baselines.covariance_estimation.num_batches", 8))
    batch_size = int(get_cfg(cfg, "baselines.covariance_estimation.batch_size", 32))
    use_training_impairments = _use_training_impairments_for_covariance(cfg)
    regularization = float(get_cfg(cfg, "baselines.covariance_estimation.diagonal_loading", 1e-4))
    normalize_trace = bool(get_cfg(cfg, "baselines.covariance_estimation.normalize_trace", False))

    acc_t: tf.Tensor | None = None
    acc_f: tf.Tensor | None = None
    acc_s: tf.Tensor | None = None
    count_t = 0.0
    count_f = 0.0
    count_s = 0.0

    covariance_mode = "training-impairment-aware" if use_training_impairments else "clean-channel"
    print(
        "[BASELINES] Estimating empirical LMMSE covariances "
        f"with num_batches={num_batches} batch_size={batch_size} "
        f"mode={covariance_mode}"
    )

    for _ in range(num_batches):
        x, _ = call_transmitter(tx, batch_size)
        y, h = call_channel(channel, x, tf.constant(0.0, tf.float32))
        if use_training_impairments:
            _, h = apply_symbol_phase_impairment(y, h, cfg, training=True)
        if h is None:
            raise ValueError("Channel did not return the true channel tensor needed for covariance estimation.")

        h_btfnu = tensor7_to_btfnu(h)  # [B, T, F, Nr, U]
        num_time = h_btfnu.shape[1] or int(tf.shape(h_btfnu)[1].numpy())
        num_freq = h_btfnu.shape[2] or int(tf.shape(h_btfnu)[2].numpy())
        num_rx_ant = h_btfnu.shape[3] or int(tf.shape(h_btfnu)[3].numpy())

        rows_t = tf.reshape(tf.transpose(h_btfnu, [0, 3, 4, 2, 1]), [-1, int(num_time)])
        rows_f = tf.reshape(tf.transpose(h_btfnu, [0, 3, 4, 1, 2]), [-1, int(num_freq)])
        rows_s = tf.reshape(tf.transpose(h_btfnu, [0, 1, 2, 4, 3]), [-1, int(num_rx_ant)])

        gram_t, rows_count_t = _covariance_from_rows(rows_t)
        gram_f, rows_count_f = _covariance_from_rows(rows_f)
        gram_s, rows_count_s = _covariance_from_rows(rows_s)

        acc_t = gram_t if acc_t is None else acc_t + gram_t
        acc_f = gram_f if acc_f is None else acc_f + gram_f
        acc_s = gram_s if acc_s is None else acc_s + gram_s
        count_t += rows_count_t
        count_f += rows_count_f
        count_s += rows_count_s

    if acc_t is None or acc_f is None or acc_s is None:
        raise RuntimeError("Failed to accumulate covariance matrices.")

    cov_mat_time = acc_t / tf.cast(max(count_t, 1.0), acc_t.dtype)
    cov_mat_freq = acc_f / tf.cast(max(count_f, 1.0), acc_f.dtype)
    cov_mat_space = acc_s / tf.cast(max(count_s, 1.0), acc_s.dtype)

    cov_mat_time = _hermitianize_and_regularize(cov_mat_time, regularization)
    cov_mat_freq = _hermitianize_and_regularize(cov_mat_freq, regularization)
    cov_mat_space = _hermitianize_and_regularize(cov_mat_space, regularization)

    if normalize_trace:
        cov_mat_time = _normalize_trace(cov_mat_time)
        cov_mat_freq = _normalize_trace(cov_mat_freq)
        cov_mat_space = _normalize_trace(cov_mat_space)

    np.savez_compressed(
        cache_path,
        cache_format_version=np.asarray([EMPIRICAL_COVARIANCE_CACHE_VERSION], dtype=np.int32),
        cov_mat_time=np.asarray(cov_mat_time.numpy()),
        cov_mat_freq=np.asarray(cov_mat_freq.numpy()),
        cov_mat_space=np.asarray(cov_mat_space.numpy()),
        num_batches=np.asarray([num_batches], dtype=np.int32),
        batch_size=np.asarray([batch_size], dtype=np.int32),
        use_training_impairments=np.asarray([int(use_training_impairments)], dtype=np.int32),
    )

    return {
        "cov_mat_time": cov_mat_time,
        "cov_mat_freq": cov_mat_freq,
        "cov_mat_space": cov_mat_space,
        "cache_path": tf.convert_to_tensor(str(cache_path)),
    }


def _sanitize_lmmse_order(order: str, use_spatial_smoothing: bool) -> str:
    order = str(order)
    if use_spatial_smoothing:
        return order
    tokens = [token for token in order.split("-") if token != "s"]
    return "-".join(tokens) if tokens else "f-t"


def build_empirical_lmmse_interpolator(
    tx: Any,
    channel: Any,
    cfg: dict[str, Any],
    paths: dict[str, Path],
) -> Any:
    LMMSEInterpolator = resolve_attr(["sionna.phy.ofdm", "sionna.ofdm"], "LMMSEInterpolator")
    resource_grid = get_resource_grid(tx)
    pilot_pattern = first_present_attr(resource_grid, ["pilot_pattern", "_pilot_pattern"], None)
    if pilot_pattern is None:
        raise AttributeError("Could not locate pilot_pattern in resource_grid.")

    covariances = estimate_empirical_covariances(tx=tx, channel=channel, cfg=cfg, paths=paths)
    use_spatial_smoothing = bool(get_cfg(cfg, "baselines.covariance_estimation.use_spatial_smoothing", False))
    order = _sanitize_lmmse_order(
        str(get_cfg(cfg, "baselines.covariance_estimation.order", "f-t-s")),
        use_spatial_smoothing=use_spatial_smoothing,
    )
    if use_spatial_smoothing and "s" not in order.split("-"):
        raise ValueError(
            "baselines.covariance_estimation.use_spatial_smoothing is true, "
            "but baselines.covariance_estimation.order does not include 's'."
        )

    kwargs = {
        "pilot_pattern": pilot_pattern,
        "cov_mat_time": covariances["cov_mat_time"],
        "cov_mat_freq": covariances["cov_mat_freq"],
        "order": order,
    }
    if use_spatial_smoothing:
        kwargs["cov_mat_space"] = covariances["cov_mat_space"]

    try:
        return instantiate_filtered(LMMSEInterpolator, **kwargs)
    except Exception as exc:
        if use_spatial_smoothing:
            raise RuntimeError(
                "Spatial smoothing was requested for the empirical LMMSE baseline, "
                "but Sionna could not construct the spatial LMMSE interpolator. "
                "Refusing to fall back to a non-spatial baseline."
            ) from exc
        kwargs.pop("cov_mat_space", None)
        kwargs["order"] = _sanitize_lmmse_order(order, use_spatial_smoothing=False)
        return instantiate_filtered(LMMSEInterpolator, **kwargs)


def _build_ddcpe_estimator(tx: Any, cfg: dict[str, Any], base_estimator: Any) -> DecisionDirectedCPEEstimator:
    return DecisionDirectedCPEEstimator(
        base_estimator=base_estimator,
        resource_grid=get_resource_grid(tx),
        bits_per_symbol=infer_num_bits_per_symbol(tx, default=6),
        cfg=cfg,
    )


def build_classical_baseline_suite(
    cfg: dict[str, Any],
    tx: Any,
    channel: Any,
    paths: dict[str, Path],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    enabled = classical_receivers_from_cfg(cfg)
    receivers: dict[str, Any] = {}
    estimators: dict[str, Any] = {}
    artifacts: dict[str, str] = {}

    if "baseline_ls_lmmse" in enabled:
        estimator = build_ls_estimator(tx, cfg, interpolation_type="lin")
        estimators["baseline_ls_lmmse"] = estimator
        receivers["baseline_ls_lmmse"] = build_receiver(tx, cfg, channel_estimator=estimator, perfect_csi=False)

    if "baseline_ls_timeavg_lmmse" in enabled:
        estimator = build_ls_estimator(tx, cfg, interpolation_type="lin_time_avg")
        estimators["baseline_ls_timeavg_lmmse"] = estimator
        receivers["baseline_ls_timeavg_lmmse"] = build_receiver(tx, cfg, channel_estimator=estimator, perfect_csi=False)

    if "baseline_ls_2dlmmse_lmmse" in enabled:
        interpolator = build_empirical_lmmse_interpolator(tx=tx, channel=channel, cfg=cfg, paths=paths)
        estimator = build_ls_estimator(tx, cfg, interpolator=interpolator)
        estimators["baseline_ls_2dlmmse_lmmse"] = estimator
        receivers["baseline_ls_2dlmmse_lmmse"] = build_receiver(tx, cfg, channel_estimator=estimator, perfect_csi=False)
        artifacts["empirical_covariances"] = str(_covariance_cache_path(paths, cfg))

    if BASELINE_DDCPE_LS_LMMSE in enabled:
        base_estimator = build_ls_estimator(tx, cfg, interpolation_type="lin")
        estimator = _build_ddcpe_estimator(tx=tx, cfg=cfg, base_estimator=base_estimator)
        estimators[BASELINE_DDCPE_LS_LMMSE] = estimator
        receivers[BASELINE_DDCPE_LS_LMMSE] = build_receiver(tx, cfg, channel_estimator=estimator, perfect_csi=False)

    return receivers, estimators, artifacts
