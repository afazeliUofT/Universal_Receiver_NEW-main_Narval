from __future__ import annotations

from typing import Any

import tensorflow as tf

from .config import get_cfg


def _sample_phase_profile(
    batch_size: int,
    num_symbols: tf.Tensor,
    impair_cfg: dict[str, Any],
    training: bool,
) -> tf.Tensor:
    batch_size = int(batch_size)
    num_symbols = tf.cast(num_symbols, tf.int32)
    t = tf.cast(tf.range(num_symbols), tf.float32)[tf.newaxis, :]  # [1, T]

    cpe_sigma = float(impair_cfg["cpe_sigma_rad"])
    rw_sigma = float(impair_cfg["rw_sigma_rad"])

    phi0 = tf.random.normal([batch_size, 1], stddev=cpe_sigma)

    if training or "slope_range_rad" in impair_cfg:
        low, high = impair_cfg["slope_range_rad"]
        slope = tf.random.uniform([batch_size, 1], minval=float(low), maxval=float(high))
    else:
        slope = tf.fill([batch_size, 1], float(impair_cfg["slope_rad"]))

    centered_t = t - tf.cast(num_symbols - 1, tf.float32) / 2.0
    linear = slope * centered_t / tf.maximum(tf.cast(num_symbols - 1, tf.float32), 1.0)

    rw_increments = tf.random.normal([batch_size, num_symbols], stddev=rw_sigma)
    rw = tf.cumsum(rw_increments, axis=1)

    phase = phi0 + linear + rw
    return phase  # [B, T]


def apply_symbol_phase_impairment(
    y: tf.Tensor,
    h: tf.Tensor | None,
    cfg: dict[str, Any],
    training: bool,
) -> tuple[tf.Tensor, tf.Tensor | None]:
    if not bool(get_cfg(cfg, "impairments.enabled", True)):
        return y, h

    impair_cfg = cfg["impairments"]["train" if training else "eval"]
    num_symbols = tf.shape(y)[-2]
    batch_size = tf.shape(y)[0]
    phase = _sample_phase_profile(batch_size, num_symbols, impair_cfg, training=training)

    phasor_y = tf.cast(tf.exp(tf.complex(tf.zeros_like(phase), phase)), y.dtype)
    phasor_y = tf.reshape(phasor_y, [tf.shape(y)[0], 1, 1, tf.shape(y)[-2], 1])
    y_out = y * phasor_y

    if h is None:
        return y_out, None

    phasor_h = tf.cast(tf.exp(tf.complex(tf.zeros_like(phase), phase)), h.dtype)
    phasor_h = tf.reshape(phasor_h, [tf.shape(h)[0], 1, 1, 1, 1, tf.shape(h)[-2], 1])
    h_out = h * phasor_h
    return y_out, h_out
