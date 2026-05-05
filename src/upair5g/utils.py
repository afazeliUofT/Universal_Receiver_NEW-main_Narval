from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import yaml

from .compat import first_present_attr, resolve_attr, safe_call_variants


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_plain_data(payload), handle, indent=2, sort_keys=True)


def save_yaml(payload: dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(to_plain_data(payload), handle, sort_keys=False)


def to_plain_data(value: Any) -> Any:
    """Convert framework wrapper objects into JSON/YAML-safe containers."""
    if isinstance(value, Mapping):
        return {str(key): to_plain_data(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [to_plain_data(item) for item in value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_plain_data(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if tf.is_tensor(value):
        array = value.numpy()
        return array.item() if np.ndim(array) == 0 else array.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        try:
            return to_plain_data(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "as_dict"):
        try:
            return to_plain_data(value.as_dict())
        except Exception:
            pass
    return str(value)


def tensor7_to_btfnc(x: tf.Tensor) -> tf.Tensor:
    x_btfnu = tensor7_to_btfnu(x)
    users = x_btfnu.shape[-1]
    if users is not None and users != 1:
        raise ValueError(f"Expected a single user/stream, got {users}. Use tensor7_to_btfnu().")
    return tf.squeeze(x_btfnu, axis=-1)


def tensor7_to_btfnu(x: tf.Tensor) -> tf.Tensor:
    """Convert Sionna channel tensors to [B, T, F, Nr, U].

    Sionna represents channel estimates as
    [B, num_rx, num_rx_ant, num_tx, num_streams_per_tx, T, F].  The present
    experiments use a single gNB and single-layer users, but we keep the
    stream axis by flattening num_tx*num_streams_per_tx into U so the same
    helper works for 1--4 scheduled users.
    """
    x = tf.convert_to_tensor(x)
    if x.shape.rank != 7:
        raise ValueError(f"Expected rank-7 tensor, got rank {x.shape.rank}")

    # [B, num_rx_ant, num_tx, num_streams_per_tx, T, F]
    x = tf.squeeze(x, axis=1)
    # [B, T, F, num_rx_ant, num_tx, num_streams_per_tx]
    x = tf.transpose(x, [0, 4, 5, 1, 2, 3])
    shape = tf.shape(x)
    return tf.reshape(x, [shape[0], shape[1], shape[2], shape[3], shape[4] * shape[5]])


def y_to_btfnc(y: tf.Tensor) -> tf.Tensor:
    y = tf.convert_to_tensor(y)
    if y.shape.rank != 5:
        raise ValueError(f"Expected rank-5 tensor, got rank {y.shape.rank}")
    y = tf.squeeze(y, axis=1)  # [B, num_rx_ant, T, F]
    y = tf.transpose(y, [0, 2, 3, 1])  # [B, T, F, num_rx_ant]
    return y


def btfnc_to_tensor7(x: tf.Tensor) -> tf.Tensor:
    x = tf.convert_to_tensor(x)
    if x.shape.rank != 4:
        raise ValueError(f"Expected rank-4 tensor, got rank {x.shape.rank}")
    return btfnu_to_tensor7(x[..., tf.newaxis])


def btfnu_to_tensor7(x: tf.Tensor) -> tf.Tensor:
    x = tf.convert_to_tensor(x)
    if x.shape.rank != 5:
        raise ValueError(f"Expected rank-5 tensor [B,T,F,Nr,U], got rank {x.shape.rank}")
    x = tf.transpose(x, [0, 3, 4, 1, 2])  # [B, Nr, U, T, F]
    x = tf.expand_dims(x, axis=1)  # num_rx
    x = tf.expand_dims(x, axis=4)  # num_streams_per_tx
    return x


def pad_user_dim(x: tf.Tensor, max_num_users: int) -> tf.Tensor:
    x = tf.convert_to_tensor(x)
    if x.shape.rank != 5:
        raise ValueError(f"Expected rank-5 tensor [B,T,F,Nr,U], got rank {x.shape.rank}")
    max_num_users = int(max_num_users)
    users = tf.shape(x)[-1]
    pad_users = tf.maximum(max_num_users - users, 0)
    paddings = tf.stack(
        [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, pad_users],
        ]
    )
    x = tf.pad(x, paddings)
    return x[..., :max_num_users]


def complex_to_ri_channels(x: tf.Tensor) -> tf.Tensor:
    x = tf.convert_to_tensor(x)
    return tf.concat([tf.math.real(x), tf.math.imag(x)], axis=-1)


def complex_sq_abs(x: tf.Tensor) -> tf.Tensor:
    x = tf.convert_to_tensor(x)
    if x.dtype.is_complex:
        xr = tf.math.real(x)
        xi = tf.math.imag(x)
        return xr * xr + xi * xi
    x = tf.cast(x, tf.float32)
    return x * x


def broadcast_no_feature(no: tf.Tensor, batch: tf.Tensor, time: tf.Tensor, freq: tf.Tensor) -> tf.Tensor:
    no = tf.cast(tf.convert_to_tensor(no), tf.float32)
    if no.shape.rank == 0:
        no = tf.fill([batch], no)
    else:
        no = tf.reshape(no, [-1])
        if tf.shape(no)[0] == 1:
            no = tf.tile(no, [batch])
    no = tf.reshape(no, [batch, 1, 1, 1])
    return tf.broadcast_to(no, [batch, time, freq, 1])


def broadcast_like_err(err_var: tf.Tensor, h_like: tf.Tensor) -> tf.Tensor:
    err_var = tf.cast(tf.convert_to_tensor(err_var), tf.float32)
    h_like = tf.cast(tf.math.real(tf.convert_to_tensor(h_like)), tf.float32)
    return err_var + tf.zeros_like(h_like)


def compute_nmse(h_true: tf.Tensor, h_hat: tf.Tensor, eps: float = 1e-9) -> tf.Tensor:
    h_true = tf.convert_to_tensor(h_true)
    h_hat = tf.convert_to_tensor(h_hat)
    num = tf.reduce_mean(complex_sq_abs(h_true - h_hat))
    den = tf.reduce_mean(complex_sq_abs(h_true)) + eps
    return tf.cast(num / den, tf.float32)


def flatten_bits(x: tf.Tensor) -> tf.Tensor:
    x = tf.convert_to_tensor(x)
    if x.dtype.is_floating:
        x = tf.cast(x > 0.5, tf.int32)
    elif x.dtype == tf.bool:
        x = tf.cast(x, tf.int32)
    else:
        x = tf.cast(x, tf.int32)
    return tf.reshape(x, [-1])


def compute_ber(bits_true: tf.Tensor, bits_hat: tf.Tensor) -> tf.Tensor:
    b_true = flatten_bits(bits_true)
    b_hat = flatten_bits(bits_hat)
    n = tf.minimum(tf.size(b_true), tf.size(b_hat))
    b_true = b_true[:n]
    b_hat = b_hat[:n]
    return tf.reduce_mean(tf.cast(tf.not_equal(b_true, b_hat), tf.float32))


def compute_bler_from_crc(tb_crc_status: tf.Tensor) -> tf.Tensor:
    crc = tf.convert_to_tensor(tb_crc_status)
    if crc.dtype != tf.bool:
        crc = tf.cast(crc > 0, tf.bool)
    return 1.0 - tf.reduce_mean(tf.cast(crc, tf.float32))


def infer_tx_signal_and_bits(tx_output: Any) -> tuple[tf.Tensor, tf.Tensor | None]:
    if isinstance(tx_output, (tuple, list)):
        complex_tensors = []
        non_complex_tensors = []
        for item in tx_output:
            tensor = tf.convert_to_tensor(item)
            if tensor.dtype.is_complex:
                complex_tensors.append(tensor)
            else:
                non_complex_tensors.append(tensor)
        if not complex_tensors:
            raise ValueError("Could not identify the complex transmit signal in transmitter output.")
        x = complex_tensors[0]
        bits = non_complex_tensors[0] if non_complex_tensors else None
        return x, bits
    tensor = tf.convert_to_tensor(tx_output)
    return tensor, None


def infer_channel_output(channel_output: Any) -> tuple[tf.Tensor, tf.Tensor]:
    if not isinstance(channel_output, (tuple, list)) or len(channel_output) < 2:
        raise ValueError("Channel output must contain at least y and h.")
    y = tf.convert_to_tensor(channel_output[0])
    h = tf.convert_to_tensor(channel_output[1])
    return y, h


def infer_receiver_output(receiver_output: Any) -> tuple[tf.Tensor, tf.Tensor | None]:
    if isinstance(receiver_output, (tuple, list)):
        if len(receiver_output) >= 2:
            return tf.convert_to_tensor(receiver_output[0]), tf.convert_to_tensor(receiver_output[1])
        if len(receiver_output) == 1:
            return tf.convert_to_tensor(receiver_output[0]), None
    return tf.convert_to_tensor(receiver_output), None


def call_transmitter(transmitter: Any, batch_size: int) -> tuple[tf.Tensor, tf.Tensor | None]:
    out = safe_call_variants(transmitter, batch_size)
    return infer_tx_signal_and_bits(out)


def call_channel(channel: Any, x: tf.Tensor, no: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    try:
        out = channel(x, no)
    except (tf.errors.ResourceExhaustedError, MemoryError):
        raise
    except Exception:
        out = safe_call_variants(channel, x, no)
    return infer_channel_output(out)


def call_receiver(receiver: Any, y: tf.Tensor, no: tf.Tensor, h: tf.Tensor | None = None) -> tuple[tf.Tensor, tf.Tensor | None]:
    attempts = []
    if h is None:
        # Sionna/Keras receivers expect a single inputs object. Trying split
        # positional tensors first can execute an invalid graph path and spam
        # TensorFlow transpose warnings before falling back.
        attempts = [
            lambda: receiver([y, no]),
            lambda: receiver((y, no)),
            lambda: receiver(y, no),
        ]
    else:
        # Perfect-CSI PUSCHReceiver on Narval/Sionna works cleanly as
        # positional (y, no, h). Other forms are compatibility fallbacks.
        attempts = [
            lambda: receiver(y, no, h),
            lambda: receiver([y, no, h]),
            lambda: receiver((y, no, h)),
            lambda: receiver(y, h, no),
            lambda: receiver([y, h, no]),
            lambda: receiver((y, h, no)),
        ]
    last_err = None
    for attempt in attempts:
        try:
            return infer_receiver_output(attempt())
        except (tf.errors.ResourceExhaustedError, MemoryError):
            raise
        except Exception as err:  # pragma: no cover - runtime compatibility helper
            last_err = err
    raise RuntimeError("All receiver calling conventions failed.") from last_err


def tf_float(value: Any) -> tf.Tensor:
    return tf.cast(tf.convert_to_tensor(value), tf.float32)


def infer_num_bits_per_symbol(tx: Any, default: int = 4) -> int:
    value = first_present_attr(tx, ["_upair_num_bits_per_symbol", "_num_bits_per_symbol", "num_bits_per_symbol"], None)
    if value is None:
        pusch_configs = first_present_attr(tx, ["_upair_pusch_configs", "pusch_configs", "_pusch_configs"], None)
        if isinstance(pusch_configs, (list, tuple)) and pusch_configs:
            tb = getattr(pusch_configs[0], "tb", None)
            value = first_present_attr(tb, ["num_bits_per_symbol", "_num_bits_per_symbol"], None)
        elif pusch_configs is not None:
            tb = getattr(pusch_configs, "tb", None)
            value = first_present_attr(tb, ["num_bits_per_symbol", "_num_bits_per_symbol"], None)
    if value is None:
        value = default
    try:
        return int(value)
    except Exception:
        return default


def infer_coderate(tx: Any, default: float = 0.5) -> float:
    value = first_present_attr(tx, ["_upair_coderate", "_coderate", "coderate"], None)
    if value is None:
        pusch_configs = first_present_attr(tx, ["_upair_pusch_configs", "pusch_configs", "_pusch_configs"], None)
        if isinstance(pusch_configs, (list, tuple)) and pusch_configs:
            tb = getattr(pusch_configs[0], "tb", None)
            value = first_present_attr(tb, ["target_coderate", "_target_coderate", "coderate", "_coderate"], None)
        elif pusch_configs is not None:
            tb = getattr(pusch_configs, "tb", None)
            value = first_present_attr(tb, ["target_coderate", "_target_coderate", "coderate", "_coderate"], None)
    if value is None:
        value = default
    try:
        return float(value)
    except Exception:
        return default


def ebno_db_to_no(
    ebno_db: float | tf.Tensor,
    tx: Any | None = None,
    resource_grid: Any | None = None,
    bits_per_symbol: int | None = None,
    coderate: float | None = None,
) -> tf.Tensor:
    ebnodb2no = resolve_attr(["sionna.phy.utils", "sionna.utils"], "ebnodb2no")
    if bits_per_symbol is None:
        bits_per_symbol = infer_num_bits_per_symbol(tx)
    if coderate is None:
        coderate = infer_coderate(tx)
    if resource_grid is None and tx is not None:
        resource_grid = first_present_attr(tx, ["resource_grid", "_resource_grid"], None)
    attempts = [
        lambda: ebnodb2no(ebno_db, bits_per_symbol, coderate, resource_grid),
        lambda: ebnodb2no(tf.convert_to_tensor(ebno_db, tf.float32), bits_per_symbol, coderate, resource_grid),
        lambda: ebnodb2no(ebno_db, bits_per_symbol, coderate),
        lambda: ebnodb2no(tf.convert_to_tensor(ebno_db, tf.float32), bits_per_symbol, coderate),
    ]
    last_err = None
    for attempt in attempts:
        try:
            no = attempt()
            return tf.cast(no, tf.float32)
        except Exception as err:  # pragma: no cover - runtime compatibility helper
            last_err = err
    # Fallback approximation if Sionna signature changes
    ebno_db = tf.cast(tf.convert_to_tensor(ebno_db), tf.float32)
    no = tf.pow(tf.constant(10.0, tf.float32), -ebno_db / 10.0)
    if last_err is not None:
        tf.print("[WARN] Falling back to approximate Eb/N0->No conversion due to:", last_err)
    return no


def serializable_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (np.floating, np.integer)):
            converted[key] = value.item()
        elif tf.is_tensor(value):
            converted[key] = float(value.numpy())
        else:
            converted[key] = value
    return converted
