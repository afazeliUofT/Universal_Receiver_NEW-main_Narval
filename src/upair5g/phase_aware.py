from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf

from .builders import extract_pilot_mask
from .compat import safe_call_variants
from .config import get_cfg
from .utils import btfnu_to_tensor7, infer_num_bits_per_symbol, tensor7_to_btfnu, y_to_btfnc


class DecisionDirectedCPEEstimator(tf.keras.layers.Layer):
    """
    Classical phase-aware channel estimator for the current PUSCH setup.

    It wraps a standard Sionna LS-based estimator, performs single-stream
    MMSE/MRC equalization, estimates one residual common phase per OFDM symbol
    from hard QAM decisions over data REs, smooths that phase profile over time,
    and rotates the base channel estimate accordingly.

    The receiver stays fully feed-forward and slot-local.
    """

    def __init__(
        self,
        base_estimator: Any,
        resource_grid: Any,
        cfg: dict[str, Any],
        bits_per_symbol: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(trainable=False, name="decision_directed_cpe_estimator", **kwargs)
        self.base_estimator = base_estimator
        self.cfg = cfg
        self.bits_per_symbol = int(bits_per_symbol if bits_per_symbol is not None else infer_num_bits_per_symbol(resource_grid, default=6))
        self.num_iterations = int(get_cfg(cfg, "baselines.phase_aware_ddcpe.num_iterations", 2))
        self.weight_power = float(get_cfg(cfg, "baselines.phase_aware_ddcpe.weight_power", 1.0))
        self.max_abs_residual_rad = float(get_cfg(cfg, "baselines.phase_aware_ddcpe.max_abs_residual_rad", 0.6))
        self.anchor_to_dmrs = bool(get_cfg(cfg, "baselines.phase_aware_ddcpe.anchor_to_dmrs", True))
        self.eps = 1e-6

        smoothing_kernel = get_cfg(cfg, "baselines.phase_aware_ddcpe.smoothing_kernel", [1.0, 2.0, 3.0, 2.0, 1.0])
        kernel = np.asarray([float(x) for x in smoothing_kernel], dtype=np.float32)
        if kernel.ndim != 1 or kernel.size == 0:
            kernel = np.asarray([1.0], dtype=np.float32)
        kernel = np.maximum(kernel, 0.0)
        if float(kernel.sum()) <= 0.0:
            kernel = np.asarray([1.0], dtype=np.float32)
        kernel = kernel / float(kernel.sum())
        self._smoothing_kernel = tf.constant(kernel.reshape([-1, 1, 1]), dtype=tf.float32)

        pilot_mask = tf.cast(extract_pilot_mask(resource_grid), tf.float32)  # [T, F, 1]
        self.pilot_mask = pilot_mask
        self.data_mask = tf.maximum(1.0 - pilot_mask, 0.0)

        dmrs_per_symbol = tf.reduce_sum(pilot_mask[..., 0], axis=1).numpy()
        self.dmrs_symbol_index: int | None = None
        if dmrs_per_symbol.size > 0 and float(np.max(dmrs_per_symbol)) > 0.0:
            self.dmrs_symbol_index = int(np.argmax(dmrs_per_symbol))

    def _parse_inputs(self, inputs: Any, *args: Any) -> tuple[tf.Tensor, tf.Tensor]:
        if isinstance(inputs, (tuple, list)):
            if len(inputs) < 2:
                raise ValueError("Expected at least y and no for phase-aware estimator.")
            y, no = inputs[0], inputs[1]
        elif len(args) >= 1:
            y, no = inputs, args[0]
        else:
            raise ValueError("Could not parse estimator inputs.")
        return tf.convert_to_tensor(y), tf.convert_to_tensor(no)

    def _call_base_estimator(self, y: tf.Tensor, no: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        out = safe_call_variants(self.base_estimator, y, no)
        if not isinstance(out, (tuple, list)) or len(out) < 2:
            raise ValueError("Base estimator must return (h_hat, err_var).")
        return tf.convert_to_tensor(out[0]), tf.convert_to_tensor(out[1])

    def _broadcast_no(self, no: tf.Tensor, batch_size: tf.Tensor) -> tf.Tensor:
        no = tf.cast(tf.convert_to_tensor(no), tf.float32)
        if no.shape.rank == 0:
            no = tf.fill([batch_size], no)
        else:
            no = tf.reshape(no, [-1])
            if no.shape.rank == 1 and no.shape[0] == 1:
                no = tf.tile(no, [batch_size])
            else:
                dyn = tf.shape(no)[0]
                no = tf.cond(tf.equal(dyn, 1), lambda: tf.tile(no, [batch_size]), lambda: no)
        return tf.reshape(no, [batch_size, 1, 1])

    def _equalize(self, y_btfnc: tf.Tensor, h_btfnu: tf.Tensor, no: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(y_btfnc)[0]
        no_bc = self._broadcast_no(no, batch_size)
        num_users = tf.shape(h_btfnu)[-1]
        gram = tf.einsum("btfru,btfrv->btfuv", tf.math.conj(h_btfnu), h_btfnu)
        eye = tf.eye(num_users, dtype=gram.dtype)[tf.newaxis, tf.newaxis, tf.newaxis, :, :]
        system = gram + tf.cast(no_bc[..., tf.newaxis, tf.newaxis], gram.dtype) * eye
        rhs = tf.einsum("btfru,btfr->btfu", tf.math.conj(h_btfnu), y_btfnc)
        sol = tf.linalg.solve(system, rhs[..., tf.newaxis])
        return sol[..., 0]

    def _hard_slice_square_qam(self, x: tf.Tensor) -> tf.Tensor:
        order = 2 ** int(self.bits_per_symbol)
        sqrt_order = int(round(order ** 0.5))
        if sqrt_order * sqrt_order != order:
            xr = tf.math.real(x)
            xh = tf.where(xr >= 0.0, tf.ones_like(xr), -tf.ones_like(xr))
            return tf.complex(xh, tf.zeros_like(xh))

        levels = tf.cast(tf.range(-(sqrt_order - 1), sqrt_order, delta=2), tf.float32)
        norm = tf.sqrt((2.0 / 3.0) * tf.cast(order - 1, tf.float32))

        xr = tf.math.real(x) * norm
        xi = tf.math.imag(x) * norm

        xr_idx = tf.argmin(tf.abs(xr[..., tf.newaxis] - levels), axis=-1, output_type=tf.int32)
        xi_idx = tf.argmin(tf.abs(xi[..., tf.newaxis] - levels), axis=-1, output_type=tf.int32)

        xr_hat = tf.gather(levels, xr_idx) / norm
        xi_hat = tf.gather(levels, xi_idx) / norm
        return tf.complex(xr_hat, xi_hat)

    def _unwrap_phase(self, phase: tf.Tensor) -> tf.Tensor:
        phase = tf.cast(phase, tf.float32)
        if phase.shape.rank not in {2, 3}:
            raise ValueError(f"Expected phase rank 2 or 3, got {phase.shape.rank}.")
        if phase.shape[1] == 1:
            return phase
        diff = phase[:, 1:, ...] - phase[:, :-1, ...]
        diff_wrapped = tf.math.angle(tf.exp(tf.complex(tf.zeros_like(diff), diff)))
        start = phase[:, :1, ...]
        return tf.concat([start, start + tf.cumsum(diff_wrapped, axis=1)], axis=1)

    def _smooth_phase(self, phase: tf.Tensor) -> tf.Tensor:
        kernel_len = int(self._smoothing_kernel.shape[0]) if self._smoothing_kernel.shape[0] is not None else 1
        if kernel_len <= 1:
            return phase
        if phase.shape.rank == 2:
            x = tf.cast(phase[..., tf.newaxis], tf.float32)
            y = tf.nn.conv1d(x, filters=self._smoothing_kernel, stride=1, padding="SAME")
            return tf.squeeze(y, axis=-1)
        phase_but = tf.transpose(phase, [0, 2, 1])
        shape = tf.shape(phase_but)
        x = tf.reshape(phase_but, [shape[0] * shape[1], shape[2], 1])
        y = tf.nn.conv1d(x, filters=self._smoothing_kernel, stride=1, padding="SAME")
        y = tf.reshape(tf.squeeze(y, axis=-1), [shape[0], shape[1], shape[2]])
        return tf.transpose(y, [0, 2, 1])

    def _estimate_symbol_phase(self, x_eq: tf.Tensor, s_hat: tf.Tensor, h_btfnu: tf.Tensor) -> tf.Tensor:
        weight = tf.reduce_sum(tf.math.real(h_btfnu * tf.math.conj(h_btfnu)), axis=-2)
        weight = tf.pow(tf.maximum(weight, self.eps), self.weight_power)

        data_mask = self.data_mask[tf.newaxis, ..., 0, tf.newaxis]
        metric = tf.reduce_sum(tf.cast(data_mask * weight, x_eq.dtype) * x_eq * tf.math.conj(s_hat), axis=2)

        phase = self._unwrap_phase(tf.math.angle(metric))
        if self.anchor_to_dmrs and self.dmrs_symbol_index is not None:
            anchor = phase[:, self.dmrs_symbol_index : self.dmrs_symbol_index + 1, ...]
            phase = phase - anchor

        phase = self._smooth_phase(phase)
        if self.anchor_to_dmrs and self.dmrs_symbol_index is not None:
            anchor = phase[:, self.dmrs_symbol_index : self.dmrs_symbol_index + 1, ...]
            phase = phase - anchor

        phase = tf.clip_by_value(phase, -self.max_abs_residual_rad, self.max_abs_residual_rad)
        return phase

    def _apply_symbol_phase(self, h_btfnu: tf.Tensor, phase: tf.Tensor) -> tf.Tensor:
        phasor = tf.exp(tf.complex(tf.zeros_like(phase), phase))
        if phase.shape.rank == 2:
            phasor = phasor[..., tf.newaxis]
        phasor = tf.cast(phasor[:, :, tf.newaxis, tf.newaxis, :], h_btfnu.dtype)
        return h_btfnu * phasor

    def estimate_with_phase_tracking(self, y: tf.Tensor, no: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        h_hat, err_var = self._call_base_estimator(y, no)
        h_btfnu = tensor7_to_btfnu(h_hat)
        y_btfnc = y_to_btfnc(y)

        if self.num_iterations <= 0:
            return h_hat, err_var, h_hat

        refined = h_btfnu
        for _ in range(self.num_iterations):
            x_eq = self._equalize(y_btfnc, refined, no)
            s_hat = self._hard_slice_square_qam(x_eq)
            residual_phase = self._estimate_symbol_phase(x_eq, s_hat, refined)
            refined = self._apply_symbol_phase(refined, residual_phase)

        return btfnu_to_tensor7(refined), tf.cast(err_var, tf.float32), h_hat

    def call(self, inputs: Any, *args: Any, training: bool = False, **kwargs: Any) -> tuple[tf.Tensor, tf.Tensor]:
        del training, kwargs
        y, no = self._parse_inputs(inputs, *args)
        h_hat, err_var, _ = self.estimate_with_phase_tracking(y, no)
        return h_hat, err_var
