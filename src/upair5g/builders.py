from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf

from .compat import first_present_attr, instantiate_filtered, resolve_attr, set_if_present
from .config import get_cfg


def multiuser_enabled(cfg: dict[str, Any]) -> bool:
    return bool(get_cfg(cfg, "multiuser.enabled", False))


def max_num_users(cfg: dict[str, Any]) -> int:
    return int(get_cfg(cfg, "multiuser.max_num_users", 1 if not multiuser_enabled(cfg) else 4))


def _multiuser_dmrs_value(cfg: dict[str, Any], key: str, default: Any) -> Any:
    if multiuser_enabled(cfg):
        value = get_cfg(cfg, f"multiuser.dmrs.{key}", None)
        if value is not None:
            return value
    return get_cfg(cfg, f"pusch.dmrs.{key}", default)


def _default_dmrs_port_sets(cfg: dict[str, Any]) -> list[list[int]]:
    configured = get_cfg(cfg, "multiuser.dmrs.port_sets", None)
    if configured is not None:
        return [[int(p) for p in port_set] for port_set in configured]
    # One single-layer user per Sionna PUSCHConfig. Disjoint DMRS ports follow
    # Sionna's multi-transmitter PUSCH tutorial and avoid pilot contamination.
    return [[i] for i in range(max_num_users(cfg))]


def _set_required(obj: Any, attr_name: str, value: Any, context: str) -> None:
    if not set_if_present(obj, attr_name, value):
        raise AttributeError(f"Could not set required {context}.{attr_name}={value!r}.")


def _set_optional(obj: Any, attr_name: str, value: Any) -> bool:
    return set_if_present(obj, attr_name, value)


def _modulation_name(num_bits_per_symbol: int) -> str:
    return {
        2: "qpsk",
        4: "qam16",
        6: "qam64",
        8: "qam256",
    }.get(int(num_bits_per_symbol), f"qm{int(num_bits_per_symbol)}")


def _normalize_mcs_table(value: Any) -> str:
    normalized = str(value).replace("_", "").replace("-", "").lower()
    # Sionna versions differ in how they expose NR MCS table identifiers.
    # On Narval/Sionna 1.2.1, the public TB field may be numeric.
    numeric_tables = {
        "1": "qam64",
        "2": "qam256",
        "3": "qam64lowse",
    }
    return numeric_tables.get(normalized, normalized)


def _validate_runtime_mcs(tb: Any, pusch_cfg: dict[str, Any]) -> None:
    if tb is None:
        raise AttributeError("PUSCHConfig.tb is required for MCS/modulation validation.")

    expected_index = int(pusch_cfg["mcs_index"])
    actual_index = first_present_attr(tb, ["mcs_index", "_mcs_index"], None)
    if actual_index is None or int(actual_index) != expected_index:
        raise ValueError(f"Requested MCS index {expected_index}, but Sionna resolved {actual_index}.")

    actual_table = first_present_attr(tb, ["mcs_table", "_mcs_table"], None)
    if actual_table is not None:
        expected_table = _normalize_mcs_table(pusch_cfg["mcs_table"])
        if _normalize_mcs_table(actual_table) != expected_table:
            raise ValueError(f"Requested MCS table {pusch_cfg['mcs_table']!r}, but Sionna resolved {actual_table!r}.")

    expected_qm = get_cfg({"pusch": pusch_cfg}, "pusch.expected_num_bits_per_symbol", None)
    if expected_qm is not None:
        actual_qm = first_present_attr(tb, ["num_bits_per_symbol", "_num_bits_per_symbol"], None)
        if actual_qm is None:
            raise AttributeError("Could not verify Sionna TB num_bits_per_symbol.")
        actual_qm = int(actual_qm)
        expected_qm = int(expected_qm)
        if actual_qm != expected_qm:
            raise ValueError(
                f"Requested MCS {expected_index} to resolve to Qm={expected_qm} "
                f"({_modulation_name(expected_qm)}), but Sionna resolved Qm={actual_qm} "
                f"({_modulation_name(actual_qm)})."
            )

    expected_modulation = get_cfg({"pusch": pusch_cfg}, "pusch.expected_modulation", None)
    if expected_modulation is not None:
        actual_qm = int(first_present_attr(tb, ["num_bits_per_symbol", "_num_bits_per_symbol"], 0))
        actual_modulation = _modulation_name(actual_qm)
        if str(expected_modulation).lower() != actual_modulation:
            raise ValueError(
                f"Requested modulation {expected_modulation!r}, but Sionna resolved {actual_modulation!r} "
                f"from Qm={actual_qm}."
            )


def build_pusch_config(
    cfg: dict[str, Any],
    dmrs_port_set: list[int] | None = None,
) -> Any:
    PUSCHConfig = resolve_attr(["sionna.phy.nr", "sionna.nr"], "PUSCHConfig")
    pc = PUSCHConfig()

    pusch_cfg = cfg["pusch"]
    dmrs_cfg = pusch_cfg["dmrs"]

    carrier = getattr(pc, "carrier", None)
    tb = getattr(pc, "tb", None)
    dmrs = getattr(pc, "dmrs", None)

    if carrier is not None:
        set_if_present(carrier, "n_size_grid", int(pusch_cfg["n_size_grid"]))
        set_if_present(carrier, "subcarrier_spacing", int(pusch_cfg["subcarrier_spacing_khz"]))
        set_if_present(carrier, "cyclic_prefix", str(pusch_cfg["cyclic_prefix"]))

    set_if_present(pc, "n_size_bwp", int(pusch_cfg["n_size_bwp"]))
    set_if_present(pc, "mapping_type", str(pusch_cfg["mapping_type"]))
    set_if_present(pc, "symbol_allocation", list(pusch_cfg["symbol_allocation"]))
    set_if_present(pc, "num_layers", int(pusch_cfg["num_layers"]))
    set_if_present(pc, "num_antenna_ports", int(pusch_cfg["num_antenna_ports"]))
    set_if_present(pc, "precoding", str(pusch_cfg["precoding"]))
    set_if_present(pc, "transform_precoding", bool(pusch_cfg["transform_precoding"]))

    if tb is not None:
        # Sionna versions differ: some expose mcs_table as a public writable
        # property, while others keep the table internal. The runtime Qm check
        # below is the authoritative modulation guard.
        _set_optional(tb, "mcs_table", pusch_cfg["mcs_table"])
        _set_required(tb, "mcs_index", int(pusch_cfg["mcs_index"]), "PUSCHConfig.tb")
        _validate_runtime_mcs(tb, pusch_cfg)

    if dmrs is not None:
        set_if_present(dmrs, "config_type", int(_multiuser_dmrs_value(cfg, "config_type", dmrs_cfg["config_type"])))
        set_if_present(dmrs, "length", int(_multiuser_dmrs_value(cfg, "length", dmrs_cfg["length"])))
        set_if_present(dmrs, "additional_position", int(_multiuser_dmrs_value(cfg, "additional_position", dmrs_cfg["additional_position"])))
        set_if_present(dmrs, "type_a_position", int(_multiuser_dmrs_value(cfg, "type_a_position", dmrs_cfg["type_a_position"])))
        set_if_present(
            dmrs,
            "num_cdm_groups_without_data",
            int(_multiuser_dmrs_value(cfg, "num_cdm_groups_without_data", dmrs_cfg["num_cdm_groups_without_data"])),
        )
        if dmrs_port_set is not None:
            set_if_present(dmrs, "dmrs_port_set", list(dmrs_port_set))

    return pc


def build_pusch_transmitter(cfg: dict[str, Any], num_users: int | None = None) -> tuple[Any, Any]:
    PUSCHTransmitter = resolve_attr(["sionna.phy.nr", "sionna.nr"], "PUSCHTransmitter")
    if multiuser_enabled(cfg):
        requested_users = int(num_users if num_users is not None else get_cfg(cfg, "multiuser.fixed_num_users", max_num_users(cfg)))
        requested_users = max(1, min(requested_users, max_num_users(cfg)))
        port_sets = _default_dmrs_port_sets(cfg)
        if len(port_sets) < requested_users:
            raise ValueError(f"Need at least {requested_users} configured DMRS port sets, got {len(port_sets)}.")
        pusch_configs = [
            build_pusch_config(cfg, dmrs_port_set=port_sets[i])
            for i in range(requested_users)
        ]
    else:
        requested_users = 1
        pusch_configs = [build_pusch_config(cfg)]

    kwargs = {
        "pusch_configs": pusch_configs,
        "output_domain": "freq",
        "return_bits": True,
        "precision": get_cfg(cfg, "system.precision", "single"),
    }

    try:
        tx = instantiate_filtered(PUSCHTransmitter, **kwargs)
    except Exception:
        try:
            kwargs["pusch_configs"] = pusch_configs[0] if len(pusch_configs) == 1 else pusch_configs
            tx = instantiate_filtered(PUSCHTransmitter, **kwargs)
        except Exception:
            try:
                tx = PUSCHTransmitter(pusch_configs, output_domain="freq", return_bits=True, precision=get_cfg(cfg, "system.precision", "single"))
            except Exception:
                tx = PUSCHTransmitter(pusch_configs[0], output_domain="freq", return_bits=True, precision=get_cfg(cfg, "system.precision", "single"))

    try:
        setattr(tx, "_upair_num_users", requested_users)
        setattr(tx, "_upair_pusch_configs", pusch_configs)
        verified_tb = getattr(pusch_configs[0], "tb", None)
        if verified_tb is not None:
            setattr(tx, "_upair_num_bits_per_symbol", int(first_present_attr(verified_tb, ["num_bits_per_symbol", "_num_bits_per_symbol"], 4)))
            setattr(tx, "_upair_modulation", _modulation_name(int(first_present_attr(verified_tb, ["num_bits_per_symbol", "_num_bits_per_symbol"], 4))))
            coderate = first_present_attr(verified_tb, ["target_coderate", "_target_coderate", "coderate", "_coderate"], None)
            if coderate is not None:
                setattr(tx, "_upair_coderate", float(coderate))
    except Exception:
        pass

    return tx, pusch_configs[0]


def get_resource_grid(tx: Any) -> Any:
    rg = first_present_attr(tx, ["resource_grid", "_resource_grid"], None)
    if rg is None:
        raise AttributeError("Could not locate resource_grid in PUSCH transmitter.")
    return rg


def build_ls_estimator(
    tx: Any,
    cfg: dict[str, Any],
    interpolation_type: str = "lin",
    interpolator: Any | None = None,
) -> Any:
    PUSCHLSChannelEstimator = resolve_attr(["sionna.phy.nr", "sionna.nr"], "PUSCHLSChannelEstimator")
    rg = get_resource_grid(tx)
    kwargs = {
        "resource_grid": rg,
        "dmrs_length": first_present_attr(tx, ["_dmrs_length"], 1),
        "dmrs_additional_position": first_present_attr(tx, ["_dmrs_additional_position"], 0),
        "num_cdm_groups_without_data": first_present_attr(tx, ["_num_cdm_groups_without_data"], 2),
        "precision": get_cfg(cfg, "system.precision", "single"),
    }
    if interpolator is not None:
        kwargs["interpolator"] = interpolator
    else:
        kwargs["interpolation_type"] = str(interpolation_type)
    try:
        return instantiate_filtered(PUSCHLSChannelEstimator, **kwargs)
    except Exception:
        if interpolator is not None:
            return PUSCHLSChannelEstimator(
                rg,
                first_present_attr(tx, ["_dmrs_length"], 1),
                first_present_attr(tx, ["_dmrs_additional_position"], 0),
                first_present_attr(tx, ["_num_cdm_groups_without_data"], 2),
                interpolator=interpolator,
                precision=get_cfg(cfg, "system.precision", "single"),
            )
        return PUSCHLSChannelEstimator(
            rg,
            first_present_attr(tx, ["_dmrs_length"], 1),
            first_present_attr(tx, ["_dmrs_additional_position"], 0),
            first_present_attr(tx, ["_num_cdm_groups_without_data"], 2),
            interpolation_type=str(interpolation_type),
            precision=get_cfg(cfg, "system.precision", "single"),
        )


def _build_single_antenna(num_ant: int, carrier_frequency: float) -> Any:
    try:
        Antenna = resolve_attr(["sionna.phy.channel.tr38901", "sionna.channel.tr38901"], "Antenna")
        return instantiate_filtered(
            Antenna,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=carrier_frequency,
        )
    except Exception:
        AntennaArray = resolve_attr(["sionna.phy.channel.tr38901", "sionna.channel.tr38901"], "AntennaArray")
        return instantiate_filtered(
            AntennaArray,
            num_rows=1,
            num_cols=num_ant,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=carrier_frequency,
        )


def _build_bs_array(num_ant: int, carrier_frequency: float) -> Any:
    AntennaArray = resolve_attr(["sionna.phy.channel.tr38901", "sionna.channel.tr38901"], "AntennaArray")
    return instantiate_filtered(
        AntennaArray,
        num_rows=1,
        num_cols=num_ant,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=carrier_frequency,
    )


def _build_cdl_channel_model(cfg: dict[str, Any]) -> Any:
    CDL = resolve_attr(["sionna.phy.channel.tr38901", "sionna.channel.tr38901"], "CDL")

    channel_cfg = cfg["channel"]
    pusch_cfg = cfg["pusch"]
    carrier_frequency = float(pusch_cfg["carrier_frequency_hz"])

    ut_array = _build_single_antenna(int(channel_cfg["num_tx_ant"]), carrier_frequency)
    bs_array = _build_bs_array(int(channel_cfg["num_rx_ant"]), carrier_frequency)

    cdl_kwargs = {
        "model": str(channel_cfg["model"]),
        "delay_spread": float(channel_cfg["delay_spread_s"]),
        "carrier_frequency": carrier_frequency,
        "ut_array": ut_array,
        "bs_array": bs_array,
        "direction": "uplink",
        "min_speed": float(channel_cfg["min_speed_mps"]),
        "max_speed": float(channel_cfg["max_speed_mps"]),
        "dtype": tf.complex64,
    }
    try:
        channel_model = instantiate_filtered(CDL, **cdl_kwargs)
    except Exception:
        channel_model = CDL(
            str(channel_cfg["model"]),
            float(channel_cfg["delay_spread_s"]),
            carrier_frequency,
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            min_speed=float(channel_cfg["min_speed_mps"]),
            max_speed=float(channel_cfg["max_speed_mps"]),
        )
    return channel_model


def _build_ofdm_channel(cfg: dict[str, Any], tx: Any, channel_model: Any, add_awgn: bool) -> Any:
    OFDMChannel = resolve_attr(["sionna.phy.channel", "sionna.channel"], "OFDMChannel")
    channel_cfg = cfg["channel"]
    ofdm_kwargs = {
        "channel_model": channel_model,
        "resource_grid": get_resource_grid(tx),
        "add_awgn": bool(add_awgn),
        "normalize_channel": bool(channel_cfg["normalize_channel"]),
        "return_channel": True,
    }
    try:
        return instantiate_filtered(OFDMChannel, **ofdm_kwargs)
    except Exception:
        return OFDMChannel(
            channel_model,
            get_resource_grid(tx),
            add_awgn=bool(add_awgn),
            normalize_channel=bool(channel_cfg["normalize_channel"]),
            return_channel=True,
        )


def _infer_channel_pair(channel_output: Any) -> tuple[tf.Tensor, tf.Tensor]:
    if not isinstance(channel_output, (tuple, list)) or len(channel_output) < 2:
        raise ValueError("OFDM channel must return (y, h).")
    return tf.convert_to_tensor(channel_output[0]), tf.convert_to_tensor(channel_output[1])


def _call_clean_ofdm_channel(channel: Any, x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    zero_no = tf.constant(0.0, tf.float32)
    attempts = [
        lambda: channel(x),
        lambda: channel([x]),
        lambda: channel((x,)),
        lambda: channel(x, zero_no),
        lambda: channel([x, zero_no]),
        lambda: channel((x, zero_no)),
    ]
    last_err = None
    for attempt in attempts:
        try:
            return _infer_channel_pair(attempt())
        except (tf.errors.ResourceExhaustedError, MemoryError):
            raise
        except Exception as err:  # pragma: no cover - Sionna version compatibility
            last_err = err
    raise RuntimeError("All clean OFDM channel calling conventions failed.") from last_err


def _add_awgn_once(y: tf.Tensor, no: tf.Tensor) -> tf.Tensor:
    no = tf.cast(tf.convert_to_tensor(no), tf.float32)
    if no.shape.rank == 0:
        scale = tf.sqrt(no / 2.0)
    else:
        no = tf.reshape(no, [-1])
        scale = tf.sqrt(no / 2.0)
        scale = tf.reshape(scale, [-1, 1, 1, 1, 1])
    noise = tf.complex(
        tf.random.normal(tf.shape(y), dtype=tf.float32),
        tf.random.normal(tf.shape(y), dtype=tf.float32),
    )
    return y + tf.cast(scale, y.dtype) * tf.cast(noise, y.dtype)


class IndependentMultiUserOFDMChannel:
    """Compose independent single-user CDL links into one multi-user channel.

    Sionna's CDL model represents one UT-BS link. A multi-user PUSCH resource
    grid therefore needs one independent CDL/OFDM channel per scheduled UE.
    The clean received grids are summed, AWGN is added once, and the true
    channel tensors are concatenated along the transmitter axis.
    """

    def __init__(self, user_channels: list[Any]) -> None:
        if not user_channels:
            raise ValueError("At least one user channel is required.")
        self.user_channels = list(user_channels)
        self.num_users = len(self.user_channels)
        self.return_channel = True

    def __call__(self, x: tf.Tensor, no: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        x = tf.convert_to_tensor(x)
        if x.shape.rank != 5:
            raise ValueError(f"Expected transmit grid rank 5 [B,U,S,T,F], got rank {x.shape.rank}.")
        static_num_tx = x.shape[1]
        if static_num_tx is not None and int(static_num_tx) != self.num_users:
            raise ValueError(f"Expected {self.num_users} transmit users, got {static_num_tx}.")

        y_sum: tf.Tensor | None = None
        h_parts: list[tf.Tensor] = []
        for user_idx, channel in enumerate(self.user_channels):
            x_user = x[:, user_idx : user_idx + 1, :, :, :]
            y_user, h_user = _call_clean_ofdm_channel(channel, x_user)
            y_sum = y_user if y_sum is None else y_sum + y_user
            h_parts.append(h_user)

        if y_sum is None:
            raise RuntimeError("No user channels were evaluated.")
        h = tf.concat(h_parts, axis=3)
        return _add_awgn_once(y_sum, no), h


def _build_independent_multiuser_channel(cfg: dict[str, Any], num_users: int) -> IndependentMultiUserOFDMChannel:
    user_channels: list[Any] = []
    for _ in range(int(num_users)):
        single_tx, _ = build_pusch_transmitter(cfg, num_users=1)
        channel_model = _build_cdl_channel_model(cfg)
        user_channels.append(_build_ofdm_channel(cfg, single_tx, channel_model, add_awgn=False))
    return IndependentMultiUserOFDMChannel(user_channels)


def build_channel(cfg: dict[str, Any], tx: Any) -> Any:
    num_tx = int(first_present_attr(tx, ["_upair_num_users", "num_tx", "_num_tx"], 1))
    if multiuser_enabled(cfg) and num_tx > 1:
        return _build_independent_multiuser_channel(cfg, num_tx)

    channel_model = _build_cdl_channel_model(cfg)
    return _build_ofdm_channel(cfg, tx, channel_model, add_awgn=True)


def _build_multiuser_detector(tx: Any, cfg: dict[str, Any]) -> Any | None:
    if not multiuser_enabled(cfg):
        return None
    num_tx = int(first_present_attr(tx, ["_upair_num_users", "num_tx", "_num_tx"], max_num_users(cfg)))
    num_layers = int(get_cfg(cfg, "pusch.num_layers", 1))
    try:
        StreamManagement = resolve_attr(["sionna.phy.mimo", "sionna.mimo"], "StreamManagement")
        LinearDetector = resolve_attr(["sionna.phy.ofdm", "sionna.ofdm"], "LinearDetector")
    except Exception:
        return None

    stream_management = StreamManagement(np.ones([1, num_tx], dtype=bool), num_layers)
    pusch_configs = first_present_attr(tx, ["_upair_pusch_configs"], [])
    pusch_config = pusch_configs[0] if pusch_configs else None
    tb = getattr(pusch_config, "tb", None)
    num_bits_per_symbol = first_present_attr(tx, ["_upair_num_bits_per_symbol", "_num_bits_per_symbol", "num_bits_per_symbol"], None)
    if num_bits_per_symbol is None and tb is not None:
        num_bits_per_symbol = first_present_attr(tb, ["num_bits_per_symbol", "_num_bits_per_symbol"], 6)
    num_bits_per_symbol = int(num_bits_per_symbol if num_bits_per_symbol is not None else 6)

    kwargs = {
        "equalizer": "lmmse",
        "output": "bit",
        "demapping_method": "maxlog",
        "resource_grid": get_resource_grid(tx),
        "stream_management": stream_management,
        "constellation_type": "qam",
        "num_bits_per_symbol": num_bits_per_symbol,
    }
    try:
        return instantiate_filtered(LinearDetector, **kwargs)
    except Exception:
        try:
            return LinearDetector(
                "lmmse",
                "bit",
                "maxlog",
                get_resource_grid(tx),
                stream_management,
                "qam",
                num_bits_per_symbol,
            )
        except Exception:
            return None


def build_receiver(tx: Any, cfg: dict[str, Any], channel_estimator: Any | None = None, perfect_csi: bool = False) -> Any:
    PUSCHReceiver = resolve_attr(["sionna.phy.nr", "sionna.nr"], "PUSCHReceiver")
    kwargs = {
        "pusch_transmitter": tx,
        "return_tb_crc_status": True,
        "input_domain": "freq",
        "precision": get_cfg(cfg, "system.precision", "single"),
    }
    detector = _build_multiuser_detector(tx, cfg)
    if detector is not None:
        kwargs["mimo_detector"] = detector
    if perfect_csi:
        kwargs["channel_estimator"] = "perfect"
    elif channel_estimator is not None:
        kwargs["channel_estimator"] = channel_estimator
    try:
        return instantiate_filtered(PUSCHReceiver, **kwargs)
    except Exception:
        fallback_kwargs = {k: v for k, v in kwargs.items() if k != "pusch_transmitter"}
        try:
            return PUSCHReceiver(tx, **fallback_kwargs)
        except Exception:
            fallback_kwargs.pop("mimo_detector", None)
            return PUSCHReceiver(tx, **fallback_kwargs)


def extract_pilot_mask_per_stream(resource_grid: Any) -> tf.Tensor:
    pilot_pattern = first_present_attr(resource_grid, ["pilot_pattern", "_pilot_pattern"], None)
    if pilot_pattern is None:
        raise AttributeError("Could not locate pilot pattern in resource_grid.")
    mask = first_present_attr(pilot_pattern, ["mask", "_mask"], None)
    if mask is None:
        raise AttributeError("Could not locate pilot mask in pilot pattern.")

    mask = tf.cast(tf.convert_to_tensor(mask), tf.float32)

    # Sionna pilot masks are typically [num_tx, num_streams, T, F]. Older
    # versions expose singleton leading axes for single-user cases. Flatten all
    # leading dimensions into a stream/user axis and leave the last two grid
    # dimensions intact.
    rank = mask.shape.rank
    if rank is None or rank < 2:
        raise ValueError(f"Unexpected pilot-mask rank: {rank}")
    if rank == 2:
        mask = mask[tf.newaxis, ...]
    else:
        shape = tf.shape(mask)
        leading = tf.reduce_prod(shape[:-2])
        mask = tf.reshape(mask, [leading, shape[-2], shape[-1]])

    # Some versions store [F, T] instead of [T, F].
    static_shape = mask.shape.as_list()
    if static_shape is not None and len(static_shape) == 3:
        _, t, f = static_shape
        if t is not None and f is not None and t > f:
            mask = tf.transpose(mask, [0, 2, 1])

    return tf.transpose(mask, [1, 2, 0])  # [T, F, num_streams]


def extract_pilot_mask(resource_grid: Any) -> tf.Tensor:
    mask = extract_pilot_mask_per_stream(resource_grid)
    return tf.reduce_max(mask, axis=-1, keepdims=True)
