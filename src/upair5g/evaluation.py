from __future__ import annotations

import signal
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

from .baselines import (
    PERFECT_RECEIVER,
    PROPOSED_RECEIVER,
    build_classical_baseline_suite,
    classical_receivers_from_cfg,
    enabled_receivers_from_cfg,
    wants_receiver,
)
from .builders import build_channel, build_ls_estimator, build_pusch_transmitter, build_receiver, get_resource_grid, max_num_users, multiuser_enabled
from .compat import safe_call_variants
from .config import ensure_output_tree, get_cfg
from .estimator import UPAIRChannelEstimator
from .impairments import apply_symbol_phase_impairment
from .utils import (
    call_channel,
    call_receiver,
    call_transmitter,
    compute_ber,
    compute_bler_from_crc,
    complex_sq_abs,
    ebno_db_to_no,
    save_json,
    save_yaml,
    set_global_seed,
)


def _call_channel_estimator(estimator: Any, y: tf.Tensor, no: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    try:
        out = estimator(y, no)
    except (tf.errors.ResourceExhaustedError, MemoryError):
        raise
    except Exception:
        out = safe_call_variants(estimator, y, no)
    if not isinstance(out, (tuple, list)) or len(out) < 2:
        raise ValueError("Channel estimator must return (h_hat, err_var).")
    return tf.convert_to_tensor(out[0]), tf.convert_to_tensor(out[1])


def _make_eval_batch(
    tx: Any,
    channel: Any,
    cfg: dict[str, Any],
    batch_size: int,
    ebno_db: float,
) -> dict[str, tf.Tensor]:
    x, bits = call_transmitter(tx, batch_size)
    no = ebno_db_to_no(tf.constant(float(ebno_db), tf.float32), tx=tx, resource_grid=get_resource_grid(tx))
    y, h = call_channel(channel, x, no)
    y, h = apply_symbol_phase_impairment(y, h, cfg, training=False)
    return {"b": bits, "y": y, "h": h, "no": no, "ebno_db": tf.constant(float(ebno_db), tf.float32)}


def _first_dim(value: tf.Tensor) -> int | None:
    tensor = tf.convert_to_tensor(value)
    if tensor.shape.rank == 0:
        return None
    if tensor.shape[0] is not None:
        return int(tensor.shape[0])
    try:
        return int(tf.shape(tensor)[0].numpy())
    except Exception:
        return None


def _slice_batch_axis(value: tf.Tensor | None, start: int, end: int, batch_size: int) -> tf.Tensor | None:
    if value is None:
        return None
    tensor = tf.convert_to_tensor(value)
    if _first_dim(tensor) == int(batch_size):
        return tensor[start:end]
    return tensor


def _iter_eval_microbatches(batch: dict[str, tf.Tensor], microbatch_size: int) -> list[dict[str, tf.Tensor]]:
    batch_size = int(tf.shape(batch["y"])[0].numpy())
    microbatch_size = max(1, min(int(microbatch_size), batch_size))
    result: list[dict[str, tf.Tensor]] = []
    for start in range(0, batch_size, microbatch_size):
        end = min(start + microbatch_size, batch_size)
        result.append(
            {
                key: _slice_batch_axis(value, start, end, batch_size)  # type: ignore[arg-type]
                for key, value in batch.items()
            }
        )
    return result


def _nmse_components(h_true: tf.Tensor, h_hat: tf.Tensor) -> tuple[float, float]:
    numerator = tf.reduce_sum(complex_sq_abs(tf.convert_to_tensor(h_true) - tf.convert_to_tensor(h_hat)))
    denominator = tf.reduce_sum(complex_sq_abs(h_true))
    return float(numerator.numpy()), float(denominator.numpy())


def _safe_concat(parts: list[tf.Tensor]) -> tf.Tensor | None:
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return tf.concat(parts, axis=0)


def _metric_min(df: pd.DataFrame, receiver: str, metric: str) -> float | None:
    sub = df[df["receiver"] == receiver][metric].dropna()
    if sub.empty:
        return None
    return float(sub.min())


def _best_classical_row(df: pd.DataFrame, metric: str, reliable_only: bool = False) -> dict[str, float | str] | None:
    sub = df[["receiver", metric]].dropna().copy()
    if reliable_only and metric in {"ber", "bler"}:
        reliability_col = f"reliable_{metric}"
        if reliability_col in df.columns:
            sub = df.loc[df[reliability_col].fillna(False), ["receiver", metric]].dropna().copy()
    if sub.empty:
        return None
    idx = sub[metric].idxmin()
    row = sub.loc[idx]
    return {"receiver": str(row["receiver"]), "value": float(row[metric])}


def _build_summary(
    df: pd.DataFrame,
    checkpoint_path: str | None,
    enabled_receivers: list[str],
    artifacts: dict[str, str],
    eval_cfg: dict[str, Any],
) -> dict[str, Any]:
    classical_receivers = classical_receivers_from_cfg({"baselines": {"enabled_receivers": enabled_receivers}})
    classical_df = df[df["receiver"].isin(classical_receivers)].copy()

    summary: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "enabled_receivers": enabled_receivers,
        "classical_receivers": classical_receivers,
        "num_users": int(df["num_users"].dropna().iloc[0]) if "num_users" in df.columns and not df["num_users"].dropna().empty else None,
        "num_curve_rows": int(len(df)),
        "evaluation_controls": {
            "min_num_batches_per_point": int(eval_cfg["min_num_batches_per_point"]),
            "max_num_batches_per_point": int(eval_cfg["max_num_batches_per_point"]),
            "target_block_errors_per_receiver": int(eval_cfg["target_block_errors_per_receiver"]),
            "reliable_min_block_errors": int(eval_cfg["reliable_min_block_errors"]),
            "reliable_min_bit_errors": int(eval_cfg["reliable_min_bit_errors"]),
            "stopping_receivers": list(eval_cfg["stopping_receivers"]),
        },
    }
    summary.update(artifacts)

    if PROPOSED_RECEIVER in enabled_receivers:
        summary["best_ber_upair5g"] = _metric_min(df, PROPOSED_RECEIVER, "ber")
        summary["best_bler_upair5g"] = _metric_min(df, PROPOSED_RECEIVER, "bler")
        summary["best_nmse_upair5g"] = _metric_min(df, PROPOSED_RECEIVER, "nmse")

    if classical_receivers:
        summary["best_ber_classical"] = _best_classical_row(classical_df, "ber", reliable_only=False)
        summary["best_bler_classical"] = _best_classical_row(classical_df, "bler", reliable_only=False)
        summary["best_nmse_classical"] = _best_classical_row(classical_df, "nmse", reliable_only=False)
        summary["best_ber_classical_reliable_only"] = _best_classical_row(classical_df, "ber", reliable_only=True)
        summary["best_bler_classical_reliable_only"] = _best_classical_row(classical_df, "bler", reliable_only=True)

    per_ebno_best_classical: list[dict[str, Any]] = []
    if classical_receivers and PROPOSED_RECEIVER in enabled_receivers:
        for ebno_db in sorted(df["ebno_db"].unique().tolist()):
            row_summary: dict[str, Any] = {"ebno_db": float(ebno_db)}
            classical_slice = classical_df[classical_df["ebno_db"] == ebno_db]
            proposed_slice = df[(df["receiver"] == PROPOSED_RECEIVER) & (df["ebno_db"] == ebno_db)]
            if proposed_slice.empty:
                continue
            proposed_row = proposed_slice.iloc[0]
            for metric in ["ber", "bler", "nmse"]:
                reliable_only = metric in {"ber", "bler"}
                best_classical = _best_classical_row(classical_slice, metric, reliable_only=reliable_only)
                if best_classical is None:
                    continue
                row_summary[f"best_{metric}_classical_receiver"] = best_classical["receiver"]
                row_summary[f"best_{metric}_classical"] = best_classical["value"]
                if pd.notna(proposed_row[metric]):
                    upair_value = float(proposed_row[metric])
                    row_summary[f"{metric}_upair5g"] = upair_value
                    row_summary[f"{metric}_gap_upair_minus_best_classical"] = upair_value - float(best_classical["value"])
                    if metric in {"ber", "bler"}:
                        row_summary[f"upair_{metric}_reliable"] = bool(proposed_row.get(f"reliable_{metric}", False))
            per_ebno_best_classical.append(row_summary)
    summary["per_ebno_best_classical"] = per_ebno_best_classical
    return summary


def _bool_cfg_list(cfg: dict[str, Any], key: str, default: list[str]) -> list[str]:
    value = get_cfg(cfg, key, default)
    if isinstance(value, str):
        return [value]
    return [str(x) for x in value]


def _init_counter() -> dict[str, float | int]:
    return {
        "bit_errors": 0,
        "num_bits": 0,
        "block_errors": 0,
        "num_blocks": 0,
        "nmse_sum": 0.0,
        "num_nmse_batches": 0,
        "num_batches_run": 0,
    }


def _update_error_counters(counter: dict[str, float | int], bits: tf.Tensor | None, b_hat: tf.Tensor | None, crc: tf.Tensor | None) -> None:
    if bits is not None and b_hat is not None:
        num_bits = int(tf.size(bits).numpy())
        ber_value = float(compute_ber(bits, b_hat).numpy())
        bit_errors = int(np.rint(ber_value * num_bits))
        counter["num_bits"] = int(counter["num_bits"]) + num_bits
        counter["bit_errors"] = int(counter["bit_errors"]) + bit_errors

    if crc is not None:
        num_blocks = int(tf.size(crc).numpy())
        bler_value = float(compute_bler_from_crc(crc).numpy())
        block_errors = int(np.rint(bler_value * num_blocks))
        counter["num_blocks"] = int(counter["num_blocks"]) + num_blocks
        counter["block_errors"] = int(counter["block_errors"]) + block_errors


def _should_stop(
    agg: dict[str, dict[str, float | int]],
    stopping_receivers: list[str],
    batches_run: int,
    min_num_batches: int,
    max_num_batches: int,
    target_block_errors: int,
) -> bool:
    if batches_run < min_num_batches:
        return False
    if batches_run >= max_num_batches:
        return True
    if target_block_errors <= 0:
        return False
    for receiver_name in stopping_receivers:
        block_errors = int(agg[receiver_name]["block_errors"])
        if block_errors < target_block_errors:
            return False
    return True


def evaluate_model(cfg: dict[str, Any], checkpoint_path: str | None = None, num_users: int | None = None) -> dict[str, Any]:
    training_seed = int(get_cfg(cfg, "system.training_seed", cfg["system"]["seed"]))
    evaluation_seed = int(get_cfg(cfg, "system.evaluation_seed", cfg["system"]["seed"]))
    set_global_seed(evaluation_seed)
    if bool(get_cfg(cfg, "system.graph_mode", True)):
        tf.config.run_functions_eagerly(False)
    paths = ensure_output_tree(cfg)
    curves_path = paths["metrics"] / "curves.csv"
    eval_state_path = paths["metrics"] / "evaluation_state.json"
    save_yaml(cfg, paths["artifacts"] / "resolved_config.yaml")

    eval_num_users = int(num_users if num_users is not None else get_cfg(cfg, "multiuser.fixed_num_users", max_num_users(cfg) if multiuser_enabled(cfg) else 1))
    tx, _ = build_pusch_transmitter(cfg, num_users=eval_num_users)
    channel = build_channel(cfg, tx)
    eval_batch_size = int(cfg["system"]["batch_size_eval"])
    receiver_microbatch_size = int(get_cfg(cfg, "evaluation.receiver_microbatch_size", eval_batch_size))
    if receiver_microbatch_size <= 0:
        receiver_microbatch_size = eval_batch_size
    receiver_microbatch_size = max(1, min(receiver_microbatch_size, eval_batch_size))
    if receiver_microbatch_size < eval_batch_size:
        print(
            "[EVAL] receiver microbatching enabled: "
            f"batch_size_eval={eval_batch_size} receiver_microbatch_size={receiver_microbatch_size}"
        )

    ls_estimator = build_ls_estimator(tx, cfg, interpolation_type="lin")
    estimator = UPAIRChannelEstimator(ls_estimator=ls_estimator, resource_grid=get_resource_grid(tx), cfg=cfg)

    warmup_batch = _make_eval_batch(
        tx=tx,
        channel=channel,
        cfg=cfg,
        batch_size=receiver_microbatch_size,
        ebno_db=float(get_cfg(cfg, "system.ebno_db_eval", [10])[0]),
    )
    estimator.estimate_with_ls(warmup_batch["y"], warmup_batch["no"], training=False)

    if checkpoint_path is not None:
        estimator.load_weights(str(checkpoint_path))

    enabled_receivers = enabled_receivers_from_cfg(cfg)
    classical_receivers, classical_estimators, baseline_artifacts = build_classical_baseline_suite(
        cfg=cfg,
        tx=tx,
        channel=channel,
        paths=paths,
    )

    proposed_rx = None
    if wants_receiver(cfg, PROPOSED_RECEIVER):
        proposed_rx = build_receiver(tx, cfg, channel_estimator=estimator, perfect_csi=False)

    perfect_rx = None
    if wants_receiver(cfg, PERFECT_RECEIVER):
        perfect_rx = build_receiver(tx, cfg, channel_estimator=None, perfect_csi=True)

    ebno_grid = [float(x) for x in get_cfg(cfg, "system.ebno_db_eval", [0, 4, 8, 12])]

    max_num_batches = int(get_cfg(cfg, "evaluation.max_num_batches_per_point", get_cfg(cfg, "evaluation.num_batches_per_point", 256)))
    min_num_batches = int(get_cfg(cfg, "evaluation.min_num_batches_per_point", min(64, max_num_batches)))
    target_block_errors = int(get_cfg(cfg, "evaluation.target_block_errors_per_receiver", 0))
    reliable_min_block_errors = int(get_cfg(cfg, "evaluation.reliable_min_block_errors", 1))
    reliable_min_bit_errors = int(get_cfg(cfg, "evaluation.reliable_min_bit_errors", 1))
    stopping_receivers = _bool_cfg_list(cfg, "evaluation.stopping_receivers", enabled_receivers)
    progress_every_batches = int(get_cfg(cfg, "evaluation.progress_every_batches", 0))

    rows: list[dict[str, Any]] = []
    completed_ebno: set[float] = set()
    resume_eval = bool(get_cfg(cfg, "evaluation.resume", True)) and not bool(get_cfg(cfg, "evaluation.force", False))
    if resume_eval and curves_path.exists():
        try:
            existing = pd.read_csv(curves_path)
            if "num_users" in existing.columns:
                existing = existing[existing["num_users"] == eval_num_users].copy()
            required_receivers = set(enabled_receivers)
            for ebno_value, group in existing.groupby("ebno_db"):
                if required_receivers.issubset(set(group["receiver"].astype(str))):
                    completed_ebno.add(float(ebno_value))
            rows = existing.to_dict("records")
            if completed_ebno:
                done = ", ".join(f"{x:g}" for x in sorted(completed_ebno))
                print(f"[EVAL] resuming num_users={eval_num_users}; completed Eb/N0 points: {done}")
        except Exception as exc:
                print(f"[EVAL] ignoring unreadable partial curves {curves_path}: {exc!r}")

    example_saved = False
    stop_requested = False

    def _request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"[EVAL] received signal {signum}; will save completed Eb/N0 points and stop after current batch.")

    previous_handlers: dict[int, Any] = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, _request_stop)
        except Exception:
            pass

    eval_cfg = {
        "min_num_batches_per_point": min_num_batches,
        "max_num_batches_per_point": max_num_batches,
        "target_block_errors_per_receiver": target_block_errors,
        "reliable_min_block_errors": reliable_min_block_errors,
        "reliable_min_bit_errors": reliable_min_bit_errors,
        "stopping_receivers": stopping_receivers,
        "progress_every_batches": progress_every_batches,
    }

    def _save_eval_state(
        *,
        complete: bool,
        reason: str,
        completed: set[float],
        current_ebno_db: float | None = None,
        partial_batches_run: int | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "num_users": int(eval_num_users),
            "completed_ebno_db": sorted(float(x) for x in completed),
            "curves_csv": str(curves_path),
            "evaluation_complete": bool(complete),
            "save_reason": reason,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
            "evaluation_parameters": {
                "batch_size_eval": int(eval_batch_size),
                "receiver_microbatch_size": int(receiver_microbatch_size),
                "seed": int(evaluation_seed),
                "training_seed": int(training_seed),
                "evaluation_seed": int(evaluation_seed),
                **eval_cfg,
            },
        }
        if current_ebno_db is not None:
            payload["current_ebno_db"] = float(current_ebno_db)
        if partial_batches_run is not None:
            payload["partial_batches_run"] = int(partial_batches_run)
        if complete:
            payload["summary_path"] = str(paths["metrics"] / "evaluation_summary.json")
        save_json(payload, eval_state_path)

    try:
        for ebno_db in ebno_grid:
            if float(ebno_db) in completed_ebno:
                print(f"[EVAL] reusing completed Eb/N0={ebno_db:g} dB for num_users={eval_num_users}")
                continue

            agg: dict[str, dict[str, float | int]] = {
                receiver_name: _init_counter()
                for receiver_name in enabled_receivers
            }
            point_interrupted = False
            batches_completed = 0

            for batch_idx in range(max_num_batches):
                batch = _make_eval_batch(
                    tx=tx,
                    channel=channel,
                    cfg=cfg,
                    batch_size=eval_batch_size,
                    ebno_db=ebno_db,
                )
                micro_batches = _iter_eval_microbatches(batch, receiver_microbatch_size)
                collect_example = not example_saved and bool(get_cfg(cfg, "evaluation.save_example_batch", True))
                example_classical_h_hats: dict[str, tf.Tensor] = {}
                example_h_hat_prop: tf.Tensor | None = None
                example_h_ls: tf.Tensor | None = None

                for receiver_name, estimator_block in classical_estimators.items():
                    h_parts: list[tf.Tensor] = []
                    nmse_num = 0.0
                    nmse_den = 0.0
                    for micro_batch in micro_batches:
                        h_hat_base, _ = _call_channel_estimator(estimator_block, micro_batch["y"], micro_batch["no"])
                        b_hat, crc = call_receiver(classical_receivers[receiver_name], micro_batch["y"], micro_batch["no"])
                        _update_error_counters(agg[receiver_name], micro_batch["b"], b_hat, crc)
                        num, den = _nmse_components(micro_batch["h"], h_hat_base)
                        nmse_num += num
                        nmse_den += den
                        if collect_example:
                            h_parts.append(h_hat_base)
                    agg[receiver_name]["nmse_sum"] = float(agg[receiver_name]["nmse_sum"]) + float(nmse_num / max(nmse_den, 1e-9))
                    agg[receiver_name]["num_nmse_batches"] = int(agg[receiver_name]["num_nmse_batches"]) + 1
                    agg[receiver_name]["num_batches_run"] = int(agg[receiver_name]["num_batches_run"]) + 1
                    if collect_example:
                        concatenated = _safe_concat(h_parts)
                        if concatenated is not None:
                            example_classical_h_hats[receiver_name] = concatenated

                if proposed_rx is not None:
                    h_prop_parts: list[tf.Tensor] = []
                    h_ls_parts: list[tf.Tensor] = []
                    nmse_num = 0.0
                    nmse_den = 0.0
                    for micro_batch in micro_batches:
                        h_hat_prop, _, h_ls, _ = estimator.estimate_with_ls(micro_batch["y"], micro_batch["no"], training=False)
                        b_hat_prop, crc_prop = call_receiver(proposed_rx, micro_batch["y"], micro_batch["no"])
                        _update_error_counters(agg[PROPOSED_RECEIVER], micro_batch["b"], b_hat_prop, crc_prop)
                        num, den = _nmse_components(micro_batch["h"], h_hat_prop)
                        nmse_num += num
                        nmse_den += den
                        if collect_example:
                            h_prop_parts.append(h_hat_prop)
                            h_ls_parts.append(h_ls)
                    agg[PROPOSED_RECEIVER]["nmse_sum"] = float(agg[PROPOSED_RECEIVER]["nmse_sum"]) + float(nmse_num / max(nmse_den, 1e-9))
                    agg[PROPOSED_RECEIVER]["num_nmse_batches"] = int(agg[PROPOSED_RECEIVER]["num_nmse_batches"]) + 1
                    agg[PROPOSED_RECEIVER]["num_batches_run"] = int(agg[PROPOSED_RECEIVER]["num_batches_run"]) + 1
                    if collect_example:
                        example_h_hat_prop = _safe_concat(h_prop_parts)
                        example_h_ls = _safe_concat(h_ls_parts)

                if perfect_rx is not None:
                    for micro_batch in micro_batches:
                        b_hat_perf, crc_perf = call_receiver(perfect_rx, micro_batch["y"], micro_batch["no"], h=micro_batch["h"])
                        _update_error_counters(agg[PERFECT_RECEIVER], micro_batch["b"], b_hat_perf, crc_perf)
                    agg[PERFECT_RECEIVER]["num_nmse_batches"] = int(agg[PERFECT_RECEIVER]["num_nmse_batches"]) + 1
                    agg[PERFECT_RECEIVER]["num_batches_run"] = int(agg[PERFECT_RECEIVER]["num_batches_run"]) + 1

                if collect_example:
                    example_payload: dict[str, Any] = {
                        "h_true": np.asarray(batch["h"].numpy()),
                        "y": np.asarray(batch["y"].numpy()),
                        "ebno_db": np.asarray([ebno_db]),
                    }
                    if "baseline_ls_lmmse" in example_classical_h_hats:
                        example_payload["h_ls_linear"] = np.asarray(example_classical_h_hats["baseline_ls_lmmse"].numpy())
                    elif example_h_ls is not None:
                        example_payload["h_ls_linear"] = np.asarray(example_h_ls.numpy())
                    if "baseline_ls_timeavg_lmmse" in example_classical_h_hats:
                        example_payload["h_ls_timeavg"] = np.asarray(example_classical_h_hats["baseline_ls_timeavg_lmmse"].numpy())
                    if "baseline_ls_2dlmmse_lmmse" in example_classical_h_hats:
                        example_payload["h_ls_2dlmmse"] = np.asarray(example_classical_h_hats["baseline_ls_2dlmmse_lmmse"].numpy())
                    if "baseline_ddcpe_ls_lmmse" in example_classical_h_hats:
                        example_payload["h_ddcpe_ls"] = np.asarray(example_classical_h_hats["baseline_ddcpe_ls_lmmse"].numpy())
                    if example_h_hat_prop is not None:
                        example_payload["h_prop"] = np.asarray(example_h_hat_prop.numpy())
                    np.savez_compressed(paths["artifacts"] / "channel_example.npz", **example_payload)
                    example_saved = True

                batches_completed = batch_idx + 1
                if progress_every_batches > 0 and batches_completed % progress_every_batches == 0:
                    _save_eval_state(
                        complete=False,
                        reason="progress",
                        completed=completed_ebno,
                        current_ebno_db=float(ebno_db),
                        partial_batches_run=batches_completed,
                    )
                    print(
                        f"[EVAL] progress num_users={eval_num_users} "
                        f"Eb/N0={ebno_db:g} dB batches={batches_completed}/{max_num_batches}"
                    )

                if stop_requested:
                    point_interrupted = True
                    break

                if _should_stop(
                    agg=agg,
                    stopping_receivers=[r for r in stopping_receivers if r in agg],
                    batches_run=batch_idx + 1,
                    min_num_batches=min_num_batches,
                    max_num_batches=max_num_batches,
                    target_block_errors=target_block_errors,
                ):
                    break

            if point_interrupted:
                pd.DataFrame(rows).to_csv(curves_path, index=False)
                _save_eval_state(
                    complete=False,
                    reason="signal",
                    completed=completed_ebno,
                    current_ebno_db=float(ebno_db),
                    partial_batches_run=batches_completed,
                )
                print("[EVAL] stopped before completing current Eb/N0; resubmit to resume from completed points.")
                return {
                    "output_dir": str(paths["root"]),
                    "curves_path": str(curves_path),
                    "summary_path": str(paths["metrics"] / "evaluation_summary.json"),
                    "evaluation_complete": False,
                    "completed_ebno_db": sorted(float(x) for x in completed_ebno),
                    "training_seed": int(training_seed),
                    "evaluation_seed": int(evaluation_seed),
                }

            for receiver_name in enabled_receivers:
                counter = agg[receiver_name]
                num_bits = int(counter["num_bits"])
                num_blocks = int(counter["num_blocks"])
                bit_errors = int(counter["bit_errors"])
                block_errors = int(counter["block_errors"])
                num_nmse_batches = int(counter["num_nmse_batches"])

                row = {
                    "receiver": receiver_name,
                    "num_users": eval_num_users,
                    "ebno_db": ebno_db,
                    "ber": float(bit_errors / num_bits) if num_bits > 0 else np.nan,
                    "bler": float(block_errors / num_blocks) if num_blocks > 0 else np.nan,
                    "nmse": float(counter["nmse_sum"] / num_nmse_batches) if num_nmse_batches > 0 else np.nan,
                    "bit_errors": bit_errors,
                    "num_bits": num_bits,
                    "block_errors": block_errors,
                    "num_blocks": num_blocks,
                    "num_batches_run": int(counter["num_batches_run"]),
                    "reliable_ber": bool(bit_errors >= reliable_min_bit_errors),
                    "reliable_bler": bool(block_errors >= reliable_min_block_errors),
                }
                rows.append(row)
                print(
                    f"[EVAL] receiver={receiver_name:>24s} "
                    f"Eb/N0={ebno_db:>4.1f} dB "
                    f"BER={row['ber']:.5e} "
                    f"BLER={row['bler']:.5e} "
                    f"NMSE={row['nmse']:.5e} "
                    f"bit_err={bit_errors:>6d}/{num_bits:<8d} "
                    f"blk_err={block_errors:>5d}/{num_blocks:<6d} "
                    f"batches={int(counter['num_batches_run']):>4d}"
                )

            completed_ebno.add(float(ebno_db))
            pd.DataFrame(rows).to_csv(curves_path, index=False)
            _save_eval_state(complete=False, reason="periodic", completed=completed_ebno)
    finally:
        for sig, handler in previous_handlers.items():
            try:
                signal.signal(sig, handler)
            except Exception:
                pass

    df = pd.DataFrame(rows)
    df.to_csv(curves_path, index=False)
    summary = _build_summary(
        df=df,
        checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
        enabled_receivers=enabled_receivers,
        artifacts=baseline_artifacts,
        eval_cfg=eval_cfg,
    )
    summary["curves_csv"] = str(curves_path)
    summary["batch_size_eval"] = int(eval_batch_size)
    summary["receiver_microbatch_size"] = int(receiver_microbatch_size)
    summary["seed"] = int(evaluation_seed)
    summary["training_seed"] = int(training_seed)
    summary["evaluation_seed"] = int(evaluation_seed)
    save_json(summary, paths["metrics"] / "evaluation_summary.json")
    _save_eval_state(complete=True, reason="final", completed=set(float(x) for x in ebno_grid))

    return {
        "output_dir": str(paths["root"]),
        "curves_path": str(curves_path),
        "summary_path": str(paths["metrics"] / "evaluation_summary.json"),
        "evaluation_complete": True,
        "completed_ebno_db": [float(x) for x in ebno_grid],
        "training_seed": int(training_seed),
        "evaluation_seed": int(evaluation_seed),
    }
