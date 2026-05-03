from __future__ import annotations

import json
import signal
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from .builders import build_channel, build_ls_estimator, build_pusch_transmitter, extract_pilot_mask, get_resource_grid, max_num_users, multiuser_enabled
from .config import ensure_output_tree, get_cfg
from .estimator import UPAIRChannelEstimator
from .impairments import apply_symbol_phase_impairment
from .utils import (
    call_channel,
    call_transmitter,
    complex_sq_abs,
    compute_nmse,
    ebno_db_to_no,
    save_json,
    save_yaml,
    set_global_seed,
    tf_float,
)


def _make_batch(
    tx: Any,
    channel: Any,
    cfg: dict[str, Any],
    batch_size: int,
    training: bool,
    fixed_ebno_db: float | None = None,
) -> dict[str, tf.Tensor]:
    x, _ = call_transmitter(tx, batch_size)

    if fixed_ebno_db is None:
        ebno_db = tf.random.uniform(
            [],
            minval=float(get_cfg(cfg, "system.ebno_db_train_min", 0.0)),
            maxval=float(get_cfg(cfg, "system.ebno_db_train_max", 16.0)),
            dtype=tf.float32,
        )
    else:
        ebno_db = tf.constant(float(fixed_ebno_db), tf.float32)

    no = ebno_db_to_no(ebno_db, tx=tx, resource_grid=get_resource_grid(tx))
    y, h = call_channel(channel, x, no)
    y, h = apply_symbol_phase_impairment(y, h, cfg, training=training)

    return {
        "y": y,
        "h": h,
        "no": no,
        "ebno_db": ebno_db,
    }


def _sample_train_num_users(cfg: dict[str, Any]) -> int:
    max_users = max_num_users(cfg)
    if max_users <= 1:
        return 1
    sampler = str(get_cfg(cfg, "multiuser.train_user_count_sampler", "triangular")).lower()
    users = np.arange(1, max_users + 1)
    if sampler == "uniform":
        weights = np.ones_like(users, dtype=np.float64)
    elif sampler == "fixed_max":
        return int(max_users)
    else:
        # Triangular weighting: P(U=4)>P(U=3)>P(U=2)>P(U=1).
        weights = users.astype(np.float64)
    probs = weights / weights.sum()
    return int(np.random.choice(users, p=probs))


def _build_system_for_num_users(cfg: dict[str, Any], num_users: int) -> dict[str, Any]:
    tx, _ = build_pusch_transmitter(cfg, num_users=num_users)
    channel = build_channel(cfg, tx)
    ls_estimator = build_ls_estimator(tx, cfg)
    resource_grid = get_resource_grid(tx)
    return {
        "num_users": num_users,
        "tx": tx,
        "channel": channel,
        "ls_estimator": ls_estimator,
        "resource_grid": resource_grid,
        "pilot_mask": extract_pilot_mask(resource_grid),
    }


def _build_training_systems(cfg: dict[str, Any]) -> dict[int, dict[str, Any]]:
    if not multiuser_enabled(cfg):
        return {1: _build_system_for_num_users(cfg, 1)}
    return {
        num_users: _build_system_for_num_users(cfg, num_users)
        for num_users in range(1, max_num_users(cfg) + 1)
    }


def _make_optimizer(cfg: dict[str, Any]) -> tf.keras.optimizers.Optimizer:
    lr = float(cfg["training"]["learning_rate"])
    wd = float(cfg["training"]["weight_decay"])
    common_kwargs = {
        "learning_rate": lr,
        "jit_compile": bool(get_cfg(cfg, "training.optimizer_jit_compile", False)),
    }
    try:
        return tf.keras.optimizers.AdamW(weight_decay=wd, **common_kwargs)
    except TypeError:
        try:
            return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
        except Exception:
            return tf.keras.optimizers.Adam(learning_rate=lr)
    except Exception:
        try:
            return tf.keras.optimizers.Adam(**common_kwargs)
        except TypeError:
            return tf.keras.optimizers.Adam(learning_rate=lr)


@tf.function(reduce_retracing=True)
def _train_step(
    estimator: UPAIRChannelEstimator,
    optimizer: tf.keras.optimizers.Optimizer,
    y: tf.Tensor,
    h: tf.Tensor,
    no: tf.Tensor,
    nmse_loss_weight: float,
    grad_clip_norm: float,
    ls_estimator: Any | None = None,
    pilot_mask: tf.Tensor | None = None,
) -> dict[str, tf.Tensor]:
    with tf.GradientTape() as tape:
        h_hat, err_hat, h_ls, _ = estimator.estimate_with_ls(
            y,
            no,
            training=True,
            ls_estimator=ls_estimator,
            pilot_mask=pilot_mask,
        )
        target = tf.convert_to_tensor(h)

        residual = target - h_hat
        residual_ls = target - h_ls
        sq_err = complex_sq_abs(residual)
        power = tf.reduce_mean(complex_sq_abs(target)) + 1e-9

        loss_nll = tf.reduce_mean(sq_err / (err_hat + 1e-6) + tf.math.log(err_hat + 1e-6))
        nmse_prop = tf.reduce_mean(sq_err) / power
        nmse_ls = tf.reduce_mean(complex_sq_abs(residual_ls)) / power
        loss = loss_nll + float(nmse_loss_weight) * nmse_prop

    grads = tape.gradient(loss, estimator.trainable_variables)
    grad_var_pairs = [(g, v) for g, v in zip(grads, estimator.trainable_variables) if g is not None]
    if grad_var_pairs:
        grad_tensors = [g for g, _ in grad_var_pairs]
        clipped_grads, _ = tf.clip_by_global_norm(grad_tensors, float(grad_clip_norm))
        optimizer.apply_gradients(zip(clipped_grads, [v for _, v in grad_var_pairs]))

    return {
        "loss": tf.cast(loss, tf.float32),
        "loss_nll": tf.cast(loss_nll, tf.float32),
        "nmse_prop": tf.cast(nmse_prop, tf.float32),
        "nmse_ls": tf.cast(nmse_ls, tf.float32),
    }


@tf.function(reduce_retracing=True)
def _validation_step(
    estimator: UPAIRChannelEstimator,
    y: tf.Tensor,
    h: tf.Tensor,
    no: tf.Tensor,
    ls_estimator: Any | None = None,
    pilot_mask: tf.Tensor | None = None,
) -> dict[str, tf.Tensor]:
    h_hat, _, h_ls, _ = estimator.estimate_with_ls(
        y,
        no,
        training=False,
        ls_estimator=ls_estimator,
        pilot_mask=pilot_mask,
    )
    return {
        "nmse_prop": compute_nmse(h, h_hat),
        "nmse_ls": compute_nmse(h, h_ls),
    }


def _gpu_memory_message() -> str:
    try:
        info = tf.config.experimental.get_memory_info("GPU:0")
    except Exception:
        return ""
    current_gib = float(info.get("current", 0)) / (1024.0**3)
    peak_gib = float(info.get("peak", 0)) / (1024.0**3)
    return f" gpu_mem={current_gib:.2f}GiB peak={peak_gib:.2f}GiB"


def _validate(
    estimator: UPAIRChannelEstimator,
    systems: dict[int, dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, float]:
    val_steps = int(cfg["training"]["val_steps"])
    eval_grid = list(get_cfg(cfg, "system.ebno_db_eval", [10]))
    ebno_for_val = float(eval_grid[min(len(eval_grid) // 2, len(eval_grid) - 1)])

    nmse_prop = []
    nmse_ls = []

    for _ in range(val_steps):
        system = systems[_sample_train_num_users(cfg)]
        batch = _make_batch(
            tx=system["tx"],
            channel=system["channel"],
            cfg=cfg,
            batch_size=int(cfg["system"]["batch_size_eval"]),
            training=False,
            fixed_ebno_db=ebno_for_val,
        )
        metrics = _validation_step(
            estimator=estimator,
            y=batch["y"],
            h=batch["h"],
            no=batch["no"],
            ls_estimator=system["ls_estimator"],
            pilot_mask=system["pilot_mask"],
        )
        nmse_prop.append(float(metrics["nmse_prop"].numpy()))
        nmse_ls.append(float(metrics["nmse_ls"].numpy()))

    return {
        "val_nmse_prop": float(np.mean(nmse_prop)),
        "val_nmse_ls": float(np.mean(nmse_ls)),
        "val_ebno_db": ebno_for_val,
    }


def _load_history(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload.get("history", [])
        if isinstance(rows, list):
            return [dict(row) for row in rows if isinstance(row, dict)]
    except Exception as exc:
        print(f"[TRAIN] ignoring unreadable history file {path}: {exc!r}")
    return []


def _warmup_estimator(
    estimator: UPAIRChannelEstimator,
    system: dict[str, Any],
    cfg: dict[str, Any],
) -> None:
    eval_grid = list(get_cfg(cfg, "system.ebno_db_eval", [10]))
    batch = _make_batch(
        tx=system["tx"],
        channel=system["channel"],
        cfg=cfg,
        batch_size=max(1, min(2, int(cfg["system"]["batch_size_train"]))),
        training=False,
        fixed_ebno_db=float(eval_grid[min(len(eval_grid) // 2, len(eval_grid) - 1)]),
    )
    estimator.estimate_with_ls(
        batch["y"],
        batch["no"],
        training=False,
        ls_estimator=system["ls_estimator"],
        pilot_mask=system["pilot_mask"],
    )


def train_model(cfg: dict[str, Any]) -> dict[str, Any]:
    set_global_seed(int(cfg["system"]["seed"]))
    if bool(get_cfg(cfg, "system.graph_mode", True)):
        tf.config.run_functions_eagerly(False)
    paths = ensure_output_tree(cfg)

    systems = _build_training_systems(cfg)
    reference_system = systems[max(systems)]
    estimator = UPAIRChannelEstimator(
        ls_estimator=reference_system["ls_estimator"],
        resource_grid=reference_system["resource_grid"],
        cfg=cfg,
    )
    optimizer = _make_optimizer(cfg)

    ckpt_path = paths["checkpoints"] / str(cfg["training"]["checkpoint_name"])
    history_path = paths["metrics"] / "history.json"
    train_state_path = paths["metrics"] / "train_state.json"
    state_dir = paths["checkpoints"] / "training_state"

    _warmup_estimator(estimator, reference_system, cfg)
    try:
        optimizer.build(estimator.trainable_variables)
    except Exception:
        pass

    step_var = tf.Variable(0, dtype=tf.int64, trainable=False, name="training_step")
    best_val_var = tf.Variable(np.inf, dtype=tf.float32, trainable=False, name="best_val")
    training_ckpt = tf.train.Checkpoint(
        step=step_var,
        best_val=best_val_var,
        optimizer=optimizer,
        estimator=estimator,
    )
    manager = tf.train.CheckpointManager(
        training_ckpt,
        directory=str(state_dir),
        max_to_keep=int(get_cfg(cfg, "training.max_resume_checkpoints", 3)),
        checkpoint_name="ckpt",
    )

    total_steps = int(cfg["training"]["steps"])
    log_every = int(cfg["training"]["log_every"])
    eval_every = int(cfg["training"]["eval_every"])
    checkpoint_every = int(get_cfg(cfg, "training.checkpoint_every", log_every))
    nmse_loss_weight = float(cfg["training"]["nmse_loss_weight"])
    grad_clip_norm = float(cfg["training"]["grad_clip_norm"])
    resume_enabled = bool(get_cfg(cfg, "training.resume", True))

    history: list[dict[str, float]] = []
    best_val = float("inf")
    start_step = 1

    if resume_enabled and manager.latest_checkpoint:
        status = training_ckpt.restore(manager.latest_checkpoint)
        status.expect_partial()
        start_step = int(step_var.numpy()) + 1
        best_val = float(best_val_var.numpy())
        history = _load_history(history_path)
        print(
            f"[TRAIN] resumed from {manager.latest_checkpoint} "
            f"at completed_step={start_step - 1} best_val={best_val:.6g}"
        )
    else:
        print("[TRAIN] starting from scratch")

    stop_requested = False

    def _request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"[TRAIN] received signal {signum}; will save and stop after current step.")

    previous_handlers: dict[int, Any] = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, _request_stop)
        except Exception:
            pass

    def _save_progress(step: int, reason: str, complete: bool = False) -> str:
        step_var.assign(int(step))
        best_val_var.assign(float(best_val))
        latest_checkpoint = manager.save(checkpoint_number=int(step))
        save_json({"history": history}, history_path)
        save_json(
            {
                "latest_step": int(step),
                "total_steps": int(total_steps),
                "training_complete": bool(complete),
                "best_val": float(best_val),
                "best_weights_path": str(ckpt_path),
                "latest_training_state_checkpoint": str(latest_checkpoint),
                "resume_enabled": bool(resume_enabled),
                "save_reason": reason,
            },
            train_state_path,
        )
        print(f"[TRAIN] saved resumable state at step={step} reason={reason}: {latest_checkpoint}")
        return latest_checkpoint

    current_step = start_step - 1
    last_completed_step = start_step - 1
    training_complete = last_completed_step >= total_steps

    try:
        for step in range(start_step, total_steps + 1):
            current_step = step
            system = systems[_sample_train_num_users(cfg)]
            batch = _make_batch(
                tx=system["tx"],
                channel=system["channel"],
                cfg=cfg,
                batch_size=int(cfg["system"]["batch_size_train"]),
                training=True,
            )
            metrics = _train_step(
                estimator=estimator,
                optimizer=optimizer,
                y=batch["y"],
                h=batch["h"],
                no=batch["no"],
                nmse_loss_weight=nmse_loss_weight,
                grad_clip_norm=grad_clip_norm,
                ls_estimator=system["ls_estimator"],
                pilot_mask=system["pilot_mask"],
            )
            row = {
                "step": step,
                "num_users": int(system["num_users"]),
                "ebno_db": float(batch["ebno_db"].numpy()),
                "loss": float(metrics["loss"].numpy()),
                "loss_nll": float(metrics["loss_nll"].numpy()),
                "nmse_prop": float(metrics["nmse_prop"].numpy()),
                "nmse_ls": float(metrics["nmse_ls"].numpy()),
            }

            did_validate = False
            if step % eval_every == 0 or step == total_steps:
                did_validate = True
                val_metrics = _validate(estimator, systems, cfg)
                row.update(val_metrics)
                if val_metrics["val_nmse_prop"] < best_val:
                    best_val = val_metrics["val_nmse_prop"]
                    estimator.save_weights(str(ckpt_path))
                    print(f"[TRAIN] saved new best weights at step={step} val_nmse={best_val:.6g}")

            history.append(row)
            last_completed_step = step
            current_step = last_completed_step

            if step % log_every == 0 or step == 1 or step == total_steps:
                print(
                    f"[TRAIN] step={step:05d} "
                    f"loss={row['loss']:.5f} "
                    f"nmse_prop={row['nmse_prop']:.5f} "
                    f"nmse_ls={row['nmse_ls']:.5f}"
                    f"{_gpu_memory_message()}"
                )

            if step % checkpoint_every == 0 or did_validate or step == total_steps or stop_requested:
                _save_progress(step, "periodic" if not stop_requested else "signal", complete=step >= total_steps)

            if stop_requested:
                print("[TRAIN] stopped early after saving resumable state; resubmit the Slurm task to continue.")
                break
        training_complete = last_completed_step >= total_steps and not stop_requested
    except KeyboardInterrupt:
        _save_progress(last_completed_step, "keyboard_interrupt", complete=False)
        raise
    except tf.errors.ResourceExhaustedError:
        _save_progress(last_completed_step, "resource_exhausted", complete=False)
        raise
    finally:
        for sig, handler in previous_handlers.items():
            try:
                signal.signal(sig, handler)
            except Exception:
                pass

    if not ckpt_path.exists():
        estimator.save_weights(str(ckpt_path))

    _save_progress(last_completed_step, "final" if training_complete else "incomplete", complete=training_complete)
    save_json(
        {
            "num_trainable_params": int(np.sum([np.prod(v.shape) for v in estimator.trainable_variables])),
            "multiuser_enabled": bool(multiuser_enabled(cfg)),
            "max_num_users": int(max_num_users(cfg)),
            "train_user_count_sampler": str(get_cfg(cfg, "multiuser.train_user_count_sampler", "triangular")),
            "training_complete": bool(training_complete),
            "latest_step": int(last_completed_step),
            "total_steps": int(total_steps),
        },
        paths["metrics"] / "model_summary.json",
    )
    save_yaml(cfg, paths["artifacts"] / "resolved_config.yaml")

    return {
        "output_dir": str(paths["root"]),
        "checkpoint_path": str(ckpt_path),
        "history_path": str(history_path),
        "model_summary_path": str(paths["metrics"] / "model_summary.json"),
        "train_state_path": str(train_state_path),
        "training_complete": bool(training_complete),
        "latest_step": int(last_completed_step),
        "total_steps": int(total_steps),
    }
