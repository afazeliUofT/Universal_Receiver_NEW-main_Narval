from __future__ import annotations

import importlib
import inspect
from typing import Any, Callable, Iterable

import tensorflow as tf


def resolve_attr(module_candidates: Iterable[str], attr_name: str) -> Any:
    last_err = None
    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, attr_name):
                return getattr(module, attr_name)
        except Exception as err:  # pragma: no cover - runtime compatibility helper
            last_err = err
    raise ImportError(
        f"Could not resolve '{attr_name}' from modules {list(module_candidates)}"
    ) from last_err


def filter_kwargs_for_callable(fn: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return dict(kwargs)
    accepted = set(signature.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in accepted}


def instantiate_filtered(cls: Callable[..., Any], /, **kwargs: Any) -> Any:
    filtered = filter_kwargs_for_callable(cls, kwargs)
    return cls(**filtered)


def safe_call_variants(fn: Callable[..., Any], *args: Any) -> Any:
    attempts = []
    if len(args) > 1:
        # Sionna/Keras layers commonly expect a single ``inputs`` object such
        # as [y, no]. Trying split positional tensors first can execute a
        # wrong graph branch and allocate large temporary tensors before
        # failing, which is especially painful during long evaluations.
        attempts.append(lambda: fn(list(args)))
        attempts.append(lambda: fn(tuple(args)))
        attempts.append(lambda: fn(*args))
    elif len(args) == 1:
        attempts.append(lambda: fn(args[0]))
    else:
        attempts.append(lambda: fn())
    last_err = None
    for attempt in attempts:
        try:
            return attempt()
        except (tf.errors.ResourceExhaustedError, MemoryError):
            raise
        except Exception as err:  # pragma: no cover - runtime compatibility helper
            last_err = err
    raise RuntimeError(
        f"All calling variants failed for {fn} with {len(args)} argument(s)."
    ) from last_err


def set_if_present(obj: Any, attr_name: str, value: Any) -> bool:
    if obj is None:
        return False
    if hasattr(obj, attr_name):
        try:
            setattr(obj, attr_name, value)
            return True
        except Exception:
            return False
    return False


def first_present_attr(obj: Any, names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if hasattr(obj, name):
            try:
                return getattr(obj, name)
            except Exception:
                continue
    return default
