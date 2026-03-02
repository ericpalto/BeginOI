from __future__ import annotations

from typing import Any, Mapping
from dataclasses import asdict, fields, replace, is_dataclass

import numpy as np


def _is_numeric(x: Any) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating, np.ndarray))


def apply_set(params: Any, updates: Mapping[str, Any]) -> Any:
    if not updates:
        return params
    if is_dataclass(params):
        known = {f.name for f in fields(params)}
        unknown = set(updates) - known
        if unknown:
            raise ValueError(f"Unknown param fields: {sorted(unknown)}")
        return replace(params, **dict(updates))
    if isinstance(params, dict):
        out = dict(params)
        out.update(dict(updates))
        return out
    raise TypeError(f"Unsupported param type for apply_set: {type(params).__name__}")


def apply_delta(params: Any, deltas: Mapping[str, Any]) -> Any:
    if not deltas:
        return params
    if is_dataclass(params):
        current = asdict(params)
        updated: dict[str, Any] = {}
        for k, dv in deltas.items():
            if k not in current:
                raise ValueError(f"Unknown param field: {k}")
            v = current[k]
            if not _is_numeric(v) or not _is_numeric(dv):
                raise TypeError(f"Non-numeric delta for field {k!r}.")
            updated[k] = np.asarray(v) + np.asarray(dv)
        return apply_set(params, updated)
    if isinstance(params, dict):
        out = dict(params)
        for k, dv in deltas.items():
            v = out.get(k, 0.0)
            if not _is_numeric(v) or not _is_numeric(dv):
                raise TypeError(f"Non-numeric delta for field {k!r}.")
            out[k] = np.asarray(v) + np.asarray(dv)
        return out
    raise TypeError(f"Unsupported param type for apply_delta: {type(params).__name__}")
