"""
Tiny shared utility. Exists because `value or default` is a real bug when
`value` can be NaN: float NaN is truthy in Python, so `NaN or 0.0` evaluates
to NaN, not 0.0. That silently poisons downstream math with NaN instead of
falling back the way the code clearly intends to. Every place in this project
that wants "use this value, or a default if it's missing/NaN/None" should
call safe_num instead of writing `x or default`.
"""

import numpy as np


def safe_num(value, default: float = 0.0) -> float:
    """Return float(value), or `default` if value is None or NaN."""
    if value is None:
        return float(default)
    if isinstance(value, float) and np.isnan(value):
        return float(default)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    return default if np.isnan(v) else v


def safe_int(value, default: int = 0) -> int:
    """Same as safe_num but returns an int; NaN/None both fall back cleanly."""
    return int(safe_num(value, default))
