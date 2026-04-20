"""
Shared Feature Engineering Utilities.

This module provides a single source of truth for all feature transformation
logic shared between the Training Pipeline (feature_engineering.py) and the
Inference Pipeline (predict_api.py).

Centralizing this eliminates Training-Serving Skew — the insidious bug where
the model is trained on features computed one way but served with features
computed differently. Both pipelines MUST import from here.

Rule (Brain vs. Brawn): This is a deterministic Tool. Never ask the LLM to
perform these transformations inline.
"""

import math


def encode_cyclical(value: float, period: float) -> tuple[float, float]:
    """
    Computes a pair of sin/cos cyclical features for a periodic signal.

    This is the canonical cyclical encoding function used by both the FTI
    Training Pipeline and the Inference API. Centralizing it eliminates
    training-serving skew caused by inconsistent period constants.

    How it works:
        By projecting a value onto the unit circle, temporally adjacent points
        (e.g., hour 23 and hour 0) become geometrically adjacent — something
        a linear feature cannot express.

    Args:
        value (float): The raw value to encode (e.g., hour=14, month=3).
        period (float): The full cycle length (e.g., 24 for hours, 12 for months).

    Returns:
        tuple[float, float]: A pair of (sin_value, cos_value) in the range [-1, 1].

    Example:
        >>> sin_h, cos_h = encode_cyclical(hour, 24)
        >>> sin_d, cos_d = encode_cyclical(day_of_week, 7)
        >>> sin_m, cos_m = encode_cyclical(month - 1, 12)
    """
    angle = 2 * math.pi * value / period
    return math.sin(angle), math.cos(angle)
