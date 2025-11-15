"""Shared helpers for camera geometry and ray construction."""

from __future__ import annotations

import ast
import math
from typing import Iterable, Sequence, Tuple

import numpy as np
from tf_conversions import transformations


_DEFAULT_OFFSET = (0.15, 0.0, 0.05)


def _coerce_sequence(value: Sequence[float] | Iterable[float], fallback: Tuple[float, float, float]) -> np.ndarray:
    try:
        items = list(value)
    except TypeError:
        return np.array(fallback, dtype=np.float32)
    if len(items) < 3:
        items = list(fallback)
    try:
        return np.array([float(items[0]), float(items[1]), float(items[2])], dtype=np.float32)
    except (TypeError, ValueError):
        return np.array(fallback, dtype=np.float32)


def parse_camera_offset(raw_value, default: Tuple[float, float, float] = _DEFAULT_OFFSET) -> np.ndarray:
    """Parses ROS parameters or YAML strings into a numeric offset vector."""

    if isinstance(raw_value, str):
        try:
            parsed = ast.literal_eval(raw_value)
        except (SyntaxError, ValueError):
            parsed = raw_value
    else:
        parsed = raw_value
    if isinstance(parsed, np.ndarray):
        if parsed.size >= 3:
            return parsed.astype(np.float32)
        return np.array(default, dtype=np.float32)
    if isinstance(parsed, (tuple, list)):
        return _coerce_sequence(parsed, default)
    if isinstance(parsed, Iterable):
        return _coerce_sequence(parsed, default)
    return np.array(default, dtype=np.float32)


def camera_mount_from_pitch(pitch_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Returns quaternion + rotation matrix for an upward pitched camera mount."""

    pitch_rad = -math.radians(float(pitch_deg))
    quat = transformations.quaternion_from_euler(0.0, pitch_rad, 0.0)
    matrix = transformations.quaternion_matrix(quat)[0:3, 0:3].astype(np.float32)
    return quat, matrix


def precompute_unit_rays(width: int, height: int, fov_deg: float) -> np.ndarray:
    """Builds normalized ray directions in the camera frame (flattened)."""

    width = max(int(width), 1)
    height = max(int(height), 1)
    fov = math.radians(float(fov_deg))
    aspect = height / float(width)
    tan_half_h = math.tan(fov / 2.0)
    tan_half_v = tan_half_h * aspect

    u = (np.arange(width, dtype=np.float32) + 0.5) / float(width)
    v = (np.arange(height, dtype=np.float32) + 0.5) / float(height)
    u = (u * 2.0) - 1.0
    v = 1.0 - (v * 2.0)

    x_components = u * tan_half_h
    y_components = v[:, np.newaxis] * tan_half_v

    ones = np.ones((height, width), dtype=np.float32)
    x_grid = np.broadcast_to(x_components, (height, width))
    y_grid = np.broadcast_to(y_components, (height, width))

    local_dirs = np.stack((ones, x_grid, y_grid), axis=-1)
    norms = np.linalg.norm(local_dirs, axis=-1, keepdims=True)
    norms = np.clip(norms, 1e-6, None)
    local_dirs = local_dirs / norms
    return local_dirs.reshape(-1, 3)

