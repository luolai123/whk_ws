"""Helper utilities for extracting safe regions and mapping navigation cues."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class SafeRegion:
    """Represents a contiguous safe component in the classification image."""

    area: int
    centroid: Tuple[float, float]
    mask: np.ndarray
    bounds: Tuple[int, int, int, int]


def find_largest_safe_region(
    safe_mask: np.ndarray, min_fraction: float = 0.05
) -> Optional[SafeRegion]:
    """Return the largest safe component whose area exceeds ``min_fraction``."""

    if safe_mask.ndim != 2:
        raise ValueError("safe_mask must be a 2-D array")

    height, width = safe_mask.shape
    total_pixels = height * width
    min_pixels = max(1, int(total_pixels * float(min_fraction)))

    visited = np.zeros_like(safe_mask, dtype=bool)
    best_region: Optional[SafeRegion] = None

    indices = np.argwhere(safe_mask)
    if indices.size == 0:
        return None

    for start_row, start_col in indices:
        if visited[start_row, start_col]:
            continue

        queue: deque[Tuple[int, int]] = deque()
        queue.append((start_row, start_col))
        visited[start_row, start_col] = True

        area = 0
        sum_row = 0.0
        sum_col = 0.0
        min_r = start_row
        max_r = start_row
        min_c = start_col
        max_c = start_col

        while queue:
            row, col = queue.pop()
            area += 1
            sum_row += row
            sum_col += col
            if row < min_r:
                min_r = row
            elif row > max_r:
                max_r = row
            if col < min_c:
                min_c = col
            elif col > max_c:
                max_c = col

            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = row + dr
                nc = col + dc
                if 0 <= nr < height and 0 <= nc < width:
                    if safe_mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

        if area >= min_pixels and (best_region is None or area > best_region.area):
            centroid = (sum_row / area, sum_col / area)
            mask_slice = safe_mask[min_r : max_r + 1, min_c : max_c + 1]
            region_mask = mask_slice.copy()
            bounds = (min_r, max_r, min_c, max_c)
            best_region = SafeRegion(area=area, centroid=centroid, mask=region_mask, bounds=bounds)

    return best_region


def compute_direction_from_pixel(
    col: float, row: float, width: int, height: int, fov_deg: float
) -> np.ndarray:
    """Convert a pixel coordinate into a unit direction vector in the camera frame."""

    width = max(1, int(width))
    height = max(1, int(height))

    fov_rad = math.radians(float(fov_deg))
    tan_half_h = math.tan(fov_rad / 2.0)
    aspect = height / float(width)
    tan_half_v = tan_half_h * aspect

    u = ((float(col) + 0.5) / width) * 2.0 - 1.0
    v = 1.0 - ((float(row) + 0.5) / height) * 2.0

    x_component = u * tan_half_h
    y_component = v * tan_half_v

    direction = np.array([1.0, x_component, y_component], dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return direction / norm


def rotate_direction(
    base_direction: np.ndarray, yaw_offset: float, pitch_offset: float
) -> np.ndarray:
    """Rotate ``base_direction`` by yaw and pitch offsets in the camera/base frame."""

    if base_direction.shape != (3,):
        raise ValueError("base_direction must be a 3-vector")

    cy = math.cos(yaw_offset)
    sy = math.sin(yaw_offset)
    cp = math.cos(pitch_offset)
    sp = math.sin(pitch_offset)

    r_yaw = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    r_pitch = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)

    rotated = r_yaw.dot(r_pitch.dot(base_direction.astype(np.float32)))
    norm = float(np.linalg.norm(rotated))
    if norm < 1e-6:
        return base_direction.astype(np.float32)
    return rotated / norm


def project_direction_to_pixel(
    direction: np.ndarray, width: int, height: int, fov_deg: float
) -> Tuple[float, float]:
    """Project a direction vector back to fractional pixel coordinates."""

    width = max(1, int(width))
    height = max(1, int(height))

    direction = direction.astype(np.float32)
    denom = float(direction[0])
    if abs(denom) < 1e-6:
        denom = 1e-6 if denom >= 0 else -1e-6

    horizontal = math.atan2(float(direction[1]), denom)
    vertical = math.atan2(float(direction[2]), math.sqrt(direction[0] ** 2 + direction[1] ** 2))

    fov_rad = math.radians(float(fov_deg))
    tan_half_h = math.tan(fov_rad / 2.0)
    aspect = height / float(width)
    tan_half_v = tan_half_h * aspect

    u = math.tan(horizontal) / tan_half_h
    v = math.tan(vertical) / tan_half_v

    u = max(-1.0, min(1.0, u))
    v = max(-1.0, min(1.0, v))

    col = ((u + 1.0) * 0.5) * width - 0.5
    row = (1.0 - (v + 1.0) * 0.5) * height - 0.5
    return col, row


def is_pixel_safe(safe_mask: np.ndarray, col: float, row: float) -> bool:
    """Return ``True`` if the rounded pixel coordinate lies in the safe mask."""

    row_idx = int(round(row))
    col_idx = int(round(col))
    return (
        0 <= row_idx < safe_mask.shape[0]
        and 0 <= col_idx < safe_mask.shape[1]
        and bool(safe_mask[row_idx, col_idx])
    )


def _smoothstep(t: float) -> float:
    """Cubic smoothstep used to ease YOPO-style trajectory blends."""

    t = max(0.0, min(1.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


def sample_yopo_directions(
    base_direction: np.ndarray,
    yaw_offset: float,
    pitch_offset: float,
    steps: int,
) -> List[np.ndarray]:
    """Return a list of interpolated directions along a YOPO-like primitive.

    The offsets are progressively blended with a ``smoothstep`` profile so the
    early portion of the trajectory closely follows ``base_direction`` while
    the later portion converges on the fully offset heading.  This mirrors the
    "You Only Plan Once" (YOPO) idea of precomputing smooth motion primitives
    that can be evaluated quickly during inference.
    """

    steps = max(1, int(steps))
    directions: List[np.ndarray] = []
    for idx in range(1, steps + 1):
        fraction = idx / float(steps)
        blend = _smoothstep(fraction)
        directions.append(
            rotate_direction(
                base_direction,
                yaw_offset * blend,
                pitch_offset * blend,
            )
        )
    return directions


def cubic_hermite_path(
    start: np.ndarray,
    goal: np.ndarray,
    tangent_start: np.ndarray,
    tangent_end: np.ndarray,
    steps: int,
) -> np.ndarray:
    """Generate a cubic Hermite path blending a primitive into a goal."""

    start = np.asarray(start, dtype=np.float32)
    goal = np.asarray(goal, dtype=np.float32)
    tangent_start = np.asarray(tangent_start, dtype=np.float32)
    tangent_end = np.asarray(tangent_end, dtype=np.float32)
    steps = max(1, int(steps))

    points = np.zeros((steps + 1, 3), dtype=np.float32)
    for idx in range(steps + 1):
        t = idx / float(steps)
        t2 = t * t
        t3 = t2 * t
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        points[idx] = (
            h00 * start
            + h10 * tangent_start
            + h01 * goal
            + h11 * tangent_end
        )
    return points


def path_smoothness(points: Iterable[np.ndarray]) -> float:
    """Return a smoothness score based on successive direction changes."""

    vectors: List[np.ndarray] = []
    previous: Optional[np.ndarray] = None
    for pt in points:
        current = np.asarray(pt, dtype=np.float32)
        if previous is not None:
            delta = current - previous
            norm = float(np.linalg.norm(delta))
            if norm > 1e-6:
                vectors.append(delta / norm)
        previous = current

    if len(vectors) < 2:
        return 1.0

    total_angle = 0.0
    for idx in range(1, len(vectors)):
        dot = float(np.dot(vectors[idx - 1], vectors[idx]))
        dot = max(-1.0, min(1.0, dot))
        total_angle += abs(math.acos(dot))

    avg_angle = total_angle / max(1, len(vectors) - 1)
    return math.exp(-avg_angle)


def jerk_metrics(points: Iterable[np.ndarray], dt: float) -> Tuple[float, float]:
    """Return a smoothness score and the maximum jerk magnitude."""

    samples = [np.asarray(pt, dtype=np.float32) for pt in points]
    if len(samples) < 4:
        return 1.0, 0.0

    dt = max(float(dt), 1e-3)
    velocities = np.diff(samples, axis=0) / dt
    if velocities.shape[0] < 3:
        return 1.0, 0.0
    accelerations = np.diff(velocities, axis=0) / dt
    if accelerations.shape[0] < 2:
        return 1.0, 0.0
    jerks = np.diff(accelerations, axis=0) / dt
    if jerks.size == 0:
        return 1.0, 0.0

    norms = np.linalg.norm(jerks, axis=1)
    max_jerk = float(np.max(norms))
    return math.exp(-max_jerk), max_jerk


def jerk_score(points: Iterable[np.ndarray], dt: float) -> float:
    """Return the smoothness score associated with ``jerk_metrics``."""

    score, _ = jerk_metrics(points, dt)
    return score


def orientation_rate_stats(
    directions: Iterable[np.ndarray], dt: float = 1.0
) -> Tuple[float, float]:
    """Return an orientation score and the max heading-change rate."""

    unit_vectors: List[np.ndarray] = []
    for direction in directions:
        vec = np.asarray(direction, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            continue
        unit_vectors.append(vec / norm)

    if len(unit_vectors) < 2:
        return 1.0, 0.0

    max_angle = 0.0
    for idx in range(1, len(unit_vectors)):
        dot = float(np.dot(unit_vectors[idx - 1], unit_vectors[idx]))
        dot = max(-1.0, min(1.0, dot))
        max_angle = max(max_angle, abs(math.acos(dot)))

    dt = max(float(dt), 1e-3)
    rate = max_angle / dt
    return math.exp(-max_angle), rate


def orientation_rate_score(directions: Iterable[np.ndarray]) -> float:
    """Return the smoothness score for heading changes."""

    score, _ = orientation_rate_stats(directions)
    return score


def clamp_normalized(vec: np.ndarray) -> np.ndarray:
    """Return ``vec`` normalized to unit length with safe fallback."""

    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return vec / norm
