"""Helper utilities for extracting safe regions and mapping navigation cues."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


def clamp_normalized(vec: np.ndarray) -> np.ndarray:
    """Return ``vec`` normalized to unit length with safe fallback."""

    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return vec / norm


def _extract_rotation(matrix: np.ndarray) -> np.ndarray:
    """Return a 3x3 rotation matrix from a 3x3/3x4/4x4 transform."""

    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape == (4, 4):
        matrix = matrix[0:3, 0:3]
    elif matrix.shape == (3, 4):
        matrix = matrix[:, 0:3]
    if matrix.shape != (3, 3):
        raise ValueError("camera_to_body must be a 3x3, 3x4, or 4x4 matrix")
    return matrix


@dataclass
class PrimitiveConfig:
    """Parameters shared by the YOPO-style primitive sampler."""

    radio_range: float = 5.0
    vel_max_train: float = 6.0
    acc_max_train: float = 3.0
    forward_log_mean: float = math.log(2.0)
    forward_log_sigma: float = 0.45
    v_std_unit: float = 0.22
    a_std_unit: float = 0.35
    yaw_range_deg: float = 360.0
    pitch_std_deg: float = 30.0
    roll_std_deg: float = 30.0
    horizon_camera_fov: float = 90.0
    vertical_camera_fov: float = 60.0
    goal_length_scale: float = 1.0
    offset_gain: float = 0.25

    @property
    def traj_time(self) -> float:
        vel = max(self.vel_max_train, 0.5)
        return max(0.5, 2.0 * self.radio_range / vel)


def denormalize_weight(weight: float, speed: float, reference_speed: float) -> float:
    """Scale ``weight`` based on the commanded ``speed``.

    The scaling keeps smoothness-relevant costs dominant when the vehicle
    accelerates, discouraging short-horizon jumps in the chosen primitives.
    """

    reference_speed = max(reference_speed, 1e-3)
    # Emphasize smoothness at higher speeds while keeping a stable lower bound.
    scale = max(1.0, speed / reference_speed)
    return weight * scale


@dataclass
class PrimitiveSample:
    """Represents a single sampled primitive in the drone body frame."""

    base_direction_camera: np.ndarray
    goal_direction_body: np.ndarray
    start_vel_body: np.ndarray
    start_acc_body: np.ndarray
    yaw_offset: float
    pitch_offset: float
    roll_offset: float
    goal_length: float
    duration: float


def _local_basis(forward: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = clamp_normalized(forward)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    lateral = np.cross(up, forward)
    lateral = clamp_normalized(lateral)
    if np.linalg.norm(lateral) < 1e-5:
        lateral = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    vertical = np.cross(forward, lateral)
    vertical = clamp_normalized(vertical)
    return forward, lateral, vertical


def _clamp_angle(angle: float, limit_deg: float) -> float:
    limit = math.radians(max(1e-3, float(limit_deg)))
    return max(-limit, min(limit, angle))


def sample_motion_primitives(
    base_direction_camera: np.ndarray,
    camera_to_body: np.ndarray,
    rng: np.random.Generator,
    config: PrimitiveConfig,
    count: int,
) -> List[PrimitiveSample]:
    """Draw ``count`` YOPO-style primitives aligned with ``base_direction_camera``."""

    base_direction_camera = clamp_normalized(base_direction_camera)
    camera_to_body = _extract_rotation(camera_to_body)
    samples: List[PrimitiveSample] = []
    count = max(1, int(count))
    yaw_limit = max(0.0, float(config.yaw_range_deg)) * 0.5
    yaw_limit_rad = math.radians(yaw_limit)
    pitch_limit = 90.0
    roll_limit = 90.0

    for _ in range(count):
        yaw_offset = _clamp_angle(
            rng.uniform(-yaw_limit_rad, yaw_limit_rad),
            yaw_limit,
        )
        pitch_offset = _clamp_angle(
            rng.normal(0.0, math.radians(config.pitch_std_deg)), pitch_limit
        )
        roll_offset = _clamp_angle(
            rng.normal(0.0, math.radians(config.roll_std_deg)), roll_limit
        )
        offset_direction_camera = rotate_direction(
            base_direction_camera, yaw_offset, pitch_offset, roll_offset
        )
        goal_direction_body = clamp_normalized(camera_to_body.dot(offset_direction_camera))

        vx = float(
            np.clip(
                rng.lognormal(config.forward_log_mean, config.forward_log_sigma),
                0.0,
                config.vel_max_train,
            )
        )
        vy = float(
            np.clip(
                rng.normal(0.0, config.vel_max_train * config.v_std_unit),
                -config.vel_max_train,
                config.vel_max_train,
            )
        )
        vz = float(
            np.clip(
                rng.normal(0.0, config.vel_max_train * config.v_std_unit * 0.8),
                -config.vel_max_train,
                config.vel_max_train,
            )
        )
        start_vel = np.array([vx, vy, vz], dtype=np.float32)

        ax = float(
            np.clip(
                rng.normal(0.0, config.acc_max_train * config.a_std_unit),
                -config.acc_max_train,
                config.acc_max_train,
            )
        )
        ay = float(
            np.clip(
                rng.normal(0.0, config.acc_max_train * config.a_std_unit),
                -config.acc_max_train,
                config.acc_max_train,
            )
        )
        az = float(
            np.clip(
                rng.normal(0.0, config.acc_max_train * config.a_std_unit),
                -config.acc_max_train,
                config.acc_max_train,
            )
        )
        start_acc = np.array([ax, ay, az], dtype=np.float32)

        length_scale = float(np.clip(rng.normal(config.goal_length_scale, 0.15), 0.4, 1.4))
        goal_length = config.radio_range * length_scale
        duration = max(0.2, config.traj_time * length_scale)

        samples.append(
            PrimitiveSample(
                base_direction_camera=offset_direction_camera,
                goal_direction_body=goal_direction_body,
                start_vel_body=start_vel,
                start_acc_body=start_acc,
                yaw_offset=yaw_offset,
                pitch_offset=pitch_offset,
                roll_offset=roll_offset,
                goal_length=goal_length,
                duration=duration,
            )
        )

    return samples


def primitive_state_vector(sample: PrimitiveSample, config: PrimitiveConfig) -> np.ndarray:
    """Return normalized features describing ``sample`` for the policy network."""

    forward, lateral, vertical = _local_basis(sample.goal_direction_body)
    vel = sample.start_vel_body
    acc = sample.start_acc_body
    vel_forward = float(np.dot(vel, forward)) / max(config.vel_max_train, 1e-3)
    vel_lateral = float(np.dot(vel, lateral)) / max(config.vel_max_train, 1e-3)
    vel_vertical = float(np.dot(vel, vertical)) / max(config.vel_max_train, 1e-3)
    acc_forward = float(np.dot(acc, forward)) / max(config.acc_max_train, 1e-3)
    acc_lateral = float(np.dot(acc, lateral)) / max(config.acc_max_train, 1e-3)
    acc_vertical = float(np.dot(acc, vertical)) / max(config.acc_max_train, 1e-3)
    goal_norm = float(sample.goal_length / max(config.radio_range, 1e-3))
    duration_norm = float(sample.duration / max(config.traj_time, 1e-3))
    goal_dir = clamp_normalized(sample.goal_direction_body)
    orientation_offsets = np.array(
        [sample.yaw_offset, sample.pitch_offset, sample.roll_offset], dtype=np.float32
    )
    orientation_norm = orientation_offsets / math.pi

    return np.array(
        [
            vel_forward,
            vel_lateral,
            vel_vertical,
            acc_forward,
            acc_lateral,
            acc_vertical,
            goal_norm,
            duration_norm,
            goal_dir[0],
            goal_dir[1],
            goal_dir[2],
            orientation_norm[0],
            orientation_norm[1],
            orientation_norm[2],
        ],
        dtype=np.float32,
    )


def primitive_state_dim(config: PrimitiveConfig) -> int:
    """Return the length of the policy state vector for ``config``."""

    dummy = PrimitiveSample(
        base_direction_camera=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        goal_direction_body=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        start_vel_body=np.zeros(3, dtype=np.float32),
        start_acc_body=np.zeros(3, dtype=np.float32),
        yaw_offset=0.0,
        pitch_offset=0.0,
        roll_offset=0.0,
        goal_length=config.radio_range,
        duration=config.traj_time,
    )
    return int(len(primitive_state_vector(dummy, config)))


def apply_goal_offset(
    sample: PrimitiveSample, offset: Sequence[float], config: PrimitiveConfig
) -> np.ndarray:
    """Return the offset goal (body frame) after applying the network output."""

    offset = np.asarray(offset, dtype=np.float32)
    if offset.shape != (3,):
        raise ValueError("offset must be length-3")
    base_goal = sample.goal_direction_body * sample.goal_length
    max_offset = config.radio_range * config.offset_gain
    offset_norm = float(np.linalg.norm(offset))
    if offset_norm > 1e-6:
        offset = offset / offset_norm * min(offset_norm, max_offset)
    goal = base_goal + offset
    max_range = config.radio_range * 1.5
    goal_norm = float(np.linalg.norm(goal))
    if goal_norm > max_range:
        goal = goal / goal_norm * max_range
    return goal


def normalize_navigation_inputs(
    position: Sequence[float],
    velocity: Sequence[float],
    goal_point: Sequence[float],
    config: PrimitiveConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize navigation-related vectors for model consumption.

    ``position`` and ``goal_point`` are scaled by ``config.radio_range`` while
    ``velocity`` is scaled by ``config.vel_max_train``. The function returns
    float32 numpy arrays to keep training and inference inputs consistent and
    dimensionless.
    """

    range_scale = max(float(config.radio_range), 1e-3)
    vel_scale = max(float(config.vel_max_train), 1e-3)

    pos_arr = np.asarray(position, dtype=np.float32) / range_scale
    vel_arr = np.asarray(velocity, dtype=np.float32) / vel_scale
    goal_arr = np.asarray(goal_point, dtype=np.float32) / range_scale
    return pos_arr.astype(np.float32), vel_arr.astype(np.float32), goal_arr.astype(np.float32)


def primitive_quintic_trajectory(
    sample: PrimitiveSample,
    goal_body: np.ndarray,
    duration_scale: float,
    steps: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return body-frame samples along the quintic primitive."""

    duration = max(0.2, float(sample.duration) * float(duration_scale))
    goal_body = np.asarray(goal_body, dtype=np.float32)
    coeffs = quintic_coefficients(
        np.zeros(3, dtype=np.float32),
        sample.start_vel_body,
        sample.start_acc_body,
        goal_body,
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        duration,
    )
    sample_count = max(steps, 1)
    points, velocities = sample_quintic(coeffs, duration, sample_count)
    return points, velocities, duration

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
    base_direction: np.ndarray,
    yaw_offset: float,
    pitch_offset: float,
    roll_offset: float = 0.0,
) -> np.ndarray:
    """Rotate ``base_direction`` by yaw, pitch, and roll offsets in the camera/base frame."""

    if base_direction.shape != (3,):
        raise ValueError("base_direction must be a 3-vector")

    cy = math.cos(yaw_offset)
    sy = math.sin(yaw_offset)
    cp = math.cos(pitch_offset)
    sp = math.sin(pitch_offset)
    cr = math.cos(roll_offset)
    sr = math.sin(roll_offset)

    r_yaw = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    r_pitch = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    r_roll = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)

    rotated = r_yaw.dot(r_pitch.dot(r_roll.dot(base_direction.astype(np.float32))))
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
    roll_offset: float,
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
                roll_offset * blend,
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


def jerk_hessian(duration: float) -> np.ndarray:
    """Return the Hessian of the integral of jerk^2 for a quintic.

    The Hessian corresponds to the quadratic form on the high-order
    coefficients ``[c3, c4, c5]`` such that ``x^T R x`` equals the
    jerk-squared integral over ``[0, duration]``.
    """

    duration = max(float(duration), 1e-3)
    T = duration
    return np.array(
        [
            [36.0 * T, 72.0 * (T**2), 120.0 * (T**3)],
            [72.0 * (T**2), 192.0 * (T**3), 360.0 * (T**4)],
            [120.0 * (T**3), 360.0 * (T**4), 720.0 * (T**5)],
        ],
        dtype=np.float32,
    )


def smoothness_penalty(
    previous_coeffs: Optional[np.ndarray],
    current_coeffs: np.ndarray,
    hessian: np.ndarray,
) -> float:
    """Compute the smoothness deviation cost between two quintics.

    The cost penalizes deviations in the high-order terms (c3..c5) to limit
    sudden changes in jerk between consecutive primitives.
    """

    if previous_coeffs is None:
        return 0.0

    current_coeffs = np.asarray(current_coeffs, dtype=np.float32)
    previous_coeffs = np.asarray(previous_coeffs, dtype=np.float32)
    hessian = np.asarray(hessian, dtype=np.float32)
    delta = current_coeffs[3:6] - previous_coeffs[3:6]

    cost = 0.0
    for axis in range(3):
        diff = delta[:, axis]
        cost += float(diff.T.dot(hessian).dot(diff))
    return cost


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


def quintic_coefficients(
    start_pos: np.ndarray,
    start_vel: np.ndarray,
    start_acc: np.ndarray,
    end_pos: np.ndarray,
    end_vel: np.ndarray,
    end_acc: np.ndarray,
    duration: float,
) -> np.ndarray:
    """Solve for quintic polynomial coefficients between two states.

    The returned array has shape ``(6, 3)`` corresponding to the coefficients
    of ``x(t) = a0 + a1 t + ... + a5 t^5`` for each spatial axis.
    """

    duration = max(float(duration), 1e-3)
    start_pos = np.asarray(start_pos, dtype=np.float32)
    start_vel = np.asarray(start_vel, dtype=np.float32)
    start_acc = np.asarray(start_acc, dtype=np.float32)
    end_pos = np.asarray(end_pos, dtype=np.float32)
    end_vel = np.asarray(end_vel, dtype=np.float32)
    end_acc = np.asarray(end_acc, dtype=np.float32)

    a0 = start_pos
    a1 = start_vel
    a2 = start_acc * 0.5

    t1 = duration
    t2 = duration * duration
    t3 = t2 * duration
    t4 = t3 * duration
    t5 = t4 * duration

    rhs0 = end_pos - (a0 + a1 * t1 + a2 * t2)
    rhs1 = end_vel - (a1 + 2.0 * a2 * t1)
    rhs2 = end_acc - (2.0 * a2)

    mat = np.array(
        [[t3, t4, t5], [3.0 * t2, 4.0 * t3, 5.0 * t4], [6.0 * t1, 12.0 * t2, 20.0 * t3]],
        dtype=np.float32,
    )
    rhs = np.stack([rhs0, rhs1, rhs2], axis=0)
    high_coeffs = np.linalg.solve(mat, rhs)

    coeffs = np.zeros((6, 3), dtype=np.float32)
    coeffs[0] = a0
    coeffs[1] = a1
    coeffs[2] = a2
    coeffs[3] = high_coeffs[0]
    coeffs[4] = high_coeffs[1]
    coeffs[5] = high_coeffs[2]
    return coeffs


def sample_quintic(
    coeffs: np.ndarray, duration: float, steps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate a quintic polynomial and its first derivative."""

    coeffs = np.asarray(coeffs, dtype=np.float32)
    steps = max(1, int(steps))
    duration = max(float(duration), 1e-3)

    times = np.linspace(0.0, duration, steps + 1, dtype=np.float32)
    points = np.zeros((steps + 1, 3), dtype=np.float32)
    velocities = np.zeros_like(points)

    for power in range(6):
        points += coeffs[power] * (times ** power)[:, None]
    for power in range(1, 6):
        velocities += power * coeffs[power] * (times ** (power - 1))[:, None]

    return points, velocities


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def calculate_yaw(
    previous_yaw: float,
    velocity_direction: np.ndarray,
    target_direction: np.ndarray,
    max_yaw_rate: float,
    dt: float,
) -> float:
    """Update yaw with a rate limit using motion context.

    The function blends the previous yaw with the current velocity direction as
    a baseline, then steers toward ``target_direction`` under the specified
    ``max_yaw_rate`` constraint.
    """

    velocity_direction = np.asarray(velocity_direction, dtype=np.float32)
    target_direction = np.asarray(target_direction, dtype=np.float32)
    dt = max(float(dt), 1e-3)
    max_yaw_rate = max(float(max_yaw_rate), 0.0)

    base_yaw = float(previous_yaw)
    speed = float(np.linalg.norm(velocity_direction))
    if speed > 1e-3:
        base_yaw = math.atan2(velocity_direction[1], velocity_direction[0])

    target_yaw = math.atan2(target_direction[1], target_direction[0])
    yaw_change = _wrap_angle(target_yaw - base_yaw)
    yaw_change = float(np.clip(yaw_change, -max_yaw_rate * dt, max_yaw_rate * dt))

    return base_yaw + yaw_change


@dataclass
class Poly5Solver:
    """Utility to generate and evaluate C²-continuous quintic trajectories."""

    duration: float

    def solve(
        self,
        start_pos: np.ndarray,
        start_vel: np.ndarray,
        start_acc: np.ndarray,
        end_pos: np.ndarray,
        end_vel: np.ndarray,
        end_acc: np.ndarray,
    ) -> np.ndarray:
        return quintic_coefficients(
            start_pos,
            start_vel,
            start_acc,
            end_pos,
            end_vel,
            end_acc,
            self.duration,
        )

    def sample(self, coeffs: np.ndarray, steps: int) -> Tuple[np.ndarray, np.ndarray]:
        return sample_quintic(coeffs, self.duration, steps)

    def jerk_hessian(self) -> np.ndarray:
        return jerk_hessian(self.duration)

    def evaluate(self, coeffs: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return position, velocity, acceleration, and jerk at time ``t``."""

        t = max(float(t), 0.0)
        pos = np.zeros(3, dtype=np.float32)
        vel = np.zeros(3, dtype=np.float32)
        acc = np.zeros(3, dtype=np.float32)
        jerk = np.zeros(3, dtype=np.float32)

        for power in range(6):
            pos += coeffs[power] * (t**power)
        for power in range(1, 6):
            vel += power * coeffs[power] * (t ** (power - 1))
        for power in range(2, 6):
            acc += power * (power - 1) * coeffs[power] * (t ** (power - 2))
        for power in range(3, 6):
            jerk += power * (power - 1) * (power - 2) * coeffs[power] * (t ** (power - 3))

        return pos, vel, acc, jerk
