"""Utility modules for the autonomy_demo package."""

from .camera_utils import camera_mount_from_pitch, parse_camera_offset, precompute_unit_rays
from .obstacle_field import ObstacleField
from .safe_navigation import (
    SafeRegion,
    clamp_normalized,
    compute_direction_from_pixel,
    cubic_hermite_path,
    find_largest_safe_region,
    is_pixel_safe,
    jerk_metrics,
    jerk_score,
    orientation_rate_score,
    orientation_rate_stats,
    path_smoothness,
    primitive_state_dim,
    project_direction_to_pixel,
    rotate_direction,
    sample_yopo_directions,
    quintic_coefficients,
    sample_quintic,
)

__all__ = [
    "ObstacleField",
    "camera_mount_from_pitch",
    "parse_camera_offset",
    "precompute_unit_rays",
    "SafeRegion",
    "clamp_normalized",
    "compute_direction_from_pixel",
    "cubic_hermite_path",
    "find_largest_safe_region",
    "is_pixel_safe",
    "jerk_metrics",
    "jerk_score",
    "orientation_rate_score",
    "orientation_rate_stats",
    "path_smoothness",
    "primitive_state_dim",
    "project_direction_to_pixel",
    "rotate_direction",
    "sample_yopo_directions",
    "quintic_coefficients",
    "sample_quintic",
]
