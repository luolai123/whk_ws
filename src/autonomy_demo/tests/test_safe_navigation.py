import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autonomy_demo.safe_navigation import (
    PrimitiveConfig,
    Poly5Solver,
    combine_axis_poly5_trajectory,
    generate_axis_poly5_solvers,
    sample_motion_primitives,
)


def test_zero_noise_primitives_align_with_body_forward():
    base_direction_camera = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    camera_to_body = np.eye(3, dtype=np.float32)
    config = PrimitiveConfig(
        yaw_range_deg=0.0,
        pitch_std_deg=0.0,
        roll_std_deg=0.0,
        forward_log_sigma=0.0,
        v_std_unit=0.0,
        a_std_unit=0.0,
    )
    rng = np.random.default_rng(0)

    samples = sample_motion_primitives(
        base_direction_camera,
        camera_to_body,
        rng,
        config,
        count=5,
    )

    expected_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    for sample in samples:
        np.testing.assert_allclose(sample.base_direction_camera, expected_dir)
        np.testing.assert_allclose(sample.goal_direction_body, expected_dir)
        assert math.isclose(sample.yaw_offset, 0.0)
        assert math.isclose(sample.pitch_offset, 0.0)
        assert math.isclose(sample.roll_offset, 0.0)


def test_axis_poly5_solvers_recombine_to_full_trajectory():
    duration = 2.5
    start_pos = np.array([0.5, -0.2, 1.0], dtype=np.float32)
    start_vel = np.array([0.1, 0.05, -0.1], dtype=np.float32)
    start_acc = np.array([0.0, 0.0, 0.05], dtype=np.float32)
    end_pos = np.array([2.0, 0.8, 1.6], dtype=np.float32)
    end_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    end_acc = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    full_solver = Poly5Solver(duration)
    full_coeffs = full_solver.solve(start_pos, start_vel, start_acc, end_pos, end_vel, end_acc)

    axis_solvers = generate_axis_poly5_solvers(
        start_pos, start_vel, start_acc, end_pos, end_vel, end_acc, duration
    )

    positions, velocities, accelerations, jerks, traj_duration = combine_axis_poly5_trajectory(
        axis_solvers, steps=10
    )

    np.testing.assert_allclose(traj_duration, duration)
    np.testing.assert_allclose(positions[0], start_pos)
    np.testing.assert_allclose(velocities[0], start_vel)
    np.testing.assert_allclose(accelerations[0], start_acc)

    np.testing.assert_allclose(positions[-1], end_pos, atol=1e-4)
    np.testing.assert_allclose(velocities[-1], end_vel, atol=1e-4)
    np.testing.assert_allclose(accelerations[-1], end_acc, atol=1e-4)

    midpoint_idx = positions.shape[0] // 2
    mid_time = duration * 0.5
    expected_state = full_solver.evaluate(full_coeffs, mid_time)
    np.testing.assert_allclose(positions[midpoint_idx], expected_state[0], atol=1e-4)
    np.testing.assert_allclose(velocities[midpoint_idx], expected_state[1], atol=1e-4)
    np.testing.assert_allclose(accelerations[midpoint_idx], expected_state[2], atol=1e-4)

    for axis, (_, coeffs) in axis_solvers.items():
        axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
        np.testing.assert_allclose(coeffs, full_coeffs[:, axis_idx])

    assert np.all(np.isfinite(jerks))
