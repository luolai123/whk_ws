import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autonomy_demo.safe_navigation import PrimitiveConfig, sample_motion_primitives


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
