#!/usr/bin/env python3
"""Automated RGB data collector with distance-based labeling."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from visualization_msgs.msg import MarkerArray


@dataclass
class Obstacle:
    x: float
    y: float
    radius: float


class DataCollector:
    def __init__(self) -> None:
        self.bridge = CvBridge()
        self.pose: Optional[PoseStamped] = None
        self.obstacles: List[Obstacle] = []

        self.max_range = rospy.get_param("~max_range", 12.0)
        self.fov_deg = rospy.get_param("~fov_deg", 120.0)
        self.near_threshold = rospy.get_param("~near_threshold", 4.0)
        self.output_dir = Path(rospy.get_param("~output_dir", str(Path.home() / "autonomy_demo" / "dataset")))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_count = 0

        rospy.Subscriber("drone/pose", PoseStamped, self.pose_callback)
        rospy.Subscriber("world/obstacles", MarkerArray, self.obstacle_callback)
        rospy.Subscriber("drone/rgb/image_raw", Image, self.image_callback, queue_size=5)

        rospy.loginfo("Data collector will store samples in %s", self.output_dir)

    def pose_callback(self, msg: PoseStamped) -> None:
        self.pose = msg

    def obstacle_callback(self, msg: MarkerArray) -> None:
        self.obstacles = [
            Obstacle(marker.pose.position.x, marker.pose.position.y, max(marker.scale.x, marker.scale.y) / 2.0)
            for marker in msg.markers
        ]

    def compute_distances(self, width: int) -> np.ndarray:
        if self.pose is None:
            return np.full(width, self.max_range, dtype=np.float32)

        fov = math.radians(self.fov_deg)
        start_angle = -fov / 2.0
        distances = np.full(width, self.max_range, dtype=np.float32)
        cx = self.pose.pose.position.x
        cy = self.pose.pose.position.y

        for u in range(width):
            angle = start_angle + (u / float(max(width - 1, 1))) * fov
            direction = (math.cos(angle), math.sin(angle))
            for obs in self.obstacles:
                dx = obs.x - cx
                dy = obs.y - cy
                b = dx * direction[0] + dy * direction[1]
                if b < 0.0:
                    continue
                closest_sq = dx * dx + dy * dy - b * b
                if closest_sq > obs.radius * obs.radius:
                    continue
                offset = math.sqrt(max(obs.radius * obs.radius - closest_sq, 0.0))
                hit = b - offset
                if 0.0 < hit < distances[u]:
                    distances[u] = hit
        return distances

    def image_callback(self, msg: Image) -> None:
        if self.pose is None or not self.obstacles:
            return
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        height, width, _ = cv_image.shape
        distances = self.compute_distances(width)
        labels = (distances < self.near_threshold).astype(np.uint8)
        label_map = np.repeat(labels[np.newaxis, :], height, axis=0)

        output_path = self.output_dir / f"sample_{self.sample_count:06d}.npz"
        np.savez_compressed(
            output_path,
            image=cv_image,
            label=label_map,
            distances=distances,
            header=self._header_to_dict(msg.header),
        )
        self.sample_count += 1
        rospy.loginfo_throttle(5.0, "Captured %d samples", self.sample_count)

    @staticmethod
    def _header_to_dict(header: Header) -> dict:
        return {
            "stamp": header.stamp.to_sec(),
            "frame_id": header.frame_id,
            "seq": header.seq,
        }


def main() -> None:
    rospy.init_node("data_collector")
    collector = DataCollector()
    rospy.spin()


if __name__ == "__main__":
    main()
