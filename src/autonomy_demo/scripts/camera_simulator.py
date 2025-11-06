#!/usr/bin/env python3
"""RGB camera simulator that renders synthetic obstacle distances."""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from visualization_msgs.msg import MarkerArray


@dataclass
class Obstacle:
    x: float
    y: float
    radius: float


class CameraSimulator:
    def __init__(self) -> None:
        self.frame_id = rospy.get_param("~frame_id", "map")
        self.image_width = self._get_int_param("~width", 128)
        self.image_height = self._get_int_param("~height", 72)
        self.fov_deg = self._get_float_param("~fov_deg", 120.0)
        self.max_range = self._get_float_param("~max_range", 12.0)
        self.publish_rate = self._get_float_param("~rate", 10.0)

        self.bridge = CvBridge()
        self.latest_pose: Optional[PoseStamped] = None
        self.obstacles: List[Obstacle] = []

        self.image_pub = rospy.Publisher("drone/rgb/image_raw", Image, queue_size=1)
        self.info_pub = rospy.Publisher("drone/rgb/camera_info", CameraInfo, queue_size=1)

        rospy.Subscriber("drone/pose", PoseStamped, self.pose_callback)
        rospy.Subscriber("world/obstacles", MarkerArray, self.obstacle_callback)

        rospy.loginfo("Camera simulator ready")

    def pose_callback(self, msg: PoseStamped) -> None:
        self.latest_pose = msg

    def obstacle_callback(self, msg: MarkerArray) -> None:
        obstacles: List[Obstacle] = []
        for marker in msg.markers:
            radius = max(marker.scale.x, marker.scale.y) / 2.0
            obstacles.append(Obstacle(marker.pose.position.x, marker.pose.position.y, radius))
        self.obstacles = obstacles

    def render_image(self) -> Optional[np.ndarray]:
        if self.latest_pose is None:
            return None

        pose = self.latest_pose.pose.position
        cx = pose.x
        cy = pose.y

        width = self.image_width
        height = self.image_height
        image = np.zeros((height, width, 3), dtype=np.uint8)
        fov = math.radians(self.fov_deg)
        start_angle = -fov / 2.0

        for u in range(width):
            angle = start_angle + (u / float(width - 1)) * fov
            direction = (math.cos(angle), math.sin(angle))
            distance = self.max_range
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
                if 0.0 < hit < distance:
                    distance = hit
            intensity = int(max(0.0, 255.0 * (1.0 - (distance / self.max_range))))
            column_color = np.array([intensity, 30, 255 - intensity], dtype=np.uint8)
            image[:, u, :] = column_color
        return image

    def publish(self) -> None:
        rate = rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            image = self.render_image()
            if image is not None:
                msg = self.bridge.cv2_to_imgmsg(image, encoding="rgb8")
                msg.header.stamp = rospy.Time.now()
                msg.header.frame_id = self.child_frame_id
                info = CameraInfo()
                info.header = msg.header
                info.height = self.image_height
                info.width = self.image_width
                fx = self.image_width / (2.0 * math.tan(math.radians(self.fov_deg) / 2.0))
                fy = fx
                cx = self.image_width / 2.0
                cy = self.image_height / 2.0
                info.distortion_model = "plumb_bob"
                info.D = [0.0, 0.0, 0.0, 0.0, 0.0]
                info.K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
                info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
                info.P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
                self.image_pub.publish(msg)
                self.info_pub.publish(info)
            rate.sleep()

    @property
    def child_frame_id(self) -> str:
        return rospy.get_param("~camera_frame", "rgb_optical")

    @staticmethod
    def _maybe_parse_literal(value, name: str = ""):
        if isinstance(value, str):
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                if name:
                    rospy.logwarn("Failed to parse parameter %s as literal, using raw string", name)
                else:
                    rospy.logwarn("Failed to parse parameter value '%s' as literal", value)
        return value

    @classmethod
    def _get_float_param(cls, name: str, default: float) -> float:
        raw = rospy.get_param(name, default)
        parsed = cls._maybe_parse_literal(raw, name)
        try:
            return float(parsed)
        except (TypeError, ValueError):
            rospy.logwarn("Parameter %s could not be parsed as float, using default", name)
            return float(default)

    @classmethod
    def _get_int_param(cls, name: str, default: int) -> int:
        raw = rospy.get_param(name, default)
        parsed = cls._maybe_parse_literal(raw, name)
        try:
            return int(parsed)
        except (TypeError, ValueError):
            rospy.logwarn("Parameter %s could not be parsed as int, using default", name)
            return int(default)


def main() -> None:
    rospy.init_node("camera_simulator")
    camera = CameraSimulator()
    camera.publish()


if __name__ == "__main__":
    main()
