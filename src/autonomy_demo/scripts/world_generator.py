#!/usr/bin/env python3
"""Random obstacle world generator for RViz visualization and occupancy data."""

import ast
import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import rospy
from nav_msgs.msg import MapMetaData, OccupancyGrid
from std_srvs.srv import Trigger, TriggerResponse
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class Obstacle:
    position: Tuple[float, float, float]
    size: Tuple[float, float, float]


class WorldGenerator:
    """Generates a random set of box obstacles and publishes them for visualization."""

    def __init__(self) -> None:
        self.world_size = self._get_float_tuple("~world_size", [20.0, 20.0, 5.0], 3)
        self.obstacle_count = self._get_int("~obstacle_count", 40)
        self.height_range = self._get_float_tuple("~height_range", [0.5, 3.0], 2)
        self.size_range = self._get_float_tuple("~size_range", [0.5, 2.5], 2)
        self.occupancy_resolution = self._get_float("~occupancy_resolution", 0.5)
        self.frame_id = rospy.get_param("~frame_id", "map")

        self.marker_pub = rospy.Publisher("world/obstacles", MarkerArray, queue_size=1, latch=True)
        self.grid_pub = rospy.Publisher("world/occupancy", OccupancyGrid, queue_size=1, latch=True)

        self.obstacles: List[Obstacle] = []

        self.regenerate_srv = rospy.Service("~regenerate", Trigger, self.handle_regenerate)
        rospy.loginfo("world_generator ready - generating initial world")
        self.generate_world()
        self.publish_world()

    def handle_regenerate(self, _req: Trigger) -> TriggerResponse:
        self.generate_world()
        self.publish_world()
        return TriggerResponse(success=True, message="World regenerated")

    def generate_world(self) -> None:
        self.obstacles = []
        for _ in range(self.obstacle_count):
            x = random.uniform(-self.world_size[0] / 2.0, self.world_size[0] / 2.0)
            y = random.uniform(-self.world_size[1] / 2.0, self.world_size[1] / 2.0)
            height = random.uniform(self.height_range[0], self.height_range[1])
            sx = random.uniform(self.size_range[0], self.size_range[1])
            sy = random.uniform(self.size_range[0], self.size_range[1])
            obstacle = Obstacle(position=(x, y, height / 2.0), size=(sx, sy, height))
            self.obstacles.append(obstacle)
        rospy.loginfo("Generated %d obstacles", len(self.obstacles))

    def publish_world(self) -> None:
        markers = MarkerArray()
        timestamp = rospy.Time.now()
        for idx, obstacle in enumerate(self.obstacles):
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.header.stamp = timestamp
            marker.ns = "obstacles"
            marker.id = idx
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = obstacle.position[0]
            marker.pose.position.y = obstacle.position[1]
            marker.pose.position.z = obstacle.position[2]
            marker.pose.orientation.w = 1.0
            marker.scale.x = obstacle.size[0]
            marker.scale.y = obstacle.size[1]
            marker.scale.z = obstacle.size[2]
            marker.color.r = 0.8
            marker.color.g = 0.1
            marker.color.b = 0.1
            marker.color.a = 0.8
            markers.markers.append(marker)
        self.marker_pub.publish(markers)
        self.grid_pub.publish(self.create_occupancy_grid(timestamp))
        rospy.loginfo("Published world markers and occupancy grid")

    def create_occupancy_grid(self, timestamp: rospy.Time) -> OccupancyGrid:
        resolution = float(self.occupancy_resolution)
        width = int(math.ceil(self.world_size[0] / resolution))
        height = int(math.ceil(self.world_size[1] / resolution))
        origin_x = -self.world_size[0] / 2.0
        origin_y = -self.world_size[1] / 2.0

        grid = OccupancyGrid()
        grid.header.frame_id = self.frame_id
        grid.header.stamp = timestamp

        meta = MapMetaData()
        meta.resolution = resolution
        meta.width = width
        meta.height = height
        meta.origin.position.x = origin_x
        meta.origin.position.y = origin_y
        meta.origin.orientation.w = 1.0
        grid.info = meta

        data = [0] * (width * height)
        for obstacle in self.obstacles:
            half_x = obstacle.size[0] / 2.0
            half_y = obstacle.size[1] / 2.0
            min_x = obstacle.position[0] - half_x
            max_x = obstacle.position[0] + half_x
            min_y = obstacle.position[1] - half_y
            max_y = obstacle.position[1] + half_y

            min_ix = max(0, int((min_x - origin_x) / resolution))
            max_ix = min(width - 1, int((max_x - origin_x) / resolution))
            min_iy = max(0, int((min_y - origin_y) / resolution))
            max_iy = min(height - 1, int((max_y - origin_y) / resolution))

            for iy in range(min_iy, max_iy + 1):
                for ix in range(min_ix, max_ix + 1):
                    data[iy * width + ix] = 100
        grid.data = data
        return grid

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
    def _coerce_sequence(cls, value) -> Iterable:
        parsed = cls._maybe_parse_literal(value)
        if isinstance(parsed, (list, tuple)):
            return parsed
        if hasattr(parsed, "__iter__") and not isinstance(parsed, (str, bytes)):
            return list(parsed)
        return [parsed]

    @classmethod
    def _get_float_list(cls, name: str, default: List[float], expected_len: int) -> List[float]:
        raw = rospy.get_param(name, default)
        sequence = cls._coerce_sequence(cls._maybe_parse_literal(raw, name))
        values: List[float] = []
        for item in sequence:
            try:
                values.append(float(item))
            except (TypeError, ValueError):
                try:
                    values.append(float(str(item)))
                except (TypeError, ValueError):
                    rospy.logwarn(
                        "Parameter %s contains non-numeric entry %r, using default",
                        name,
                        item,
                    )
                    return list(default)
        if len(values) != expected_len:
            rospy.logwarn(
                "Parameter %s expected %d values but got %d, falling back to default",
                name,
                expected_len,
                len(values),
            )
            return list(default)
        return values

    @classmethod
    def _get_float_tuple(
        cls, name: str, default: List[float], expected_len: int
    ) -> Tuple[float, ...]:
        values = cls._get_float_list(name, default, expected_len)
        sanitized: List[float] = []
        for idx, item in enumerate(values):
            try:
                sanitized.append(float(item))
            except (TypeError, ValueError):
                rospy.logwarn(
                    "Parameter %s entry %d=%r could not be coerced to float, using default",
                    name,
                    idx,
                    item,
                )
                return tuple(float(v) for v in default)
        return tuple(sanitized)

    @classmethod
    def _get_float(cls, name: str, default: float) -> float:
        raw = rospy.get_param(name, default)
        parsed = cls._maybe_parse_literal(raw, name)
        try:
            return float(parsed)
        except (TypeError, ValueError):
            try:
                return float(str(parsed))
            except (TypeError, ValueError):
                rospy.logwarn(
                    "Parameter %s could not be parsed as float, using default", name
                )
            return float(default)

    @classmethod
    def _get_int(cls, name: str, default: int) -> int:
        raw = rospy.get_param(name, default)
        parsed = cls._maybe_parse_literal(raw, name)
        try:
            return int(parsed)
        except (TypeError, ValueError):
            try:
                return int(float(str(parsed)))
            except (TypeError, ValueError):
                rospy.logwarn(
                    "Parameter %s could not be parsed as int, using default", name
                )
            return int(default)


def main() -> None:
    rospy.init_node("world_generator")
    WorldGenerator()
    rospy.spin()


if __name__ == "__main__":
    main()
