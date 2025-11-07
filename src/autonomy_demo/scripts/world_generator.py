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
    shape: str
    category: str = "obstacle"


class WorldGenerator:
    """Generates a random set of box and sphere obstacles for visualization."""

    def __init__(self) -> None:
        self.world_size = self._ensure_float_tuple(
            self._get_float_tuple("~world_size", [120.0, 120.0, 12.0], 3),
            "~world_size",
            fallback=(120.0, 120.0, 12.0),
        )
        self.obstacle_count = self._get_int("~obstacle_count", 70)
        self.obstacle_density = max(0.0, self._get_float("~obstacle_density", 0.0))
        self.height_range = self._ensure_float_tuple(
            self._get_float_tuple("~height_range", [1.2, 7.5], 2),
            "~height_range",
            fallback=(1.2, 7.5),
        )
        self.size_range = self._ensure_float_tuple(
            self._get_float_tuple("~size_range", [1.0, 4.5], 2),
            "~size_range",
            fallback=(1.0, 4.5),
        )
        self.occupancy_resolution = self._get_float("~occupancy_resolution", 0.5)
        self.frame_id = rospy.get_param("~frame_id", "map")
        self.sphere_ratio = max(0.0, min(1.0, self._get_float("~sphere_ratio", 0.3)))
        raw_gate_ratio = self._get_float("~gate_ratio", 0.2)
        self.gate_ratio = max(0.0, min(1.0 - self.sphere_ratio, raw_gate_ratio))
        self.gate_opening_range = self._ensure_float_tuple(
            self._get_float_tuple("~gate_opening_range", [3.0, 6.5], 2),
            "~gate_opening_range",
            fallback=(3.0, 6.5),
        )
        self.gate_height_range = self._ensure_float_tuple(
            self._get_float_tuple("~gate_height_range", [4.0, 8.5], 2),
            "~gate_height_range",
            fallback=(4.0, 8.5),
        )
        self.gate_post_thickness_range = self._ensure_float_tuple(
            self._get_float_tuple("~gate_post_thickness_range", [0.4, 0.9], 2),
            "~gate_post_thickness_range",
            fallback=(0.4, 0.9),
        )
        self.gate_depth_range = self._ensure_float_tuple(
            self._get_float_tuple("~gate_depth_range", [0.7, 1.4], 2),
            "~gate_depth_range",
            fallback=(0.7, 1.4),
        )
        self.gate_top_thickness_range = self._ensure_float_tuple(
            self._get_float_tuple("~gate_top_thickness_range", [0.3, 0.6], 2),
            "~gate_top_thickness_range",
            fallback=(0.3, 0.6),
        )
        self.obstacle_color = (0.55, 0.55, 0.55, 0.9)

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
        obstacle_total = self._compute_obstacle_total()
        count = 0
        while count < obstacle_total:
            x = random.uniform(-self.world_size[0] / 2.0, self.world_size[0] / 2.0)
            y = random.uniform(-self.world_size[1] / 2.0, self.world_size[1] / 2.0)
            roll = random.random()
            if roll < self.sphere_ratio:
                diameter = random.uniform(self.size_range[0], self.size_range[1])
                radius = min(diameter / 2.0, self.world_size[2] / 2.0)
                obstacle = Obstacle(
                    position=(x, y, radius),
                    size=(radius * 2.0, radius * 2.0, radius * 2.0),
                    shape="sphere",
                )
                self.obstacles.append(obstacle)
            elif roll < self.sphere_ratio + self.gate_ratio:
                gate_parts = self._create_gate_obstacles(x, y)
                if gate_parts:
                    self.obstacles.extend(gate_parts)
                else:
                    self.obstacles.append(self._create_box_obstacle(x, y))
            else:
                self.obstacles.append(self._create_box_obstacle(x, y))
            count += 1
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
            marker.type = Marker.SPHERE if obstacle.shape == "sphere" else Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = obstacle.position[0]
            marker.pose.position.y = obstacle.position[1]
            marker.pose.position.z = obstacle.position[2]
            marker.pose.orientation.w = 1.0
            marker.scale.x = obstacle.size[0]
            marker.scale.y = obstacle.size[1]
            marker.scale.z = obstacle.size[2]
            marker.color.r = self.obstacle_color[0]
            marker.color.g = self.obstacle_color[1]
            marker.color.b = self.obstacle_color[2]
            marker.color.a = self.obstacle_color[3]
            markers.markers.append(marker)
        self.marker_pub.publish(markers)
        self.grid_pub.publish(self.create_occupancy_grid(timestamp))
        rospy.loginfo("Published world markers and occupancy grid")

    def _create_box_obstacle(self, x: float, y: float) -> Obstacle:
        height = random.uniform(self.height_range[0], self.height_range[1])
        max_height = self.world_size[2] * 0.95
        height = min(height, max_height)
        sx = random.uniform(self.size_range[0], self.size_range[1])
        sy = random.uniform(self.size_range[0], self.size_range[1])
        return Obstacle(
            position=(x, y, height / 2.0),
            size=(sx, sy, height),
            shape="box",
        )

    def _create_gate_obstacles(self, x: float, y: float) -> List[Obstacle]:
        half_world_x = self.world_size[0] / 2.0
        half_world_y = self.world_size[1] / 2.0
        max_height = self.world_size[2] * 0.95
        gate_height = random.uniform(self.gate_height_range[0], self.gate_height_range[1])
        gate_height = min(gate_height, max_height)
        opening = random.uniform(self.gate_opening_range[0], self.gate_opening_range[1])
        post_thickness = random.uniform(
            self.gate_post_thickness_range[0], self.gate_post_thickness_range[1]
        )
        depth = random.uniform(self.gate_depth_range[0], self.gate_depth_range[1])
        top_thickness = random.uniform(
            self.gate_top_thickness_range[0], self.gate_top_thickness_range[1]
        )
        top_thickness = min(top_thickness, max(gate_height * 0.3, self.gate_top_thickness_range[0]))
        beam_height = gate_height - top_thickness / 2.0
        if beam_height <= 0.0:
            return []
        left_center_x = x - (opening / 2.0 + post_thickness / 2.0)
        right_center_x = x + (opening / 2.0 + post_thickness / 2.0)
        if (
            left_center_x - post_thickness / 2.0 < -half_world_x
            or right_center_x + post_thickness / 2.0 > half_world_x
            or y - depth / 2.0 < -half_world_y
            or y + depth / 2.0 > half_world_y
        ):
            return []
        posts_z = gate_height / 2.0
        left_post = Obstacle(
            position=(left_center_x, y, posts_z),
            size=(post_thickness, depth, gate_height),
            shape="box",
            category="gate",
        )
        right_post = Obstacle(
            position=(right_center_x, y, posts_z),
            size=(post_thickness, depth, gate_height),
            shape="box",
            category="gate",
        )
        beam = Obstacle(
            position=(x, y, beam_height),
            size=(opening + post_thickness * 2.0, depth, top_thickness),
            shape="box",
            category="gate",
        )
        return [left_post, right_post, beam]

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
            if obstacle.shape == "sphere":
                radius = obstacle.size[0] / 2.0
                min_x = obstacle.position[0] - radius
                max_x = obstacle.position[0] + radius
                min_y = obstacle.position[1] - radius
                max_y = obstacle.position[1] + radius
            else:
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
                    if obstacle.shape == "sphere":
                        cell_x = origin_x + (ix + 0.5) * resolution
                        cell_y = origin_y + (iy + 0.5) * resolution
                        if (cell_x - obstacle.position[0]) ** 2 + (
                            cell_y - obstacle.position[1]
                        ) ** 2 > radius ** 2:
                            continue
                    data[iy * width + ix] = 100
        grid.data = data
        return grid

    def _compute_obstacle_total(self) -> int:
        if self.obstacle_density > 0.0:
            area = self.world_size[0] * self.world_size[1]
            total = int(area * self.obstacle_density)
            return max(total, 1)
        return max(self.obstacle_count, 1)

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

    @staticmethod
    def _ensure_float_tuple(
        values: Iterable, name: str, fallback: Iterable
    ) -> Tuple[float, ...]:
        coerced: List[float] = []
        for idx, item in enumerate(values):
            try:
                coerced.append(float(item))
            except (TypeError, ValueError):
                rospy.logwarn(
                    "Parameter %s entry %d=%r could not be coerced to float, using fallback",
                    name,
                    idx,
                    item,
                )
                return tuple(float(v) for v in fallback)
        return tuple(coerced)

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
