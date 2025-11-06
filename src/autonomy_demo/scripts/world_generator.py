#!/usr/bin/env python3
"""Random obstacle world generator for RViz visualization and occupancy data."""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

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
        self.world_size = rospy.get_param("~world_size", [20.0, 20.0, 5.0])
        self.obstacle_count = rospy.get_param("~obstacle_count", 40)
        self.height_range = rospy.get_param("~height_range", [0.5, 3.0])
        self.size_range = rospy.get_param("~size_range", [0.5, 2.5])
        self.occupancy_resolution = rospy.get_param("~occupancy_resolution", 0.5)
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


def main() -> None:
    rospy.init_node("world_generator")
    WorldGenerator()
    rospy.spin()


if __name__ == "__main__":
    main()
