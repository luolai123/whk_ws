#!/usr/bin/env python3
"""Simple kinematic UAV simulator that follows RViz goals."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf_conversions import transformations
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class DroneState:
    position: list
    velocity: list
    orientation: tuple
    yaw: float
    pitch: float
    roll: float = 0.0


class DroneSimulator:
    def __init__(self) -> None:
        self.frame_id = rospy.get_param("~frame_id", "map")
        self.child_frame_id = rospy.get_param("~child_frame_id", "base_link")
        self.update_rate = rospy.get_param("~update_rate", 30.0)
        self.max_speed = rospy.get_param("~max_speed", 2.0)
        self.goal_tolerance = rospy.get_param("~goal_tolerance", 0.1)
        self.default_altitude = rospy.get_param("~altitude", 1.5)
        self.primitive_dt = rospy.get_param("~primitive_dt", 0.25)
        self.primitive_steps = rospy.get_param("~primitive_steps", 4)
        self.attitude_gain = rospy.get_param("~attitude_gain", 4.0)
        self.max_yaw_rate = math.radians(rospy.get_param("~max_yaw_rate_deg", 90.0))
        self.max_pitch_rate = math.radians(rospy.get_param("~max_pitch_rate_deg", 60.0))
        self.mesh_resource = rospy.get_param(
            "~mesh_resource", "package://autonomy_demo/config/uav.dae"
        )
        self.mesh_scale = rospy.get_param("~mesh_scale", 1.0)

        self.state = DroneState(
            position=[0.0, 0.0, self.default_altitude],
            velocity=[0.0, 0.0, 0.0],
            orientation=(0.0, 0.0, 0.0, 1.0),
            yaw=0.0,
            pitch=0.0,
        )
        self.goal: Optional[list] = None
        self.path_points: List[List[float]] = []
        self.path_orientations: List[Tuple[float, float]] = []
        self.path_index = 0
        self.follow_path = False
        self.segment_time_remaining = 0.0
        self.desired_yaw = 0.0
        self.desired_pitch = 0.0
        self.total_duration_hint = max(self.primitive_dt, self.primitive_dt * self.primitive_steps)
        self.segment_duration_hint = max(self.primitive_dt, self.total_duration_hint / max(self.primitive_steps, 1))

        self.pose_pub = rospy.Publisher("drone/pose", PoseStamped, queue_size=1)
        self.odom_pub = rospy.Publisher("drone/odometry", Odometry, queue_size=1)
        self.goal_sub = rospy.Subscriber("move_base_simple/goal", PoseStamped, self.goal_callback)
        self.marker_pub = rospy.Publisher("drone/visualization", MarkerArray, queue_size=1)
        self.path_sub = rospy.Subscriber("drone/safe_trajectory", Path, self.trajectory_callback, queue_size=1)
        self.offset_sub = rospy.Subscriber("drone/movement_offsets", Float32MultiArray, self.offset_callback)

        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        rospy.loginfo("Drone simulator initialized")

    def goal_callback(self, msg: PoseStamped) -> None:
        self.goal = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z or self.default_altitude]
        if self.goal[2] <= 0.1:
            self.goal[2] = self.default_altitude
        rospy.loginfo("Received new goal: %s", self.goal)
        self.follow_path = False
        self.path_points = []
        self.path_index = 0

    def trajectory_callback(self, msg: Path) -> None:
        if not msg.poses:
            return
        points: List[List[float]] = []
        orientations: List[Tuple[float, float]] = []
        for pose in msg.poses:
            points.append(
                [
                    pose.pose.position.x,
                    pose.pose.position.y,
                    pose.pose.position.z if pose.pose.position.z > 0.0 else self.default_altitude,
                ]
            )
            quat = (
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w,
            )
            roll, pitch, yaw = transformations.euler_from_quaternion(quat)
            orientations.append((pitch, yaw))
        self.path_points = points
        self.path_orientations = orientations
        self.path_index = 1 if len(points) > 1 else 0
        if self.path_points:
            current = self.state.position
            best_idx = self.path_index
            best_dist = float("inf")
            for idx, point in enumerate(self.path_points[1:], start=1):
                dist = math.sqrt(
                    (point[0] - current[0]) ** 2
                    + (point[1] - current[1]) ** 2
                    + (point[2] - current[2]) ** 2
                )
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            self.path_index = best_idx
        self.follow_path = len(self.path_points) > 1
        if self.path_points:
            self.goal = self.path_points[-1][:]
            self._update_segment_duration()
            if self.path_orientations:
                idx = min(self.path_index, len(self.path_orientations) - 1)
                pitch, yaw = self.path_orientations[idx]
                self.desired_pitch = pitch
                self.desired_yaw = yaw

    def offset_callback(self, msg: Float32MultiArray) -> None:
        if not msg.data:
            return
        if len(msg.data) >= 4 and msg.data[3] > 0.0:
            self.total_duration_hint = float(msg.data[3])
        elif len(msg.data) >= 3 and msg.data[2] > 0.0:
            self.total_duration_hint = max(
                self.primitive_dt,
                self.primitive_dt * self.primitive_steps * float(msg.data[2]),
            )
        else:
            self.total_duration_hint = max(self.primitive_dt, self.primitive_dt * self.primitive_steps)
        self._update_segment_duration()

    def _update_segment_duration(self) -> None:
        segments = max(1, len(self.path_points) - 1)
        self.segment_duration_hint = max(
            self.primitive_dt * 0.5, self.total_duration_hint / max(segments, 1)
        )
        self.segment_time_remaining = self.segment_duration_hint

    def step(self, dt: float) -> None:
        if self.follow_path and self.path_points and self.path_index < len(self.path_points):
            self._follow_primitive(dt)
        elif self.goal is not None:
            direction = [self.goal[0] - self.state.position[0],
                         self.goal[1] - self.state.position[1],
                         self.goal[2] - self.state.position[2]]
            distance = math.sqrt(sum(d * d for d in direction))
            if distance < self.goal_tolerance:
                self.goal = None
                self.state.velocity = [0.0, 0.0, 0.0]
            else:
                direction = [d / distance for d in direction]
                speed = min(self.max_speed, distance / max(dt, 1e-3))
                self.state.velocity = [direction[0] * speed,
                                       direction[1] * speed,
                                       direction[2] * speed]
            if distance > 1e-6:
                self.desired_yaw = math.atan2(direction[1], direction[0])
                horizontal = math.hypot(direction[0], direction[1])
                self.desired_pitch = math.atan2(-direction[2], max(horizontal, 1e-4))
        else:
            self.state.velocity = [0.0, 0.0, 0.0]

        for i in range(3):
            self.state.position[i] += self.state.velocity[i] * dt

        self._track_attitude(dt)

    def _follow_primitive(self, dt: float) -> None:
        target = self.path_points[self.path_index]
        direction = [
            target[0] - self.state.position[0],
            target[1] - self.state.position[1],
            target[2] - self.state.position[2],
        ]
        distance = math.sqrt(sum(d * d for d in direction))
        if distance < self.goal_tolerance or self.segment_time_remaining <= 0.0:
            self.path_index += 1
            if self.path_index >= len(self.path_points):
                self.follow_path = False
                self.goal = None
                self.state.velocity = [0.0, 0.0, 0.0]
                return
            target = self.path_points[self.path_index]
            direction = [
                target[0] - self.state.position[0],
                target[1] - self.state.position[1],
                target[2] - self.state.position[2],
            ]
            distance = math.sqrt(sum(d * d for d in direction))
            self.segment_time_remaining = self.segment_duration_hint

        if distance < 1e-6:
            self.state.velocity = [0.0, 0.0, 0.0]
            return

        direction = [d / distance for d in direction]
        if self.segment_time_remaining > 1e-6:
            target_speed = min(self.max_speed, distance / self.segment_time_remaining)
        else:
            target_speed = min(self.max_speed, distance / max(dt, 1e-3))
        self.state.velocity = [direction[0] * target_speed, direction[1] * target_speed, direction[2] * target_speed]
        self.segment_time_remaining = max(0.0, self.segment_time_remaining - dt)

        if self.path_orientations and self.path_index < len(self.path_orientations):
            pitch, yaw = self.path_orientations[self.path_index]
            self.desired_pitch = pitch
            self.desired_yaw = yaw

    def _track_attitude(self, dt: float) -> None:
        if not self.follow_path:
            speed = math.sqrt(sum(v * v for v in self.state.velocity))
            if speed > 1e-4:
                horizontal = math.hypot(self.state.velocity[0], self.state.velocity[1])
                if horizontal > 1e-4:
                    self.desired_yaw = math.atan2(self.state.velocity[1], self.state.velocity[0])
                self.desired_pitch = math.atan2(-self.state.velocity[2], max(horizontal, 1e-4))

        yaw_error = math.atan2(math.sin(self.desired_yaw - self.state.yaw), math.cos(self.desired_yaw - self.state.yaw))
        pitch_error = self.desired_pitch - self.state.pitch

        yaw_rate = max(-self.max_yaw_rate, min(self.max_yaw_rate, self.attitude_gain * yaw_error))
        pitch_rate = max(-self.max_pitch_rate, min(self.max_pitch_rate, self.attitude_gain * pitch_error))

        self.state.yaw += yaw_rate * dt
        self.state.pitch += pitch_rate * dt

        quat = transformations.quaternion_from_euler(self.state.roll, self.state.pitch, self.state.yaw)
        self.state.orientation = tuple(quat)

    def publish_state(self) -> None:
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = self.frame_id
        pose_msg.pose.position.x = self.state.position[0]
        pose_msg.pose.position.y = self.state.position[1]
        pose_msg.pose.position.z = self.state.position[2]
        pose_msg.pose.orientation.x = self.state.orientation[0]
        pose_msg.pose.orientation.y = self.state.orientation[1]
        pose_msg.pose.orientation.z = self.state.orientation[2]
        pose_msg.pose.orientation.w = self.state.orientation[3]
        self.pose_pub.publish(pose_msg)

        odom_msg = Odometry()
        odom_msg.header = pose_msg.header
        odom_msg.child_frame_id = self.child_frame_id
        odom_msg.pose.pose = pose_msg.pose
        odom_msg.twist.twist.linear.x = self.state.velocity[0]
        odom_msg.twist.twist.linear.y = self.state.velocity[1]
        odom_msg.twist.twist.linear.z = self.state.velocity[2]
        self.odom_pub.publish(odom_msg)

        transform = TransformStamped()
        transform.header = pose_msg.header
        transform.child_frame_id = self.child_frame_id
        transform.transform.translation.x = self.state.position[0]
        transform.transform.translation.y = self.state.position[1]
        transform.transform.translation.z = self.state.position[2]
        transform.transform.rotation.x = self.state.orientation[0]
        transform.transform.rotation.y = self.state.orientation[1]
        transform.transform.rotation.z = self.state.orientation[2]
        transform.transform.rotation.w = self.state.orientation[3]
        self.tf_broadcaster.sendTransform(transform)
        self.publish_markers(pose_msg.header.stamp)

    def publish_markers(self, stamp: rospy.Time) -> None:
        markers = MarkerArray()
        namespace = "drone"

        def make_marker(marker_id: int, marker_type: int) -> Marker:
            marker = Marker()
            marker.header.frame_id = self.child_frame_id
            marker.header.stamp = stamp
            marker.ns = namespace
            marker.id = marker_id
            marker.type = marker_type
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.color.a = 1.0
            return marker

        if self.mesh_resource:
            mesh_marker = make_marker(0, Marker.MESH_RESOURCE)
            mesh_marker.mesh_resource = self.mesh_resource
            mesh_marker.mesh_use_embedded_materials = True
            mesh_marker.scale.x = self.mesh_scale
            mesh_marker.scale.y = self.mesh_scale
            mesh_marker.scale.z = self.mesh_scale
            mesh_marker.color.r = 0.6
            mesh_marker.color.g = 0.6
            mesh_marker.color.b = 0.6
            markers.markers.append(mesh_marker)
        else:
            fallback = make_marker(0, Marker.CYLINDER)
            fallback.scale.x = 0.24
            fallback.scale.y = 0.24
            fallback.scale.z = 0.08
            fallback.color.r = 0.2
            fallback.color.g = 0.2
            fallback.color.b = 0.2
            markers.markers.append(fallback)

        heading = make_marker(10, Marker.ARROW)
        heading.pose.position.z = 0.05
        heading.scale.x = 0.4
        heading.scale.y = 0.05
        heading.scale.z = 0.05
        heading.color.r = 0.9
        heading.color.g = 0.2
        heading.color.b = 0.2
        markers.markers.append(heading)

        self.marker_pub.publish(markers)

    def run(self) -> None:
        rate = rospy.Rate(self.update_rate)
        last_time = rospy.Time.now()
        while not rospy.is_shutdown():
            current_time = rospy.Time.now()
            dt = (current_time - last_time).to_sec()
            if dt <= 0.0:
                dt = 1.0 / self.update_rate
            self.step(dt)
            self.publish_state()
            last_time = current_time
            rate.sleep()


def main() -> None:
    rospy.init_node("drone_simulator")
    sim = DroneSimulator()
    sim.run()


if __name__ == "__main__":
    main()
