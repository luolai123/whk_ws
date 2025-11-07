#!/usr/bin/env python3
"""Simple kinematic UAV simulator that follows RViz goals."""

import math
from dataclasses import dataclass
from typing import Optional

import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf_conversions import transformations
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class DroneState:
    position: list
    velocity: list
    orientation: tuple
    yaw: float
    pitch: float


class DroneSimulator:
    def __init__(self) -> None:
        self.frame_id = rospy.get_param("~frame_id", "map")
        self.child_frame_id = rospy.get_param("~child_frame_id", "base_link")
        self.update_rate = rospy.get_param("~update_rate", 30.0)
        self.max_speed = rospy.get_param("~max_speed", 2.0)
        self.goal_tolerance = rospy.get_param("~goal_tolerance", 0.1)
        self.default_altitude = rospy.get_param("~altitude", 1.5)

        self.state = DroneState(
            position=[0.0, 0.0, self.default_altitude],
            velocity=[0.0, 0.0, 0.0],
            orientation=(0.0, 0.0, 0.0, 1.0),
            yaw=0.0,
            pitch=0.0,
        )
        self.goal: Optional[list] = None

        self.pose_pub = rospy.Publisher("drone/pose", PoseStamped, queue_size=1)
        self.odom_pub = rospy.Publisher("drone/odometry", Odometry, queue_size=1)
        self.goal_sub = rospy.Subscriber("move_base_simple/goal", PoseStamped, self.goal_callback)
        self.marker_pub = rospy.Publisher("drone/visualization", MarkerArray, queue_size=1)

        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        rospy.loginfo("Drone simulator initialized")

    def goal_callback(self, msg: PoseStamped) -> None:
        self.goal = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z or self.default_altitude]
        if self.goal[2] <= 0.1:
            self.goal[2] = self.default_altitude
        rospy.loginfo("Received new goal: %s", self.goal)

    def step(self, dt: float) -> None:
        if self.goal is not None:
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
        else:
            self.state.velocity = [0.0, 0.0, 0.0]

        for i in range(3):
            self.state.position[i] += self.state.velocity[i] * dt

        self.update_orientation()

    def update_orientation(self) -> None:
        speed = math.sqrt(sum(v * v for v in self.state.velocity))
        if speed > 1e-4:
            horizontal = math.hypot(self.state.velocity[0], self.state.velocity[1])
            if horizontal > 1e-4:
                self.state.yaw = math.atan2(self.state.velocity[1], self.state.velocity[0])
            # Pitch the drone in the direction of travel; negative pitch means nose down when moving forward.
            self.state.pitch = math.atan2(-self.state.velocity[2], max(horizontal, 1e-4))
        quat = transformations.quaternion_from_euler(0.0, self.state.pitch, self.state.yaw)
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

        body = make_marker(0, Marker.CYLINDER)
        body.scale.x = 0.24
        body.scale.y = 0.24
        body.scale.z = 0.08
        body.color.r = 0.1
        body.color.g = 0.4
        body.color.b = 0.9
        markers.markers.append(body)

        arm_x = make_marker(1, Marker.CUBE)
        arm_x.scale.x = 0.4
        arm_x.scale.y = 0.06
        arm_x.scale.z = 0.03
        arm_x.color.r = 0.15
        arm_x.color.g = 0.15
        arm_x.color.b = 0.15
        markers.markers.append(arm_x)

        arm_y = make_marker(2, Marker.CUBE)
        arm_y.scale.x = 0.06
        arm_y.scale.y = 0.4
        arm_y.scale.z = 0.03
        arm_y.color.r = 0.15
        arm_y.color.g = 0.15
        arm_y.color.b = 0.15
        markers.markers.append(arm_y)

        rotor_offsets = [
            (0.2, 0.2, 0.0),
            (0.2, -0.2, 0.0),
            (-0.2, 0.2, 0.0),
            (-0.2, -0.2, 0.0),
        ]
        for idx, (rx, ry, rz) in enumerate(rotor_offsets, start=3):
            rotor = make_marker(idx, Marker.CYLINDER)
            rotor.pose.position.x = rx
            rotor.pose.position.y = ry
            rotor.pose.position.z = rz
            rotor.scale.x = 0.12
            rotor.scale.y = 0.12
            rotor.scale.z = 0.02
            rotor.color.r = 0.8
            rotor.color.g = 0.8
            rotor.color.b = 0.8
            markers.markers.append(rotor)

        heading = make_marker(7, Marker.ARROW)
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
