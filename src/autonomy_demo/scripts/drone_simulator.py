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


@dataclass
class DroneState:
    position: list
    velocity: list


class DroneSimulator:
    def __init__(self) -> None:
        self.frame_id = rospy.get_param("~frame_id", "map")
        self.child_frame_id = rospy.get_param("~child_frame_id", "base_link")
        self.update_rate = rospy.get_param("~update_rate", 30.0)
        self.max_speed = rospy.get_param("~max_speed", 2.0)
        self.goal_tolerance = rospy.get_param("~goal_tolerance", 0.1)
        self.default_altitude = rospy.get_param("~altitude", 1.5)

        self.state = DroneState(position=[0.0, 0.0, self.default_altitude], velocity=[0.0, 0.0, 0.0])
        self.goal: Optional[list] = None

        self.pose_pub = rospy.Publisher("drone/pose", PoseStamped, queue_size=1)
        self.odom_pub = rospy.Publisher("drone/odometry", Odometry, queue_size=1)
        self.goal_sub = rospy.Subscriber("move_base_simple/goal", PoseStamped, self.goal_callback)

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

    def publish_state(self) -> None:
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = self.frame_id
        pose_msg.pose.position.x = self.state.position[0]
        pose_msg.pose.position.y = self.state.position[1]
        pose_msg.pose.position.z = self.state.position[2]
        pose_msg.pose.orientation.w = 1.0
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
        quat = transformations.quaternion_from_euler(0.0, 0.0, 0.0)
        transform.transform.rotation.x = quat[0]
        transform.transform.rotation.y = quat[1]
        transform.transform.rotation.z = quat[2]
        transform.transform.rotation.w = quat[3]
        self.tf_broadcaster.sendTransform(transform)

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
