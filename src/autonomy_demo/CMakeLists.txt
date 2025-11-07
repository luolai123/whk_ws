cmake_minimum_required(VERSION 3.0.2)
project(autonomy_demo)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  nav_msgs
  visualization_msgs
  message_generation
  cv_bridge
  image_transport
  tf2_ros
  tf_conversions
)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
  scripts/world_generator.py
  scripts/drone_simulator.py
  scripts/camera_simulator.py
  scripts/data_collector.py
  scripts/inference_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
