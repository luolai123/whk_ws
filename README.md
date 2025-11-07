# Autonomy Demo Workspace

This workspace contains a complete ROS1 (Noetic) simulation for a UAV navigating a randomly generated obstacle field with an RGB camera, automated dataset collection, PyTorch-based training, and an inference pipeline that can be driven from RViz.

## Features

- **Random obstacle world** featuring a mix of colorful boxes and spheres published as `visualization_msgs/MarkerArray` and `nav_msgs/OccupancyGrid`, ready for RViz visualization and programmatic access.
- **Configurable environment scale** with tunable obstacle density, size ranges, and shape ratios.
- **Kinematic UAV simulator** that responds to RViz 2D Nav Goal inputs and publishes pose/odometry data.
- **Synthetic RGB camera** that renders obstacle distances into color-coded imagery.
- **Automated data collection** pipeline that records RGB frames with near/far obstacle labels using analytic distance computation.
- **PyTorch training utilities** for a two-class distance classifier and an inference node that streams classification maps in real time.

## Prerequisites

- Ubuntu 20.04 with ROS Noetic.
- Python dependencies (install via pip):

  ```bash
  pip install numpy torch torchvision pillow
  ```

  `cv_bridge` and `image_transport` are provided by ROS Noetic desktop-full installations.

## Build Instructions

1. Source your ROS environment:

   ```bash
   source /opt/ros/noetic/setup.bash
   ```

2. Build the workspace:

   ```bash
   cd /path/to/whk_ws
   catkin_make
   source devel/setup.bash
   ```

## Simulation and Visualization

Launch the core simulation (world generator, UAV, RGB camera, and RViz):

```bash
roslaunch autonomy_demo sim.launch
```

- RViz loads with the preconfigured view (`config/world.rviz`).
- Use the **2D Nav Goal** tool in RViz to command the drone. The `drone_simulator` node will move toward the clicked goal at a fixed altitude.
- Obstacle markers and the occupancy grid appear in RViz under the `/world/obstacles` and `/world/occupancy` topics.
- The synthetic RGB stream is visible on `/drone/rgb/image_raw`.

To regenerate obstacles at runtime:

```bash
rosservice call /world_generator/regenerate
```

### Customizing the World Layout

- Increase the simulated airspace or avoid seeing beyond the borders by raising `world_size` (default `40 x 40 x 8` meters).
- Control crowding with `obstacle_density` (obstacles per square meter). When the density is zero the system falls back to the legacy `obstacle_count` parameter.
- Mix shape variety with `sphere_ratio` (0.0–1.0) to choose how many obstacles are rendered as spheres versus boxes.
- Adjust `size_range` and `height_range` to refine individual obstacle proportions. All parameters can be overridden in `sim.launch` or via ROS command-line arguments when launching.

## Automated Dataset Collection

Start the simulator together with the data collector:

```bash
roslaunch autonomy_demo data_collection.launch output_dir:=/your/dataset/path
```

- Samples are stored as compressed `.npz` files containing the RGB image, the 2-class label map (near obstacle vs. safe), per-column distance estimates, and the ROS header metadata.
- The default near/far threshold is 4 meters; override with the `near_threshold` parameter if required.

> Tip: move the UAV around using RViz goals while data collection is running to diversify the dataset. You can also call the regenerate service to change obstacle layouts between runs.

## Training the Distance Classifier

After gathering data, train the neural network:

```bash
python3 src/autonomy_demo/training/train_classifier.py \
  /your/dataset/path \
  --epochs 15 \
  --batch 8 \
  --lr 5e-4 \
  --output ~/autonomy_demo/model.pt
```

- The script automatically creates a validation split and saves the trained weights.
- Adjust hyperparameters as needed. GPU acceleration is used automatically when CUDA is available.

## Real-Time Inference

With a trained model saved to `~/autonomy_demo/model.pt`, launch the inference stack:

```bash
roslaunch autonomy_demo inference.launch model_path:=/absolute/path/to/model.pt
```

- The `distance_inference` node subscribes to the RGB feed, performs the two-class segmentation, and publishes a color-coded result on `/drone/rgb/distance_class` (green = safe, red = near obstacle).
- RViz displays both the raw camera stream and the classification overlay. Continue using the 2D Nav Goal tool to steer the drone while observing the live predictions.

## File Overview

```
src/autonomy_demo/
├── config/world.rviz           # RViz configuration
├── launch/
│   ├── sim.launch              # Core simulation stack
│   ├── data_collection.launch  # Simulation + automated data capture
│   └── inference.launch        # Simulation + trained model inference
├── package.xml
├── CMakeLists.txt
├── scripts/
│   ├── world_generator.py      # Random obstacle publisher + occupancy grid
│   ├── drone_simulator.py      # Goal-driven UAV kinematics + TF
│   ├── camera_simulator.py     # Synthetic RGB distance camera
│   ├── data_collector.py       # Dataset recording with distance-based labels
│   └── inference_node.py       # PyTorch inference node for classification
└── training/train_classifier.py # Offline training entry point
```

## Troubleshooting

- If RViz cannot find topics, ensure the workspace has been built (`catkin_make`) and the `devel/setup.bash` file is sourced in the current shell.
- The `camera_simulator` assumes the drone yaw is aligned with the positive X-axis. For more advanced dynamics, extend `drone_simulator.py` to track yaw and feed it into the camera rendering logic.
- To adjust world density or composition, tune `obstacle_density`, `obstacle_count`, `sphere_ratio`, `size_range`, and `height_range` in `sim.launch` or override them on the launch command line.

## License

MIT License.
