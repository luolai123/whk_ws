# Autonomy Demo Workspace

This workspace contains a complete ROS1 (Noetic) simulation for a UAV navigating a randomly generated obstacle field with an RGB camera, automated dataset collection, PyTorch-based training, and an inference pipeline that can be driven from RViz.

## Features

- **Random obstacle world** featuring a mix of colorful boxes and spheres published as `visualization_msgs/MarkerArray` and `nav_msgs/OccupancyGrid`, ready for RViz visualization and programmatic access.
- **Configurable environment scale** with tunable obstacle density, size ranges, and shape ratios.
- **Kinematic UAV simulator** that responds to RViz 2D Nav Goal inputs and publishes pose/odometry data.
- **Synthetic RGB camera** that renders obstacle distances into color-coded imagery with optional torch-powered acceleration.
- **Automated data collection** pipeline that records RGB frames with near/far obstacle labels using analytic distance computation, reusing the same accelerated ray casting path as the camera.
- **PyTorch training utilities** for a two-class distance classifier and an inference node that streams classification maps in real time.
- **Binary-image-driven safe navigation** that extracts dominant safe zones, emits goal-aligned motion primitives, and evaluates torch-trained offset policies in under 2 ms per frame.

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

- Increase the simulated airspace or avoid seeing beyond the borders by raising `world_size` (the default launch configuration spans `160 x 160 x 14` meters).
- Control crowding with `obstacle_density` (obstacles per square meter). When the density is zero the system falls back to the legacy `obstacle_count` parameter.
- Mix shape variety with `sphere_ratio` (0.0–1.0) to choose how many obstacles are rendered as spheres versus boxes.
- Adjust `size_range` and `height_range` to refine individual obstacle proportions. All parameters can be overridden in `sim.launch` or via ROS command-line arguments when launching.

## Automated Dataset Collection

Start the simulator together with the data collector:

```bash
roslaunch autonomy_demo data_collection.launch output_dir:=/your/dataset/path
```

- Samples are stored as compressed `.npz` files containing the RGB image, the 2-class label map (near obstacle vs. safe), per-column distance estimates, the capturing pose/orientation, camera offset, and the obstacle geometry snapshot used during labeling.
- The default near/far threshold is 4 meters; override with the `near_threshold` parameter if required.

> Tip: move the UAV around using RViz goals while data collection is running to diversify the dataset. You can also call the regenerate service to change obstacle layouts between runs.

### Optional Hardware Acceleration

- Both the RGB camera simulator and the automated data collector can offload ray casting to PyTorch. Enable the GPU-backed code paths (requires a CUDA-capable build of PyTorch) by appending launch arguments, for example:

  ```bash
  roslaunch autonomy_demo sim.launch hardware_accel:=true hardware_device:=cuda
  ```

  The `hardware_device` parameter accepts any device string understood by `torch.device` (e.g. `cuda:0`, `cuda:1`, or `cpu`).

- The data collection pipeline inherits the same arguments:

  ```bash
  roslaunch autonomy_demo data_collection.launch hardware_accel:=true output_dir:=/tmp/dataset
  ```

- When CUDA is unavailable or PyTorch is not installed, the nodes automatically fall back to the optimized NumPy implementation.

### Performance Tuning

- Use `max_obstacle_candidates` (default `512`) to cap how many nearby objects each ray considers. Lower values improve frame rate at the cost of ignoring far obstacles; `0` keeps the full set. The parameter is exposed on `sim.launch` and `data_collection.launch` for both the camera simulator and the dataset recorder.

## Training the Classifier and Safe-Navigation Policy

After gathering data, train both the segmentation network and the differentiable navigation policy:

```bash
python3 src/autonomy_demo/training/train_classifier.py \
  /your/dataset/path \
  --epochs 15 \
  --batch 8 \
  --lr 5e-4 \
  --policy_epochs 40 \
  --policy_batch 8 \
  --policy_lr 5e-4 \
  --policy_noise 0.03 \
  --output ~/autonomy_demo/model.pt \
  --policy_output ~/autonomy_demo/navigation_policy.pt
```

- The script still splits the dataset for validation and saves the classifier weights to `model.pt`.
- A second differentiable reinforcement-learning loop optimizes the safe-navigation policy across 3–7 m/s speeds using the stored obstacle geometry; the learned offsets are written to `navigation_policy.pt`.
- Use `--no_policy` to skip the policy stage when you only need the segmentation network.
- Adjust hyperparameters as needed. GPU acceleration is used automatically when CUDA is available.

## Real-Time Inference

With classifier and policy weights stored under `~/autonomy_demo`, launch the full inference stack:

```bash
roslaunch autonomy_demo inference.launch \
  model_path:=/absolute/path/to/model.pt \
  policy_path:=/absolute/path/to/navigation_policy.pt
```

- The `distance_inference` node subscribes to the RGB stream, produces the binary classification overlay on `/drone/rgb/distance_class`, and extracts the largest contiguous safe zone (default minimum area 5%).
- A differentiable policy evaluates the noisy safe mask and current airspeed to emit motion primitives and offset recommendations. Outputs are published on:
  - `/drone/safe_center` (`geometry_msgs/PointStamped`): normalized safe-zone centroid and area fraction.
  - `/drone/movement_primitive` (`geometry_msgs/Vector3Stamped`): base vector toward the safe centroid with current-speed magnitude.
  - `/drone/movement_command` (`geometry_msgs/Vector3Stamped`): final command after applying length/angle offsets.
  - `/drone/movement_offsets` (`std_msgs/Float32MultiArray`): `[length_scale, pitch_deg, yaw_deg]` adjustments chosen by the policy.
  - `/drone/fallback_primitives` (`geometry_msgs/PoseArray`): rear/side slip options that remain available when no safe region is detected.
- End-to-end processing is throttled to under 2 ms per frame; the node logs a warning if runtime exceeds the budget.
- RViz continues to display the RGB feed, the classification overlay, and the drone model. Use the 2D Nav Goal tool to validate how the navigation cues react to new viewpoints.

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
