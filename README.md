# Autonomy Demo Workspace

This workspace contains a complete ROS1 (Noetic) simulation for a UAV navigating a randomly generated obstacle field with an RGB camera, automated dataset collection, PyTorch-based training, and an inference pipeline that can be driven from RViz.

## Features

- **Random obstacle world** featuring a mix of colorful boxes and spheres published as `visualization_msgs/MarkerArray` and `nav_msgs/OccupancyGrid`, ready for RViz visualization and programmatic access.
- **Configurable environment scale** with tunable obstacle density, size ranges, and shape ratios.
- **Kinematic UAV simulator** that consumes YOPO motion primitives from the inference stack (when available), otherwise falling back to direct goal chasing, while publishing pose/odometry data. The simulator now exposes a lightweight attitude controller that tracks 3–5 step primitives with explicit yaw/pitch offsets.
- **Synthetic RGB camera** that renders obstacle distances into color-coded imagery with optional torch-powered acceleration.
- **Automated data collection** pipeline that records RGB frames with near/far obstacle labels using analytic distance computation, reusing the same accelerated ray casting path as the camera.
- **PyTorch training utilities** featuring a UNet-style distance classifier trained with class-balanced augmentation plus cross-entropy + dice loss, along with detailed IoU/precision/recall reporting.
- **Binary-image-driven safe navigation** that extracts dominant safe zones, emits YOPO-style motion primitives spanning only 3–5 discrete `dt` steps (each with commanded offsets), optimizes differentiable policies for 3–7 m/s flight, and evaluates offsets in under 2 ms per frame.

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
- Use the **2D Nav Goal** tool in RViz to command the drone. When the inference node is running the UAV repeatedly executes the short-horizon YOPO primitives (3–5 × `primitive_dt`) that the policy publishes; otherwise it flies the straight-line fallback at a fixed altitude.
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

### Offline Batch Dataset Generation

- To procedurally capture large datasets without running ROS, use the standalone generator:

  ```bash
  python3 src/autonomy_demo/training/auto_dataset_generator.py \
    src/autonomy_demo/config/auto_dataset.yaml \
    --overwrite
  ```

- The script mirrors the live simulator: it samples the same obstacle shapes, reuses the analytic ray caster for RGB rendering and distance labels, and stores each environment (geometry snapshot plus RGB/label/depth triplets) under the configured output directory.

- Customize `src/autonomy_demo/config/auto_dataset.yaml` to tweak world scale, obstacle density, camera intrinsics, altitude/orientation bounds, or dataset counts. Provide `--output` to override the destination directory and `--overwrite` to clear an existing dataset before generation.

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
- The classifier stage applies horizontal flips, brightness/noise augmentation, and class-balanced weighting; each epoch reports training/validation loss together with safe-class IoU, precision, recall, and accuracy.
- A second differentiable reinforcement-learning loop optimizes the safe-navigation policy across 3–7 m/s speeds using the stored obstacle geometry; the learned offsets are written to `navigation_policy.pt`. The reward blends safety, goal alignment, smoothness, speed regulation, and stability, and the console summarizes the averaged metrics every epoch.
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
- The classifier output is smoothed with a probability threshold (`~safe_probability_threshold`, default 0.55) and morphological cleanup before region extraction, which helps the downstream policy maintain crisp masks.
- A differentiable policy evaluates the noisy safe mask and current airspeed to emit motion primitives and YOPO-inspired trajectories (3–5 discrete steps with explicit yaw/pitch offsets) while also proposing fallback commands. Outputs are published on:
  - `/drone/safe_center` (`geometry_msgs/PointStamped`): normalized safe-zone centroid and area fraction.
  - `/drone/movement_primitive` (`geometry_msgs/Vector3Stamped`): base vector toward the safe centroid with current-speed magnitude.
  - `/drone/movement_command` (`geometry_msgs/Vector3Stamped`): final command after applying length/angle offsets.
  - `/drone/movement_offsets` (`std_msgs/Float32MultiArray`): `[length_scale, pitch_deg, yaw_deg]` adjustments chosen by the policy.
  - `/drone/fallback_primitives` (`geometry_msgs/PoseArray`): rear/side slip options that remain available when no safe region is detected.
  - `/drone/safe_trajectory` (`nav_msgs/Path`): YOPO-blended primitive covering 3–5 × `primitive_dt` seconds. Each pose encodes the commanded yaw/pitch so the simulator’s attitude controller can track the offsets while marching toward the goal over successive replans.
- End-to-end processing is throttled to under 2 ms per frame; the node logs a warning if runtime exceeds the budget.
- RViz continues to display the RGB feed, the classification overlay, and the drone model. Use the 2D Nav Goal tool to validate how the navigation cues react to new viewpoints.
- Tune `~goal_tolerance` (default 0.3 m) to adjust how close the drone must get before a goal is considered complete.
- Tune `~primitive_steps` (clamped between 3 and 5) and `~primitive_dt` (seconds per step) on the inference node to adjust the horizon, and mirror `primitive_dt`/`attitude_gain` on the drone simulator for consistent tracking dynamics.

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
