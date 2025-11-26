# 无人机单目安全导航（ROS1 Noetic）

基于**单目 RGB 相机**与**像素级安全/障碍物二分类**的无人机导航项目。流程包含：
- 自动数据采集（在线模拟或离线生成）：同步保存图像、mask、机体状态与目标。
- 离线训练：UNet 分割网络 + 末端状态预测（PlannerNet）+ 五阶多项式（quintic）轨迹生成。
- 在线推理与控制：SegNet 与 PlannerNet 推理后发布轨迹与 `/cmd_vel`。

## 依赖环境
- Ubuntu 20.04 + ROS Noetic（需 `cv_bridge`、`image_transport` 等常见依赖）。
- Python 3，推荐使用虚拟环境。
- 主要 Python 包：`torch`、`torchvision`、`numpy`、`pillow`、`opencv-python`、`pyyaml`。

## 编译与安装
1. 将仓库放入 `~/catkin_ws/src`（或任意 catkin 工作空间的 `src`）：
   ```bash
   cd ~/catkin_ws/src
   git clone <repo_url> autonomy_demo
   ```
2. 在工作空间根目录编译并加载环境：
   ```bash
   cd ~/catkin_ws
   catkin_make
   source devel/setup.bash
   ```

## 数据采集
支持两种模式：

- **离线批量生成**（默认）：
  ```bash
  roslaunch autonomy_demo data_collection.launch \
    dataset_config:=config/auto_dataset.yaml \
    output_dir:=__from_config__ \
    overwrite:=true
  ```
  - 输出路径由 `dataset_config` 的 `dataset.output_dir` 指定，或通过 `output_dir` 覆盖。
  - 生成的目录结构：`env_xxx/world_snapshot.npz` + `env_xxx/sample_yyyyy.npz`，每个样本包含 `image`、`label`/`mask`、`distances`、`pose_position`、`pose_orientation`、`camera_offset` 等字段。

- **在线采集（仿真驱动）**：
  ```bash
  roslaunch autonomy_demo data_collection.launch mode:=online
  ```
  - 会启动 `sim.launch` 的世界、无人机与相机模拟器，并运行 `data_collector` 节点。
  - 默认保存到 `~/autonomy_demo/dataset`，文件名形如 `sample_000000.npz`，字段与离线生成一致。

## 训练步骤
1. **分割网络（SegNet）**：
   ```bash
   python3 src/autonomy_demo/training/train_segmentation.py \
     /path/to/dataset --epochs 20 --batch 16 --lr 1e-3 \
     --output ~/autonomy_demo/segmentation_model.pt --distill
   ```

2. **末端状态 / 规划网络（PlannerNet）**：
   ```bash
   python3 src/autonomy_demo/training/train_navigation_policy.py \
     /path/to/dataset --epochs 40 --batch 8 --lr 5e-4 \
     --output ~/autonomy_demo/navigation_policy.pt
   ```

两者共用 `*.npz` 样本：`train_segmentation.py` 读取 `label`/`mask` 或由 `distances` + `near_threshold` 生成标签；`train_navigation_policy.py` 额外使用 `pose_*`、障碍物快照等元数据以生成 quintic 基元监督。

## 在线推理与控制
启动端到端推理（SegNet + PlannerNet）：
```bash
roslaunch autonomy_demo inference.launch \
  model_path:=~/autonomy_demo/model.pt \
  policy_path:=~/autonomy_demo/navigation_policy.pt
```

关键话题：
- 订阅：`/drone/rgb/image_raw`、`/drone/rgb/camera_info`、`/drone/odometry`、`/move_base_simple/goal`。
- 发布：
  - `/drone/rgb/distance_class`：红/绿二分类可视化。
  - `/drone/safe_center`：最大安全连通块中心。
  - `/drone/movement_offsets`：规划偏移与时长（供模拟器跟踪）。
  - `/drone/safe_trajectory`：五阶多项式轨迹（`Path` 消息，可被 `drone_simulator` 或实际控制器跟踪）。

若需要将分割与规划分开运行，可使用 `inference_separate.launch` 启动 `inference_segmentation.py` 与 `inference_navigation_policy.py` 两个节点。

## 注意事项
- **训练 vs. 推理的障碍物信息**：
  - 训练/数据生成阶段可直接获取环境真值并生成精确 mask 与障碍物快照。
  - 在线推理阶段仅依赖相机图像和已训练模型，不再访问环境真值；因此请确保相机参数（分辨率、FOV、pitch、`camera_offset`）与训练一致，避免域偏移。

- **超参数调节**：
  - 轨迹采样：`radio_range`（视野半径）、`v_max`/`a_max`、`path_samples_per_step`、`yaw/pitch_std_deg` 等参数在 `inference.launch` 与训练脚本中一一对应，需保持一致。
  - 规划代价：`primitive_safety_gate`、`primitive_clearance_gate`、`min_clearance_fraction`、`offset_gain` 控制安全裕度；`goal_stop_distance`、`goal_tolerance` 决定收敛判据。
  - 损失权重：`train_segmentation.py` 中可调整 `distill_weight`、`teacher_weight`；`train_navigation_policy.py` 中可修改 jerk/姿态变化等惩罚系数以平衡平滑度与机动性。

## 数据流与逻辑校验摘要
- **数据采集链路**：`world_generator` 发布 `world/obstacles`；`drone_simulator` 发布 `/drone/pose` 与 `/drone/odometry`；`camera_simulator` 基于上述信息渲染 `/drone/rgb/image_raw` 与 `camera_info`；`data_collector` 订阅上述话题生成 `*.npz` 样本（图像 + mask/距离 + 机体姿态 + 障碍物快照）。
- **分割训练链路**：`train_segmentation.py`/`train_classifier.py` 通过 `ObstacleDataset` 递归读取 `*.npz`，可直接使用 mask，或从 `distances` + `near_threshold` 重建标签；均值方差在训练时统计并写入 checkpoint，推理端自动复用。
- **末端状态训练链路**：`train_navigation_policy.py`/`NavigationDataset` 读取同一批样本，使用安全掩码与障碍物快照生成安全区域中心、清障距离与相机射线，进一步评估 jerk、姿态变化、goal 对齐等奖励来拟合 PlannerNet。
- **推理链路**：`inference_node.py` 订阅相机、里程计和 goal，将图像经 SegNet 得到安全 mask，再由 PlannerNet 预测末端状态与偏移，结合 quintic 轨迹（`path_samples_per_step`、`radio_range`、`vel_max_train` 等参数）生成 `/drone/safe_trajectory` 与 `/drone/movement_offsets`，实现闭环控制。

