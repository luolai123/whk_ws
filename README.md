# Autonomy Demo Workspace

## 项目概览 / Project Overview
- **中文**：该工作区面向 Ubuntu 20.04 + ROS Noetic，提供随机障碍世界、四旋翼+RGB 相机仿真、自动数据采集、二分类+安全导航训练以及推理闭环，可直接通过 RViz 进行交互。
- **English**: Complete ROS1 workspace for a UAV navigating procedurally generated obstacle fields with a pitched RGB camera, automated dataset capture, PyTorch training, and short-horizon safe navigation that can be monitored and directed from RViz.

## 功能亮点 / Feature Highlights
- **随机世界 Randomized world**：支持巨大的可配置空域、盒体/球体/门框等多类障碍，并输出 `MarkerArray` + `OccupancyGrid` 供 RViz 与算法订阅。
- **统一相机工具链 Unified camera utilities**：摄像头与数据采集器共享射线预计算与安装矩阵（`camera_utils.py`），确保上抬视角、相对机体系偏移完全一致，避免冗余代码与视角错位。
- **自动采集 + 真实标签 Automated data capture**：`data_collection.launch` 默认调用离线采集器，逐像素光线投射生成 RGB/深度/红绿二分类标签，并保存完整障碍快照供训练使用。
- **可微训练 Differentiable training**：UNet 二分类器 + 可微安全导航策略同时训练，奖励函数覆盖安全距离、目标导向、平滑与速度指标，适配 3–7 m/s 速度段。
- **推理与轨迹 Tracking inference**：推理节点提取最大安全区，使用五阶多项式在 3–5×`primitive_dt` 内生成安全运动基元，并由姿态控制器跟踪。

## 环境需求 / Requirements
- Ubuntu 20.04, ROS Noetic (desktop-full 建议). 需要 `cv_bridge`、`image_transport` 等 ROS 包。
- Python 依赖 / Python deps:
  ```bash
  pip install numpy torch torchvision pillow
  ```

## 编译步骤 / Build
```bash
source /opt/ros/noetic/setup.bash
cd /path/to/whk_ws
catkin_make
source devel/setup.bash
```

## 运行仿真 / Run the Simulation
```bash
roslaunch autonomy_demo sim.launch
```
- RViz 会自动加载 `config/world.rviz`。Use the **2D Nav Goal** tool to select the next waypoint.
- 主题 Topics：`/world/obstacles` (MarkerArray), `/world/occupancy` (OccupancyGrid), `/drone/rgb/image_raw` (RGB feed), `/drone/safe_trajectory` (短期安全轨迹)。
- 通过 `rosservice call /world_generator/regenerate` 可即时刷新场景。

## 自动数据采集 / Automated Dataset Collection
默认离线无界面：
```bash
roslaunch autonomy_demo data_collection.launch \
  dataset_config:=config/auto_dataset.yaml \
  output_dir:=/your/dataset/path \
  overwrite:=true
```
- `dataset_config` 可写绝对路径、`package://autonomy_demo/...`，或简单地写成 `config/auto_dataset.yaml`（会自动按包路径解析，即使误写成 `/config/...` 也会自动修正）；`dataset_config` accepts absolute paths, `package://autonomy_demo/...` URIs, or short `config/auto_dataset.yaml` strings and will auto-resolve them to the package.
- `output_dir:=__from_config__`（默认值）时使用 YAML 中的 `dataset.output_dir`；显式填写则覆盖。
- `overwrite` 接收 `true/false` 字符串。
- YAML 描述环境范围、采样次数、安全距离、相机姿态等，生成的 `env_xxx/sample_xxxxx.npz` 包含 RGB、红/绿二分类标签（红=障碍，绿=安全）、深度、相机偏移与障碍快照。
- 默认 `config/auto_dataset.yaml` 将分辨率固定为 128×72，与 `sim.launch` 中的相机参数一致，便于训练-推理对齐；如需更高分辨率，可同步修改 launch 与 YAML 中的宽高。

若需要旧版在线采集（打开 RViz + 手动移动）：
```bash
roslaunch autonomy_demo data_collection.launch mode:=online hardware_accel:=true
```
此模式会启动模拟器，采集节点实时记录图像/标签。

### CLI (Headless) 方式 / Standalone CLI
```bash
python3 src/autonomy_demo/training/auto_dataset_generator.py \
  src/autonomy_demo/config/auto_dataset.yaml \
  --output /tmp/dataset_run \
  --overwrite true
```
参数支持 `true/false` 字符串，便于脚本或 `roslaunch` 统一调用。

## 训练流程 / Training Pipeline
```bash
python3 src/autonomy_demo/training/train_classifier.py \
  /your/dataset/path \
  --epochs 15 --batch 8 --lr 5e-4 \
  --policy_epochs 40 --policy_batch 8 --policy_lr 5e-4 \
  --output ~/autonomy_demo/model.pt \
  --policy_output ~/autonomy_demo/navigation_policy.pt
```
- 如需专注提升像素级二分类，可运行 `training/train_segmentation.py`：
  ```bash
  python3 src/autonomy_demo/training/train_segmentation.py \
    /your/dataset/path --epochs 20 --batch 16 --lr 1e-3 \
    --output ~/autonomy_demo/segmentation_model.pt --distill
  ```
- For a segmentation-only curriculum run `training/train_segmentation.py` with the same dataset; it reuses the UNet, applies class-balanced losses, and optionally distills from the teacher branch.
- 训练集自动划分验证集，报告 loss、IoU、precision/recall。
- 训练脚本会在切分后统计训练集 RGB 通道均值/方差，并写入模型 checkpoint；推理节点会自动按同样的分辨率与标准化重新采样输
  入帧，避免出现“全部红色”的 domain shift。
- 二分类标签使用红/绿二值图，损失函数为加权交叉熵 + dice；推理阶段会渲染红色障碍与绿色安全区，并仅对聚类得到的最大安全斑块执行轨迹规划，确保路径永远位于真实连通区域内。
- 策略阶段以五阶多项式轨迹评估安全性（碰撞率、最小距离）、目标导向度、轨迹 jerk 峰值与姿态变化率，并在训练时强制将采样目标与安全斑块边界的距离、角度关联起来，使策略真正学会“指向目标又保持安全”。

### YOPO 式安全运动基元 / YOPO-style motion primitives
- **中文**：`train_classifier.py` 与 `inference_node.py` 现共享同一套 YOPO 运动基元参数。`radio_range` 定义轨迹地平线（默认 5 m），结合 `vel_max_train`、`primitive_dt/primitive_steps` 自动推导 3–5×`dt` 的轨迹时长。`v_forward_mean/sigma`（对数正态）、`v_std_unit`、`a_std_unit` 控制机体系速度/加速度采样分布，而 `yaw_std_deg`、`pitch_std_deg` + `horizon/vertical fov` 确保采样方向位于相机视野内。策略网络输出 3 维偏移量 + 1 个时长缩放，`offset_gain` 负责约束偏移幅度。`inference.launch` 中的参数与训练脚本的 `--radio_range --vel_max_train ... --camera_pitch_deg` 保持一一对应，可直接复现训练时的运动基元云。
- **English**: Both the trainer and the inference node now draw YOPO-style primitives from the same `PrimitiveConfig`. `radio_range` fixes the horizon (5 m by default), while `vel_max_train`, `primitive_dt`, and `primitive_steps` determine the 3–5 × dt duration. Forward velocity follows a log-normal law (`v_forward_mean`/`v_forward_sigma`), lateral/vertical components use zero-mean Gaussians scaled by `v_std_unit`, and accelerations mirror `a_std_unit`. `yaw_std_deg`/`pitch_std_deg` together with the camera FOV restrict candidate directions to what the RGB sensor can actually observe. The policy outputs a 3D offset and a duration scale that are clamped via `offset_gain` and `duration_scale_[min|max]`. Keep the new CLI flags (`--radio_range`, `--primitive_dt`, `--path_samples_per_step`, `--camera_pitch_deg`, etc.) aligned with the ROS parameters in `inference.launch` to guarantee that training and runtime primitives stay consistent.

## 推理部署 / Inference Deployment
```bash
roslaunch autonomy_demo inference.launch \
  model_path:=/abs/path/model.pt \
  policy_path:=/abs/path/navigation_policy.pt
```
- 节点输出：
  - `/drone/rgb/distance_class`：红/绿叠加图。
  - `/drone/safe_center`：最大安全斑块中心点。
  - `/drone/movement_command` / `/drone/movement_offsets`：长度、俯仰、偏航调节（offsets 现包含完整轨迹时长，模拟器据此平滑跟踪）。
  - `/drone/safe_trajectory`：3–5 dt 的五阶多项式安全轨迹，姿态控制器按该轨迹跟踪至下个 goal；若当前路径与上一条相似，会自动保持旧轨迹以避免“走走停停”。
- `~goal_direction_blend`、`~goal_bias_distance` 控制安全方向与 RViz 目标方向的融合比例；`~plan_publish_period` 决定最小轨迹刷新周期，而 `~plan_hold_time` / `~plan_similarity_epsilon` 则决定何时强制推送最新轨迹，可显著减少实时运行时的顿挫感。
- `~primitive_steps`、`~primitive_dt`、`~path_samples_per_step`（用于在同一 3–5 dt 轨迹内增加采样点，从而提升模拟器跟踪的连续性）、`~camera_pitch_deg`、`~max_obstacle_candidates` 等参数可在 launch 文件内调整，实现实时性与安全性折中。
- 模型文件保存为包含 `model_state`、`normalization(mean/std)` 以及 `input_size` 的字典；旧版仅含 `state_dict` 的模型仍可加载，但推理节点不会应用额外的标准化或重采样。

## 性能优化与容错 / Performance & Robustness
- `camera_utils.py` 在摄像机和数据采集端共享射线与安装矩阵，避免重复计算并保证视角一致。`max_obstacle_candidates` + Torch 加速可控制单帧耗时。
- 推理阶段对候选轨迹执行安全一票否决：任何低概率或低净空的轨迹会被直接丢弃，并触发后退/侧移备用基元。
- `data_collection.launch` 的 offline 模式不再启动 RViz，可在服务器/CI 上持续采集；需要图形界面时切换到 `mode:=online`。

## 常见问题 / Troubleshooting
- **构建失败 Build issues**：确保执行 `catkin_make` 前已 `source /opt/ros/noetic/setup.bash`。
- **相机视角错误 Camera pitch**：调整 `camera_pitch_deg`，正值表示向上抬头；摄像机、数据采集器、离线生成器都会共享该偏置。
- **运行缓慢 Performance**：降低 `max_obstacle_candidates`、减小分辨率，或设置 `hardware_accel:=true hardware_device:=cuda` 以启用 torch。

## 目录结构 / Repository Layout
```
whk_ws/
├── README.md
├── src/autonomy_demo/
│   ├── config/                # RViz & dataset 配置
│   ├── launch/                # sim / data_collection / inference 启动文件
│   ├── scripts/               # world_generator, drone_simulator, camera_simulator, data_collector, inference_node
│   ├── src/autonomy_demo/     # Python 模块：obstacle_field, safe_navigation, camera_utils
│   ├── training/              # train_classifier.py, auto_dataset_generator.py
│   └── setup.py, CMakeLists.txt, package.xml
└── ...
```

## License
MIT License.
