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
  dataset_config:=$(find autonomy_demo)/config/auto_dataset.yaml \
  output_dir:=/your/dataset/path \
  overwrite:=true
```
- `output_dir:=__from_config__`（默认值）时使用 YAML 中的 `dataset.output_dir`；显式填写则覆盖。
- `overwrite` 接收 `true/false` 字符串。
- YAML 描述环境范围、采样次数、安全距离、相机姿态等，生成的 `env_xxx/sample_xxxxx.npz` 包含 RGB、红/绿二分类标签（红=障碍，绿=安全）、深度、相机偏移与障碍快照。

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
- 训练集自动划分验证集，报告 loss、IoU、precision/recall。
- 二分类标签使用红/绿二值图，损失函数为加权交叉熵 + dice；推理阶段会渲染红色障碍与绿色安全区。
- 策略阶段以五阶多项式轨迹评估安全性（碰撞率、最小距离）、目标导向度、轨迹 jerk 峰值与姿态变化率。

## 推理部署 / Inference Deployment
```bash
roslaunch autonomy_demo inference.launch \
  model_path:=/abs/path/model.pt \
  policy_path:=/abs/path/navigation_policy.pt
```
- 节点输出：
  - `/drone/rgb/distance_class`：红/绿叠加图。
  - `/drone/safe_center`：最大安全斑块中心点。
  - `/drone/movement_command` / `/drone/movement_offsets`：长度、俯仰、偏航调节。
  - `/drone/safe_trajectory`：3–5 dt 的五阶多项式安全轨迹，姿态控制器按该轨迹跟踪至下个 goal。
- `~primitive_steps`、`~primitive_dt`、`~camera_pitch_deg`、`~max_obstacle_candidates` 等参数可在 launch 文件内调整，实现实时性与安全性折中。

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
