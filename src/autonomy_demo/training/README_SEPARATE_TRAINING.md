# 独立训练和推理说明

本文档说明如何使用独立的训练和推理脚本，将图像分割（安全区域 vs 障碍物）和安全基元偏航（根据安全区域作为下一个目标点）分开训练和推演。

## 文件说明

### 训练脚本

1. **train_segmentation.py** - 图像分割训练脚本
   - 功能：训练二分类模型，区分安全区域和障碍物
   - 输入：包含图像和标签的数据集目录
   - 输出：分割模型权重文件（默认：`~/autonomy_demo/segmentation_model.pt`）

2. **train_navigation_policy.py** - 安全导航策略训练脚本
   - 功能：训练安全基元偏航策略，根据安全区域生成目标点
   - 输入：包含安全掩码和元数据的数据集目录
   - 输出：导航策略权重文件（默认：`~/autonomy_demo/navigation_policy.pt`）

3. **train_depth_model.py** - 深度回归训练脚本（DepthAnything 思路）
   - 功能：用模拟器提供的 `distances` 监督，训练 RGB->Depth 模型
   - 输出：深度模型权重文件（默认：`~/autonomy_demo/depth_model.pt`）

### 推理脚本

1. **inference_segmentation.py** - 图像分割推理节点（RGB-only）
2. **inference_navigation_policy.py** - 安全导航策略推理节点
3. **inference_depthseg.py** - 深度推理+自由空间分割（RGB-only）
   - 发布：`/drone/depth`, `/drone/safe_mask`, `/drone/rgb/distance_class`

## 使用方法

### A. 经典二分类分割

```bash
python3 src/autonomy_demo/training/train_segmentation.py \
  /path/to/dataset --epochs 20 --batch 16 --lr 1e-3 \
  --output ~/autonomy_demo/segmentation_model.pt

rosrun autonomy_demo inference_segmentation.py \
  _model_path:=~/autonomy_demo/segmentation_model.pt
```

### B. DepthAnything 式深度先验（推荐获得更稳定的地/障分离）

1) 训练深度模型：
```bash
python3 src/autonomy_demo/training/train_depth_model.py \
  /path/to/dataset --epochs 20 --batch 8 --lr 1e-3 \
  --max_range 12.0 \
  --output ~/autonomy_demo/depth_model.pt
```

2) 仅依赖 RGB 的深度推理 + 掩码：
```bash
rosrun autonomy_demo inference_depthseg.py \
  _depth_model:=~/autonomy_demo/depth_model.pt \
  _max_range:=12.0 \
  _hysteresis_low:=0.35 \
  _hysteresis_high:=0.45 \
  _morph:=3 \
  _floor_ratio:=0.18
```

- 参数含义：
  - `_hysteresis_low/_high`：双阈值（相对地面中位深度），高阈值更“保守”地认为远处为安全。
  - `_morph`：形态学核大小（像素），用于去噪和填洞。
  - `_floor_ratio`：用于估计地面带（图像底部百分比）。

3) 如需将深度法与导航策略联动，只需让 `inference_navigation_policy.py` 订阅 `/drone/safe_mask` 即可（无需更改）。

## 提示

- 数据采集阶段 `data_collector.py` 已保存 `distances`，可直接用作深度监督。
- 若场景跨度较大，可调大 `--max_range` 并在推理时保持一致。
- 深度法通常对阴影、纹理缺失更鲁棒，能得到更“平滑”的地/障边界（类似图四的效果）。

