# 第2章仿真说明

本目录用于生成论文第2章“系统建模与约束分析”相关图件。

## 当前使用的平台模型

第2章仿真当前使用的最新平台模型为：`simulation/common/uniform_rod_platform_dynamics.py` 中的 `UniformRodPlatform3DOF`。

该模型在基础 3-DOF 平台模型之上，补充了“均匀杆件”带来的动力学影响。

### 当前已纳入的动力学项

- 增强后的质量矩阵 $M(q)$
  - 包含平台本体项
  - 包含杆件平动动能对应项 `M_rod_trans`
  - 包含杆件转动动能对应项 `M_rod_rot`
- 增强后的重力项 $G(q)$
  - 包含平台重力项
  - 包含杆件重力项 `G_rod`

### 当前近似处理的动力学项

- 科里奥利项 $C(q, \dot q)$
  - 当前仍复用基础模型 `ParallelPlatform3DOF` 的解析形式
  - 因此，第2章仿真已使用最新的均匀杆件质量/重力建模，但尚未引入严格一致的 rod-Coriolis 项

## 仿真入口

第2章图件生成脚本：`simulation/chapter2/experiments/chapter2_simulation.py`

该脚本会同时加载：

- `simulation/common` 中的平台运动学/动力学模型
- `simulation/disturbance` 中的波浪扰动模型 `WaveDisturbance`
- `simulation/disturbance` 中的 UAV 降落扰动模型 `uav_landing_disturbance.py`

图输出目录：`simulation/chapter2/figures/`

其中第2章现已包含一张复合扰动建模对比图，用于展示：

- UAV only
- Wave only
- Wave + UAV

## 说明

如果后续补齐了杆件对应的严格 $C(q, \dot q)$ 项，建议同步更新本说明文件与论文第2章相关文字，使“建模假设”和“仿真使用模型”保持一致。