# V5 Uniform 训练说明

这个版本把 `V5 Simplified` 的核心思想迁移到了 `UniformRodPlatform3DOF`：

- 纯 ISE 奖励：`reward = -||e||^2`
- 保留模型补偿 + RL 残差结构
- 保留 V3 式三阶段课程学习
- Stage 3 默认启用 burst-step 突变扰动

## 文件

- `env/rl_env_v5_uniform.py`：V5 风格 Uniform 环境
- `experiments/train_v5_curriculum_uniform.py`：训练脚本

## 训练阶段

1. Stage 1：10 秒，定点稳定，低海况
2. Stage 2：20 秒，小幅正弦跟踪，中等海况
3. Stage 3：30 秒，完整正弦跟踪 + burst-step 突变扰动

## 设计备注

- 这里沿用了 `AdvancedSAC`，因为它已经在当前 Uniform 训练链路中验证过可用。
- 这样做的目标是：优先比较“奖励设计变化”带来的影响，而不是一次同时更换 reward、agent、network 三者。
- 如果这一版稳定，再进一步尝试更贴近旧版 `V4/V5` 的 SAC 变体会更稳妥。

## 快速验证

可以运行 `train_curriculum_v5_uniform_quick()` 做极小规模冒烟测试。