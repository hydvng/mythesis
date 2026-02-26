# 第3章V4优化训练实施计划（50秒回合）

## 目标
- 回合长度：50秒（比V3提升67%）
- 跟踪性能：误差RMS降低50%以上
- 收敛性：最后10秒误差 < 5mm, < 1°

## V4核心优化策略

### 1. 网络架构优化
基于V3发散问题分析，采用更深更宽的网络：
```python
# V4网络架构
hidden_dims = [512, 512, 512, 256, 256]  # 增加一层512
# 参数量：~650K (比V3提升38%)

# 改进点：
- 添加残差连接（Residual connections）
- LayerNorm → GroupNorm（更适合长序列）
- 添加Dropout(0.1)防止过拟合
```

### 2. 训练策略优化
针对50秒长回合的问题：
```python
# 学习率调整
lr_critic = 5e-4      # V3: 1e-3 → V4: 5e-4 (降低50%)
lr_actor = 2e-4       # V3: 3e-4 → V4: 2e-4
lr_alpha = 2e-4

# 梯度裁剪
grad_clip = 0.5       # V3: 1.0 → V4: 0.5 (更严格)

# 折扣因子
gamma = 0.995         # V3: 0.99 → V4: 0.995 (更关注长期)

# 目标网络更新
tau = 0.002           # V3: 0.005 → V4: 0.002 (更慢更新，更稳定)
```

### 3. 四阶段课程学习
更渐进的长度增加：
```python
# Stage 1: 基础 (20秒)
stage1 = {
    'episode_length': 20,
    'max_episodes': 250,
    'Hs': 1.0,
    'q_des_type': 'constant',
    'lr_critic': 5e-4,
    'lr_actor': 2e-4
}

# Stage 2: 跟踪 (35秒)
stage2 = {
    'episode_length': 35,
    'max_episodes': 350,
    'Hs': 2.0,
    'q_des_type': 'sinusoidal_small',  # 小幅正弦
    'lr_critic': 5e-4,
    'lr_actor': 2e-4
}

# Stage 3: 完整 (50秒)
stage3 = {
    'episode_length': 50,
    'max_episodes': 450,
    'Hs': 2.0,
    'q_des_type': 'sinusoidal',
    'lr_critic': 3e-4,  # 进一步降低
    'lr_actor': 1e-4
}

# Stage 4: 微调 (50秒, 高难度)
stage4 = {
    'episode_length': 50,
    'max_episodes': 300,
    'Hs': 2.5,          # 更大扰动
    'q_des_type': 'sinusoidal',
    'lr_critic': 1e-4,  # 很低的学习率微调
    'lr_actor': 5e-5
}
```

### 4. 奖励函数改进
更强的收敛激励：
```python
def _compute_reward_v4(...):
    # 1. 时序权重（非线性增加）
    progress = step / max_steps
    time_weight = 1.0 + 2.0 * progress**2  # 后期权重更大
    
    # 2. 误差惩罚（更严格）
    pos_weights = np.array([2000.0, 800.0, 800.0])  # 提高权重
    
    # 3. 收敛奖励（多阶段）
    if progress > 0.8 and error < threshold:
        reward += 5.0  # 最后20%收敛奖励
    if progress > 0.9 and error < threshold/2:
        reward += 10.0  # 最后10%更严格奖励
    
    # 4. 平滑性奖励（新）
    reward -= 0.001 * np.sum((u - u_prev)**2)  # 控制量变化平滑
```

### 5. 早停与正则化
防止V3的发散问题：
```python
# 早停策略
early_stop_patience = 30  # 30回合没有提升就停止
min_improvement = 10.0    # 最小提升阈值

# 学习率衰减
lr_decay_episodes = 100   # 每100回合衰减
lr_decay_factor = 0.95

# 检查发散
if episode_reward < -50:
    # 加载之前最佳模型，降低学习率继续
    agent.load(best_model_path)
    agent.lr_critic *= 0.8
```

## 文件清单

```
simulation/chapter3/
├── agents/
│   ├── v4_sac.py              # V4 SAC实现（需要创建）⭐
│   ├── advanced_sac.py        # V3版本
│   └── sac_agent.py           # 原版
├── env/
│   └── rl_env_v4.py           # V4环境（改进奖励）⭐
├── experiments/
│   ├── train_v4_curriculum.py # V4四阶段训练 ⭐
│   ├── evaluate_v4.py         # V4评估脚本 ⭐
│   └── models_v4/             # V4模型保存
│       ├── stage1/
│       ├── stage2/
│       ├── stage3/
│       └── stage4/
└── figures_v4/                # V4控制效果图
```

## 预期性能

| 指标 | V3 (30s) | V4目标 (50s) | 提升 |
|------|----------|--------------|------|
| **回合长度** | 30s | 50s | +67% |
| **z RMS误差** | 6.5mm | < 4mm | -38% |
| **α RMS误差** | 1.37° | < 0.8° | -42% |
| **β RMS误差** | 1.04° | < 0.6° | -42% |
| **最后10s z误差** | 5.9mm | < 3mm | -49% |
| **最后10s α/β误差** | 0.99°/0.76° | < 0.5° | -50% |

## 关键优化点总结

1. **更大网络**: 512×3+256×2 (~650K参数)
2. **残差连接**: 改善梯度流动
3. **更低学习率**: 防止长回合发散
4. **更慢目标更新**: tau 0.005→0.002
5. **更高gamma**: 0.99→0.995关注长期
6. **四阶段课程**: 20s→35s→50s→50s(高难度)
7. **更强收敛奖励**: 非线性时序权重+多阶段收敛奖
8. **平滑性奖励**: 减少控制抖动
9. **早停机制**: 防止发散
10. **学习率衰减**: 渐进优化

## 时间安排

- **Stage 1**: ~3小时 (250回合 × 20秒)
- **Stage 2**: ~6小时 (350回合 × 35秒)
- **Stage 3**: ~10小时 (450回合 × 50秒)
- **Stage 4**: ~6小时 (300回合 × 50秒)
- **总计**: ~25小时

## 风险控制

1. **发散预防**: 每个阶段设置早停和学习率衰减
2. **检查点**: 每50回合保存，可恢复
3. **验证**: 每阶段结束后验证最佳模型
4. **回退**: 如果发散，回退到上一阶段最佳模型

---

*计划创建时间: 2026-02-20*
*基于V3经验优化*
