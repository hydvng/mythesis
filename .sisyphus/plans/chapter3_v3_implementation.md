# 第3章V3训练实施计划

## 待完成任务

### 1. 修改RL环境 (rl_env.py)
- [x] 实现时序加权奖励函数 `_compute_reward` with time_weight
- [x] 添加 `_compute_temporal_weight` 方法
- [x] 添加 `_compute_convergence_reward` 方法
- [ ] 在step()中调用时序奖励计算

### 2. 创建课程学习训练脚本 (train_v3_curriculum.py)
- [ ] 导入AdvancedSAC
- [ ] 定义课程学习配置类
- [ ] 实现三阶段训练逻辑
- [ ] 每阶段保存模型

### 3. 训练执行
- [ ] 阶段1: 10秒回合, Hs=1.0m, 200回合
- [ ] 阶段2: 20秒回合, Hs=2.0m, 300回合 (加载阶段1模型)
- [ ] 阶段3: 30秒回合, Hs=2.0m, 500回合 (加载阶段2模型)

### 4. 评估和对比
- [ ] 对比V2和V3的控制效果
- [ ] 分析稳态收敛性能
- [ ] 绘制对比图表

## 关键配置参数

### 网络架构
```python
hidden_dims = [512, 512, 256, 256]
lr_actor = 3e-4
lr_critic = 1e-3
batch_size = 512
```

### 课程学习
```python
stage1 = {
    'episode_length': 10,  # 秒
    'max_episodes': 200,
    'Hs': 1.0,
    'trajectory': 'constant'
}

stage2 = {
    'episode_length': 20,
    'max_episodes': 300,
    'Hs': 2.0,
    'trajectory': 'sinusoidal_small'
}

stage3 = {
    'episode_length': 30,
    'max_episodes': 500,
    'Hs': 2.0,
    'trajectory': 'sinusoidal_full'
}
```

## 文件清单

```
simulation/chapter3/
├── agents/
│   ├── advanced_sac.py          # ✓ 已完成
│   └── sac_agent.py             # 原版
├── env/
│   └── rl_env.py                # 需要修改step()
├── experiments/
│   ├── train_v3_curriculum.py   # 需要创建
│   └── plot_v3_results.py       # 需要创建
└── SYSTEM_ANALYSIS_AND_DESIGN.md # ✓ 已完成
```

## 预期结果

- z方向RMS误差 < 0.01m
- α/β方向RMS误差 < 2°
- 稳态收敛在最后5秒
- 约束满足率100%
