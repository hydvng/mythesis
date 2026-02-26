# V4 Improved 训练脚本修改说明

## 需要修改的内容

### 1. 修改超参数 (第185-186行)

```python
# 修改前:
self.gamma = 0.99
self.tau = 0.005

# 修改后:
self.gamma = 0.995  # 与V4 Optimized一致
self.tau = 0.002    # 与V4 Optimized一致
```

### 2. 修改warmup_steps (第380行)

```python
# 修改前:
warmup_steps = 2000

# 修改后:
warmup_steps = 5000  # 与V4 Optimized一致
```

### 3. 移除探索噪声 (第384-398行)

```python
# 修改前:
for episode in range(env_config['max_episodes']):
    exploration_noise = max(0.1, 0.5 * (0.98 ** (episode // 50)))
    
    state = env.reset()
    ...
    while not done:
        if replay_buffer.size < warmup_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)
            noise = np.random.normal(0, exploration_noise, action_dim)
            action = np.clip(action + noise, -1, 1)

# 修改后:
for episode in range(env_config['max_episodes']):
    
    state = env.reset()
    ...
    while not done:
        if replay_buffer.size < warmup_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, deterministic=False)  # SAC自带探索
```

## 修改原因

| 参数 | 原值 | 新值 | 原因 |
|------|------|------|------|
| gamma | 0.99 | 0.995 | 更关注长期奖励 |
| tau | 0.005 | 0.002 | 目标网络更新更稳定 |
| warmup | 2000 | 5000 | 更多随机探索 |
| 探索噪声 | 有 | 无 | SAC已有熵探索 |

## 训练命令

```bash
cd /home/haydn/Documents/AERAOFMINE/mythesis/simulation/chapter3/experiments
conda activate leisaac
python train_v4_improved.py 2>&1 | tee train_v4_improved_v2.log
```
