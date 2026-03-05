# 扰动模型使用说明

## 概述

统一使用 `simulation/common/wave_disturbance.py`，该文件包含完整的扰动生成功能：
- 🌊 持续海浪扰动（ITTC谱 + MSS真实RAO数据）
- 💥 突变扰动（Burst Step）- 用于模拟撞击、载荷突变等

---

## 基础用法

### 1. 导入

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

from wave_disturbance import WaveDisturbance
```

### 2. 创建扰动模型

```python
# 基础海浪扰动
wave = WaveDisturbance(
    Hs=2.0,              # 有义波高 (m)
    T1=8.0,              # 平均周期 (s)
    vessel_file='supply.mat',
    wave_heading=180.0,  # 浪向角: 180=迎浪, 0=随浪, 90=横浪
    random_seed=42       # 随机种子（可重复实验）
)
```

### 3. 生成扰动

```python
import numpy as np

t = np.linspace(0, 30, 3000)  # 时间序列
disturbance = wave.generate_disturbance(t)

# 返回: (n_steps, 3) 数组
# disturbance[:, 0] -> Fz (升沉力, N)
# disturbance[:, 1] -> M_alpha (横摇力矩, N·m)
# disturbance[:, 2] -> M_beta (纵摇力矩, N·m)
```

---

## 突变扰动 (Burst Step)

### 用途
模拟突然的外部干扰：
- 🚁 UAV 着陆撞击
- 📦 载荷突变
- ⚡ 执行器故障
- 🛢️ 海上平台碰撞

### 方式一：手动配置

```python
wave = WaveDisturbance(
    Hs=2.0,
    T1=8.0,
    
    # 开启突变扰动
    enable_burst_step=True,
    step_t0=15.0,           # 扰动开始时间 (秒)
    step_duration=0.3,      # 扰动持续时间 (秒)
    step_ramp_time=0.03,    # 斜坡过渡时间 (避免数值冲击)
    
    # 扰动幅值 [Fz, M_alpha, M_beta]
    step_amplitude=[3000.0, 150.0, 150.0]
)
```

### 方式二：UAV 着陆预设

自动计算物理合理的着陆冲击：

```python
# 使用预设（推荐）
preset = WaveDisturbance.landing_uav_500kg_preset(
    t0=15.0,           # 着陆时间
    v_sink=0.5,        # 下沉速度 (m/s)
    impact_duration=0.25,  # 撞击持续时间
    ramp_time=0.06,    # 斜坡时间
    mass_uav=500.0,    # UAV质量
    eccentricity_x=0.1,  # 偏心距 x (m)
    eccentricity_y=0.05  # 偏心距 y (m)
)

wave = WaveDisturbance(
    Hs=2.0,
    T1=8.0,
    **preset  # 解包预设参数
)
```

**UAV 预设物理模型：**
- 静态载荷：$F_{static} = m \cdot g$
- 冲击增量：$\Delta F = m \cdot (v / \Delta t)$
- 偏心力矩：$M = e \times F_z$

---

## 高级配置

### 方向谱（推荐用于论文）

```python
wave = WaveDisturbance(
    Hs=2.0,
    T1=8.0,
    use_directional_spectrum=True,  # 开启方向谱
    n_directions=9,                  # 方向分量数
    spreading_exponent=2,             # cos^2s 分布
)
```

**推荐设置：**
| 场景 | n_directions | 说明 |
|------|--------------|------|
| 快速测试 | 1 (关闭方向谱) | 计算快 |
| 控制设计 | 9 | 精度与速度平衡 |
| 论文验证 | 15-21 | 最高精度 |

### 海况选择

```python
SEA_STATE_TABLE = {
    3: {"Hs": 1.0, "T1": 6.0},   # 粗糙
    4: {"Hs": 2.0, "T1": 8.0},   # 中等 (常用)
    5: {"Hs": 3.0, "T1": 9.5},   # 恶劣
    6: {"Hs": 4.0, "T1": 11.0},  # 非常恶劣
}
```

---

## 完整示例

### 示例1：基础海浪

```python
import numpy as np
import sys
sys.path.insert(0, 'simulation/common')
from wave_disturbance import WaveDisturbance

# 创建
wave = WaveDisturbance(Hs=2.0, T1=8.0, random_seed=42)

# 生成
t = np.linspace(0, 60, 6000)
d = wave.generate_disturbance(t)

print(f"Fz 扰动标准差: {np.std(d[:,0]):.2f} N")
print(f"M_alpha 标准差: {np.std(d[:,1]):.4f} N·m")
print(f"M_beta 标准差: {np.std(d[:,2]):.4f} N·m")
```

### 示例2：海浪 + 突变扰动

```python
import numpy as np
import sys
sys.path.insert(0, 'simulation/common')
from wave_disturbance import WaveDisturbance

# 使用 UAV 着陆预设
preset = WaveDisturbance.landing_uav_500kg_preset(
    t0=15.0,
    v_sink=0.5,
    impact_duration=0.25
)

wave = WaveDisturbance(Hs=2.0, T1=8.0, **preset)

# 生成（包含突变）
t = np.linspace(0, 30, 3000)
disturbance = wave.generate_disturbance(t)

# 可视化
wave.plot_burst_step_demo(save_name='uav_landing_disturbance.png')
```

### 示例3：在 RL 环境中使用

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

from wave_disturbance import WaveDisturbance
from your_rl_env import YourRLEnv

# 创建扰动（开启突变）
wave = WaveDisturbance(
    Hs=2.0,
    T1=8.0,
    enable_burst_step=True,
    step_t0=10.0,
    step_duration=0.5,
    step_amplitude=[2000, 100, 100]
)

# 传递给环境
env = YourRLEnv(wave_disturbance=wave)

# 或在环境中手动使用
for step in range(1000):
    t = step * dt
    disturbance = wave.generate_disturbance(np.array([t]))
    # 应用扰动到系统
```

---

## 参数速查表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `Hs` | float | 2.0 | 有义波高 (m) |
| `T1` | float | 8.0 | 平均周期 (s) |
| `vessel_file` | str | 'supply.mat' | 船型RAO数据 |
| `wave_heading` | float | 180.0 | 浪向角 (°) |
| `use_directional_spectrum` | bool | False | 开启方向谱 |
| `n_components` | int | 50 | 频率分量数 |
| `random_seed` | int | None | 随机种子 |

### 突变扰动参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_burst_step` | bool | False | 开启突变扰动 |
| `step_t0` | float | 15.0 | 开始时间 (s) |
| `step_duration` | float | 0.3 | 持续时间 (s) |
| `step_amplitude` | array | [3000,150,150] | 幅值 [Fz, Ma, Mb] |
| `step_ramp_time` | float | 0.03 | 斜坡时间 (s) |

---

## 文件位置

统一后的实现位于：

```
simulation/common/wave_disturbance.py  ← 唯一扰动源
```

所有 chapter 的 RL 环境应从 common 导入：

```python
# ✅ 正确
from wave_disturbance import WaveDisturbance

# ❌ 错误（已废弃）
# from simulation.disturbance.wave_disturbance import WaveDisturbance
```
