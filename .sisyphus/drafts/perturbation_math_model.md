# 船载Stewart平台扰动传递数学模型

## 1. 问题背景

### 1.1 物理场景

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              船体坐标系 (惯性坐标系)                          │
│                                                                             │
│    ┌───────────────────────────────────────────────────────────────────┐    │
│    │                        船体                                        │    │
│    │                                                                       │    │
│    │           ┌─────────────────────┐                                   │    │
│    │           │      下平台         │  ← 固定在船上，跟随船体运动        │    │
│    │           │   q_lower = q_ship  │                                   │    │
│    │           └─────────────────────┘                                   │    │
│    │                   ↑ L₁,L₂,L₃                                       │    │
│    │                   │   (活动杆，长度不变)                             │    │
│    │                   ↓                                                │    │
│    │           ┌─────────────────────┐                                   │    │
│    │           │      上平台         │  ← 需要保持稳定，抵消船体运动       │    │
│    │           │      q_upper        │                                   │    │
│    │           └─────────────────────┘                                   │    │
│    │                                                                       │    │
│    └───────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 坐标定义

| 符号 | 含义 | 说明 |
|------|------|------|
| $O$ | 惯性坐标系原点 | 固定在空间中 |
| $O_b$ | 船体坐标系原点 | 随船体运动 |
| $q_s = [z_s, \alpha_s, \beta_s]^T$ | 船体运动位移 | 升沉、横摇、纵摇 |
| $q_{sd} = [\dot{z}_s, \dot{\alpha}_s, \dot{\beta}_s]^T$ | 船体运动速度 |  |
| $q_{sdd} = [\ddot{z}_s, \ddot{\alpha}_s, \ddot{\beta}_s]^T$ | 船体运动加速度 |  |
| $q_u = [z_u, \alpha_u, \beta_u]^T$ | 上平台位移 | 相对于惯性系 |
| $L_i$ | 第 $i$ 根活动杆长度 | $i=1,2,3$ |

---

## 2. 扰动传递模型

### 2.1 基本假设

1. **下平台固定在船上**：完全跟随船体运动，无相对运动
2. **活动杆长度不变**：忽略主动控制带来的杆长变化（扰动分析时）
3. **上平台被"拖动"**：由于杆长约束，上平台跟随下平台运动

### 2.2 运动传递关系

根据上述假设，活动杆长度保持不变：

$$
L_i = \text{constant}, \quad i = 1, 2, 3
$$

当活动杆长度不变时，下平台的运动直接传递到上平台：

$$
\boxed{
\begin{aligned}
q_u &= q_s + q_c \\
\dot{q}_u &= \dot{q}_s + \dot{q}_c \\
\ddot{q}_u &= \ddot{q}_s + \ddot{q}_c
\end{aligned}
}
$$

其中：
- $q_c$ 为Stewart平台的主动控制量（可调节）
- $q_s$ 为船体运动（不可控的扰动）

### 2.3 上平台受到的等效扰动

**关键结论**：上平台受到的扰动就是船体运动的**负值**（如果完全抵消）

$$
\boxed{
\text{扰动} = -q_s \quad \text{或} \quad -k_s q_s \quad (0 < k_s \leq 1)
}
$$

完整表达：

$$
\boxed{
\begin{aligned}
\text{位置扰动} & : & \Delta q_{pos} & = q_s \\
\text{速度扰动} & : & \Delta q_{vel} & = q_{sd} \\
\text{加速度扰动} & : & \Delta q_{acc} & = q_{sdd}
\end{aligned}
}
$$

---

## 3. 海浪扰动生成模型

### 3.1 频域模型：波面 → 船体运动

使用**ITTC谱**和**MSS运动RAO**：

$$
\boxed{
S_{\eta}(\omega) = \frac{173 H_s^2}{T_1^4 \omega^5} \exp\left(-\frac{691}{T_1^4 \omega^4}\right)
}
$$

其中：
- $H_s$ ：有义波高 (m)
- $T_1$ ：平均周期 (s)
- $\omega$ ：波浪频率 (rad/s)

船体运动通过**运动响应幅值算子 (RAO)** 传递：

$$
\boxed{
X_s(\omega) = \text{RAO}(\omega) \cdot \eta(\omega)
}
$$

其中：
- $X_s$ ：船体运动幅值（位移）
- $\text{RAO}$ ：运动响应幅值算子 (m/m 或 rad/m)
- $\eta$ ：波面幅值

### 3.2 惯性力（可选：力层面扰动）

如果需要计算扰动力：

$$
\boxed{
F_{dist} = M_{platform} \cdot \ddot{q}_s
}
$$

其中 $M_{platform}$ 为平台等效质量矩阵：

$$
M_{platform} = \text{diag}(m, I_{xx}, I_{yy})
$$

### 3.3 时域合成

将各频率分量叠加得到时域信号：

$$
\boxed{
\begin{aligned}
q_s(t) &= \sum_{k=1}^{N} A_k \cdot \text{RAO}_k \cdot \cos(\omega_k t + \phi_k) \\
q_{sd}(t) &= \frac{dq_s}{dt} \\
q_{sdd}(t) &= \frac{d^2q_s}{dt^2}
\end{aligned}
}
$$

其中：
- $N$ ：频率分量数（通常取50）
- $A_k = \sqrt{2 S_{\eta}(\omega_k) \Delta\omega}$ ：波幅
- $\phi_k$ ：随机相位

---

## 4. 扰动在控制中的使用

### 4.1 控制目标

让上平台保持稳定（相对于惯性坐标系）：

$$
q_u \to 0 \quad \text{或} \quad q_u \approx 0
$$

### 4.2 补偿量计算

由运动传递关系：

$$
q_u = q_s + q_c = 0 \quad \Rightarrow \quad q_c = -q_s
$$

**完全补偿**：

$$
\boxed{
q_c = -q_s, \quad \dot{q}_c = -\dot{q}_s, \quad \ddot{q}_c = -\ddot{q}_s
}
$$

**部分补偿**（更保守）：

$$
\boxed{
q_c = -k \cdot q_s, \quad 0 < k \leq 1
}
$$

### 4.3 奖励函数设计

RL环境中，扰动可直接用于奖励计算：

```python
# 状态误差 = 上平台实际位置 - 船体扰动位置
error = q_upper - q_ship  # 即 q_u - q_s

# 或者
error = q_upper + q_compensation  # 补偿后的误差
```

---

## 5. 两种扰动模型对比

| 层面 | 方法 | 公式 | 优点 | 缺点 |
|------|------|------|------|------|
| **力** | 动力学补偿 | $q_{dd} = M^{-1}(\tau - C\dot{q} - G - F_{dist})$ | 与现有动力学模型一致 | 需要求逆矩阵 |
| **运动学** | 运动传递 | $q_{perturb} = q_s$ | 直观、物理清晰、不需矩阵求逆 | 忽略动力学效应 |

---

## 6. 简化模型总结

### 最终公式

$$
\boxed{
\begin{aligned}
\text{上平台位移} & : & q_u(t) = q_s(t) + q_c(t) \\
\text{上平台速度} & : & \dot{q}_u(t) = \dot{q}_s(t) + \dot{q}_c(t) \\
\text{上平台加速度} & : & \ddot{q}_u(t) = \ddot{q}_s(t) + \ddot{q}_c(t) \\
\text{扰动量} & : & \Delta q(t) = q_s(t) \\
\text{补偿量} & : & q_c(t) = -q_s(t)
\end{aligned}
}
$$

### 实现流程

```
1. WaveDisturbance.generate_ship_motion(t)
   └─→ 返回: {position, velocity, acceleration}

2. 环境中:
   ├─ ship = wave.generate_ship_motion(time)
   ├─ perturbation = ship['position']  # 即 Δq
   └─ control_target = -ship['position']  # 补偿量
```

---

## 7. 参考文献

1. ITTC - Recommended Procedures and Guidelines: "Sea Keeping"
2. Fossen, T. I. (2021). Handbook of Marine Craft Hydrodynamics and Motion Control
3. MSS (Marine Systems Simulator) - www.marine-technology.cn
