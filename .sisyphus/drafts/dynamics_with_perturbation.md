# 船载Stewart平台动力学方程（含扰动）

## 1. 系统描述

### 1.1 坐标定义

| 符号 | 含义 | 维度 |
|------|------|------|
| $q_s = [z_s, \alpha_s, \beta_s]^T$ | 船体运动（扰动） | 3×1 |
| $q_c = [z_c, \alpha_c, \beta_c]^T$ | Stewart平台控制量 | 3×1 |
| $q_u = [z_u, \alpha_u, \beta_u]^T$ | 上平台运动（输出） | 3×1 |

### 1.2 运动传递关系

$$
\boxed{
\begin{aligned}
q_u &= q_s + q_c \\
\dot{q}_u &= \dot{q}_s + \dot{q}_c \\
\ddot{q}_u &= \ddot{q}_s + \ddot{q}_c
\end{aligned}}
$$

---

## 2. 上平台动力学方程

### 2.1 标准动力学形式

$$M(q_u)\ddot{q}_u + C(q_u, \dot{q}_u)\dot{q}_u + G(q_u) = \tau_u$$

其中：
- $M(q_u)$ ：质量矩阵
- $C(q_u, \dot{q}_u)$ ：科里奥利矩阵
- $G(q_u)$ ：重力向量
- $\tau_u$ ：驱动力矩

### 2.2 质量矩阵

$$
M(q_u) = \begin{bmatrix}
m_p & 0 & 0 \\
0 & I_{xx} & 0 \\
0 & 0 & I_{yy}\cos^2\alpha_u + I_{zz}\sin^2\alpha_u
\end{bmatrix}
$$

简化（忽略耦合项）：

$$
M(q_u) \approx \begin{bmatrix}
m_p & 0 & 0 \\
0 & I_{xx} & 0 \\
0 & 0 & I_{yy}
\end{bmatrix} = \text{diag}(m_p, I_{xx}, I_{yy})
$$

### 2.3 科里奥利矩阵

$$
C(q_u, \dot{q}_u) = \begin{bmatrix}
0 & 0 & 0 \\
0 & 0 & 0 \\
0 & -(I_{yy}-I_{zz})\dot{\alpha}_u\sin\alpha_u\cos\alpha_u & 0
\end{bmatrix}
$$

简化（忽略非线性项）：

$$
C(q_u, \dot{q}_u) \approx 0
$$

### 2.4 重力向量

$$
G(q_u) = \begin{bmatrix}
m_p g \\ 0 \\ 0
\end{bmatrix}
$$

---

## 3. 扰动引入形式

### 3.1 扰动来源

扰动来自船体运动，通过运动学约束传递到上平台：

$$
\ddot{q}_u = \ddot{q}_s + \ddot{q}_c
$$

将 $\ddot{q}_s$ 作为扰动加速度引入动力学方程。

### 3.2 带扰动的动力学方程

将 $q_u = q_s + q_c$ 代入标准动力学方程：

$$M(q_u)\ddot{q}_u + C(q_u, \dot{q}_u)\dot{q}_u + G(q_u) = \tau_u$$

展开 $\ddot{q}_u$：

$$M(q_u)(\ddot{q}_s + \ddot{q}_c) + C(q_u, \dot{q}_u)(\dot{q}_s + \dot{q}_c) + G(q_u) = \tau_u$$

整理为**关于控制量 $q_c$ 的方程**：

$$\boxed{
M(q_u)\ddot{q}_c + C(q_u, \dot{q}_u)\dot{q}_c + G(q_u) = \tau_u - M(q_u)\ddot{q}_s - C(q_u, \dot{q}_u)\dot{q}_s
}$$

### 3.3 扰动项定义

$$
\boxed{
\tau_{dist} = -M(q_u)\ddot{q}_s - C(q_u, \dot{q}_u)\dot{q}_s
}
$$

展开：

$$
\tau_{dist} = -\begin{bmatrix}
m_p & 0 & 0 \\
0 & I_{xx} & 0 \\
0 & 0 & I_{yy}
\end{bmatrix} \begin{bmatrix}
\ddot{z}_s \\ \ddot{\alpha}_s \\ \ddot{\beta}_s
\end{bmatrix} - \begin{bmatrix}
0 \\ 0 \\ C_{31}\dot{q}_s
\end{bmatrix}
$$

简化（忽略科里奥利项）：

$$
\boxed{
\tau_{dist} \approx -M(q_u)\ddot{q}_s = \begin{bmatrix}
-m_p \ddot{z}_s \\ -I_{xx} \ddot{\alpha}_s \\ -I_{yy} \ddot{\beta}_s
\end{bmatrix}}
$$

---

## 4. 最终动力学方程（含扰动）

### 4.1 标准形式

$$\boxed{
M(q_u)\ddot{q}_c + C(q_u, \dot{q}_u)\dot{q}_c + G(q_u) = \tau_u + \tau_{dist}
}$$

或写作：

$$\boxed{
M(q_u)\ddot{q}_c + C(q_u, \dot{q}_u)\dot{q}_c + G(q_u) = \tau_u - M(q_u)\ddot{q}_s - C\dot{q}_s
}$$

### 4.2 简化形式（忽略非线性项）

$$\boxed{
m_p \ddot{z}_c = \tau_z - m_p \ddot{z}_s \\
I_{xx} \ddot{\alpha}_c = \tau_{\alpha} - I_{xx} \ddot{\alpha}_s \\
I_{yy} \ddot{\beta}_c = \tau_{\beta} - I_{yy} \ddot{\beta}_s
}$$

### 4.3 扰动等效为位置/速度项

如果将扰动等效为位置和速度的函数：

$$\tau_{dist} = -K_d \dot{q}_s - K_p q_s$$

其中：
- $K_d$ ：速度扰动增益（等效阻尼）
- $K_p$ ：位置扰动增益（等效刚度）

这等价于添加了一个"虚拟弹簧-阻尼"连接。

---

## 5. 离散时间形式（数值仿真）

### 5.1 欧拉积分

```python
# 给定当前状态 q_c, qd_c 和输入 tau_u
# 以及扰动 q_s, qd_s, qdd_s

# 1. 计算扰动力矩
tau_dist = -M(q_u) @ qdd_s  # 简化形式

# 2. 净力矩
tau_net = tau_u + tau_dist - G(q_u) - C(q_u, qd_u) @ qd_c

# 3. 计算控制加速度
qdd_c = M_inv @ tau_net

# 4. 积分得到控制速度和位置
qd_c = qd_c + qdd_c * dt
q_c = q_c + qd_c * dt

# 5. 上平台实际运动
q_u = q_s + q_c
qd_u = qd_s + qd_c
qdd_u = qdd_s + qdd_c
```

### 5.2 状态空间形式

定义状态 $x = [q_c^T, \dot{q}_c^T]^T$：

$$
\boxed{
\dot{x} = \begin{bmatrix}
0 & I \\
-M^{-1}K_p & -M^{-1}C
\end{bmatrix} x + \begin{bmatrix}
0 \\ M^{-1}
\end{bmatrix} (\tau_u + \tau_{dist}) + \begin{bmatrix}
0 \\ -I
\end{bmatrix} \ddot{q}_s
}
$$

---

## 6. 与RL环境的结合

### 6.1 状态定义

```python
# RL观测状态
state = [
    # 上平台运动
    q_u[0], q_u[1], q_u[2],      # z, alpha, beta
    qd_u[0], qd_u[1], qd_u[2],   # z_dot, alpha_dot, beta_dot
    
    # 目标位置
    q_des[0], q_des[1], q_des[2],
    qd_des[0], qd_des[1], qd_des[2],
    
    # 扰动信息（可选，作为状态的一部分）
    q_s[0], q_s[1], q_s[2],       # 船体位移
    qd_s[0], qd_s[1], qd_s[2],   # 船体速度
    # qdd_s[0], qdd_s[1], qdd_s[2]  # 船体加速度（可选）
]
```

### 6.2 奖励函数设计

```python
# 核心目标：上平台保持稳定 (q_u ≈ 0)
# 即 q_s + q_c ≈ 0 或 q_c ≈ -q_s

# 方案1：直接用上平台误差
error = q_u - q_target  # q_target = 0
reward = -k1 * ||error|| - k2 * ||qd_u||

# 方案2：用控制量补偿船体运动
error = q_c + q_s  # 理想应为0
reward = -k1 * ||error|| - k2 * ||qd_c|| - k3 * ||u||

# 方案3：综合
error_u = q_u  # 上平台偏离0
error_c = q_c + q_s  # 补偿是否到位
reward = -k1*||error_u|| - k2*||error_c|| - k3*||u||
```

---

## 7. 完整方程总结

### 最终形式

$$\boxed{
M\ddot{q}_c + C\dot{q}_c + G = \tau_u - M\ddot{q}_s - C\dot{q}_s
}$$

或

$$\boxed{
M(\ddot{q}_u - \ddot{q}_s) + C(\dot{q}_u - \dot{q}_s) + G = \tau_u
}$$

### 物理意义

- 左边：Stewart平台自身动力学（控制产生的加速度）
- 右边：驱动力矩 - 扰动力矩（船体运动引起的等效力矩）

### 特殊情况

如果忽略 $C$ 和 $G$：

$$\boxed{
\ddot{q}_c = M^{-1}\tau_u - \ddot{q}_s
}$$

即：

$$\boxed{
\ddot{q}_u = \ddot{q}_c + \ddot{q}_s = M^{-1}\tau_u
}$$

这说明：**只要驱动力能产生足够的加速度，上平台就能稳定**。

---

## 8. 参考文献

1. Spong, M. W., & Vidyasagar, M. (2008). Robot Dynamics and Control
2. Fossen, T. I. (2021). Handbook of Marine Craft Hydrodynamics and Motion Control
3. Siciliano, B., et al. (2010). Robotics: Modelling, Planning and Control
