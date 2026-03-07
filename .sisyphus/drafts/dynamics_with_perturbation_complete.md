# 船载Stewart平台动力学方程（含扰动）- 完整版

---

## 1. 系统描述

### 1.1 自由度定义

本项目采用 **3-UPS/PU** 并联平台结构：

| 索引 | 符号 | 含义 | 单位 |
|------|------|------|------|
| q[0] | $z$ | 升沉位移 | m |
| q[1] | $\alpha$ | 横滚角 (Roll) | rad |
| q[2] | $\beta$ | 俯仰角 (Pitch) | rad |

> **注意**：3-UPS/PU 平台的中心是被动支链（P），不参与主动控制。

### 1.2 坐标定义

| 符号 | 含义 | 说明 |
|------|------|------|
| $q_s = [z_s, \alpha_s, \beta_s]^T$ | 船体运动（扰动） | 下平台跟随船体 |
| $q_c = [z_c, \alpha_c, \beta_c]^T$ | Stewart平台控制量 | 主动支链产生 |
| $q_u = [z_u, \alpha_u, \beta_u]^T$ | 上平台运动 | $q_u = q_s + q_c$ |

---

## 2. 船体运动与RAO

### 2.1 RAO基本概念

**RAO（Response Amplitude Operator）** 是船舶运动响应幅值算子，表示单位波幅引起的船体运动：

$$X(\omega) = \text{RAO}(\omega) \cdot \eta$$

其中：
- $\eta$ ：波面幅值 (m)
- $X(\omega)$ ：船体运动幅值 (m 或 rad)
- $\text{RAO}(\omega)$ ：运动RAO (m/m 或 rad/m)

### 2.2 物理链路（完整）

```
波面 η(t)
    → [频率谱] → 各频率分量 η(ω)
    → [运动RAO] → 船体运动幅值 X(ω) = RAO(ω) × η(ω)
    → [运动学] → 速度 v(ω) = jω × X(ω)
    → [运动学] → 加速度 a(ω) = (jω)² × X(ω) = -ω² × X(ω)
    → [牛顿定律] → 惯性力 F(ω) = m × a(ω)
```

### 2.3 从RAO生成船体运动（时间域）

#### 2.3.1 单频分量

对于单一频率 $\omega_k$ 和相位 $\phi_k$：

$$\eta_k(t) = A_k \cos(\omega_k t + \phi_k)$$

其中波幅 $A_k = \sqrt{2 S(\omega_k) \Delta\omega}$

船体运动：

$$X_k(t) = \text{RAO}(\omega_k) \cdot \eta_k(t) = \text{RAO}(\omega_k) \cdot A_k \cos(\omega_k t + \phi_k)$$

速度：

$$\dot{X}_k(t) = -\text{RAO}(\omega_k) \cdot A_k \cdot \omega_k \sin(\omega_k t + \phi_k)$$

加速度：

$$\ddot{X}_k(t) = -\text{RAO}(\omega_k) \cdot A_k \cdot \omega_k^2 \cos(\omega_k t + \phi_k)$$

#### 2.3.2 多频叠加（完整求和）

$$q_s(t) = \sum_{k=1}^{N} \text{RAO}(\omega_k) \cdot A_k \cos(\omega_k t + \phi_k)$$

$$\dot{q}_s(t) = \sum_{k=1}^{N} \text{RAO}(\omega_k) \cdot A_k \cdot (-\omega_k) \sin(\omega_k t + \phi_k)$$

$$\ddot{q}_s(t) = \sum_{k=1}^{N} \text{RAO}(\omega_k) \cdot A_k \cdot (-\omega_k^2) \cos(\omega_k t + \phi_k)$$

---

## 3. 扰动计算详细推导

### 3.1 运动传递关系

上平台跟随下平台（船体），通过运动学约束：

$$q_u = q_s + q_c$$

$$\dot{q}_u = \dot{q}_s + \dot{q}_c$$

$$\ddot{q}_u = \ddot{q}_s + \ddot{q}_c$$

其中：
- $q_s$ ：船体运动（扰动输入）
- $q_c$ ：Stewart平台控制量
- $q_u$ ：上平台实际运动

### 3.2 上平台动力学方程

标准拉格朗日形式：

$$M(q_u)\ddot{q}_u + C(q_u, \dot{q}_u)\dot{q}_u + G(q_u) = \tau_u$$

其中 $\tau_u = J^T \tau_{legs}$ 是等效到任务空间的驱动力矩。

### 3.3 质量矩阵 $M(q_u)$

$$M(q_u) = \begin{bmatrix} 
m_{total} & 0 & 0 \\
0 & I_{xx} & 0 \\
0 & 0 & I_{yy}\cos^2\alpha + I_{zz}\sin^2\alpha
\end{bmatrix}$$

其中 $m_{total} = M_{\text{platform}} + M_{\text{rods}}$ = 平台质量 + 杆等效质量

### 3.4 科里奥利矩阵 $C(q_u, \dot{q}_u)$

令 $s_\alpha = \sin\alpha, c_\alpha = \cos\alpha, K = (I_{zz} - I_{yy})s_\alpha c_\alpha$：

$$C = \begin{bmatrix}
0 & 0 & 0 \\
0 & 0 & -K\dot{\beta} \\
0 & K\dot{\beta} & K\dot{\alpha}
\end{bmatrix}$$

### 3.5 重力向量 $G(q_u)$

$$G(q_u) = \begin{bmatrix} m_{total} g \\ 0 \\ 0 \end{bmatrix}$$

---

## 4. 扰动项完整推导

### 4.1 问题描述

已知船体运动 $q_s, \dot{q}_s, \ddot{q}_s$，求扰动对平台动力学的影响。

### 4.2 直接代入法

将 $q_u = q_s + q_c$ 代入动力学方程：

$$M(q_u)(\ddot{q}_s + \ddot{q}_c) + C(q_u, \dot{q}_s + \dot{q}_c)(\dot{q}_s + \dot{q}_c) + G(q_u) = \tau_u$$

展开：

$$M\ddot{q}_c + C\dot{q}_c + G + M\ddot{q}_s + C\dot{q}_s + C_{cs}\dot{q}_s = \tau_u$$

整理得：

$$M\ddot{q}_c + C\dot{q}_c + G = \tau_u - M\ddot{q}_s - C\dot{q}_s$$

### 4.3 扰动项定义

$$\tau_{\text{dist}} = -M(q_u)\ddot{q}_s - C(q_u, \dot{q}_u)\dot{q}_s$$

> **注意**：扰动项不含重力项$G$。重力项保留在动力学方程左侧。

### 4.4 扰动项展开（不简化）

#### （1）加速度项（惯性力）

$$-M\ddot{q}_s = -\begin{bmatrix}
m_{total} & 0 & 0 \\
0 & I_{xx} & 0 \\
0 & 0 & I_{eq}
\end{bmatrix}
\begin{bmatrix}
\ddot{z}_s \\
\ddot{\alpha}_s \\
\ddot{\beta}_s
\end{bmatrix}
= \begin{bmatrix}
-m_{total}\ddot{z}_s \\
-I_{xx}\ddot{\alpha}_s \\
-I_{eq}\ddot{\beta}_s
\end{bmatrix}$$

其中 $I_{eq} = I_{yy}\cos^2\alpha + I_{zz}\sin^2\alpha$

#### （2）速度项（科里奥利力）

$$-C\dot{q}_s = -\begin{bmatrix}
0 & 0 & 0 \\
0 & 0 & -K\dot{\beta}_s \\
0 & K\dot{\beta}_s & K\dot{\alpha}_s
\end{bmatrix}
\begin{bmatrix}
\dot{z}_s \\
\dot{\alpha}_s \\
\dot{\beta}_s
\end{bmatrix}
= \begin{bmatrix}
0 \\
K\dot{\alpha}_s\dot{\beta}_s \\
-K\dot{\beta}_s^2
\end{bmatrix}$$

其中 $K = (I_{zz} - I_{yy})\sin\alpha\cos\alpha$

#### （3）完整扰动项（不含重力）

$$\tau_{\text{dist}} = -M\ddot{q}_s - C\dot{q}_s = \begin{bmatrix}
-m_{total}\ddot{z}_s \\
-I_{xx}\ddot{\alpha}_s + K\dot{\alpha}_s\dot{\beta}_s \\
-I_{eq}\ddot{\beta}_s - K\dot{\beta}_s^2
\end{bmatrix}$$

---

## 5. 最终动力学方程

### 5.1 完整形式

$$\boxed{M(q_u)\ddot{q}_c + C(q_u, \dot{q}_u)\dot{q}_c + G(q_u) = \tau_u + \tau_{\text{dist}}}$$

其中 $\tau_{\text{dist}} = -M(q_u)\ddot{q}_s - C(q_u, \dot{q}_u)\dot{q}_s$（不含重力）。

### 5.2 移项后形式

$$\boxed{M(q_u)\ddot{q}_c + C(q_u, \dot{q}_u)\dot{q}_c + G(q_u) = \tau_u - M(q_u)\ddot{q}_s - C(q_u, \dot{q}_u)\dot{q}_s}$$

### 5.3 分量形式

$$
m_{total} \ddot{z}_c = \tau_z - m_{total}\ddot{z}_s - (C_{11}\dot{z}_u + C_{12}\dot{\alpha}_u + C_{13}\dot{\beta}_u) - m_{total}g
$$

$$
I_{xx} \ddot{\alpha}_c = \tau_\alpha - I_{xx}\ddot{\alpha}_s + K\dot{\alpha}_s\dot{\beta}_s - (C_{21}\dot{z}_u + C_{22}\dot{\alpha}_u + C_{23}\dot{\beta}_u)
$$

$$
I_{eq} \ddot{\beta}_c = \tau_\beta - I_{eq}\ddot{\beta}_s - K\dot{\beta}_s^2 - (C_{31}\dot{z}_u + C_{32}\dot{\alpha}_u + C_{33}\dot{\beta}_u)
$$

## 6. 扰动计算总结

### 6.1 扰动输入

已知船体运动（通过RAO计算）：
- 位置：$q_s = [z_s, \alpha_s, \beta_s]^T$
- 速度：$\dot{q}_s = [\dot{z}_s, \dot{\alpha}_s, \dot{\beta}_s]^T$
- 加速度：$\ddot{q}_s = [\ddot{z}_s, \ddot{\alpha}_s, \ddot{\beta}_s]^T$

### 6.2 扰动输出

广义力扰动（任务空间）：

$$\tau_{\text{dist}} = \begin{bmatrix} 
\tau_{dist,z} \\ 
\tau_{dist,\alpha} \\ 
\tau_{dist,\beta} 
\end{bmatrix} = \begin{bmatrix}
-m_{total} \ddot{z}_s \\
-I_{xx} \ddot{\alpha}_s + K\dot{\alpha}_s\dot{\beta}_s \\
-I_{eq} \ddot{\beta}_s - K\dot{\beta}_s^2
\end{bmatrix}$$

其中：
- $I_{eq} = I_{yy}\cos^2\alpha + I_{zz}\sin^2\alpha$
- $K = (I_{zz} - I_{yy})\sin\alpha\cos\alpha$

### 6.3 物理意义

| 扰动分量 | 表达式 | 物理来源 |
|----------|--------|----------|
| $\tau_{dist,z}$ | $-m_{total}\ddot{z}_s$ | 船体升沉加速度产生的惯性力 |
| $\tau_{dist,\alpha}$ | $-I_{xx}\ddot{\alpha}_s + K\dot{\alpha}_s\dot{\beta}_s$ | 船体横滚加速度惯性力 + 科里奥利力 |
| $\tau_{dist,\beta}$ | $-I_{eq}\ddot{\beta}_s - K\dot{\beta}_s^2$ | 船体俯仰加速度惯性力 + 离心力 |

## 7. 仿真中扰动的引入方法

### 7.1 核心思想

**关键洞察**：不需要显式计算扰动力 $\tau_{dist}$，只需将船体运动直接叠加到上平台状态即可。

**物理含义**：
- 上平台位置：$q_u = q_c + q_s$
- 上平台速度：$\dot{q}_u = \dot{q}_c + \dot{q}_s$
- 上平台加速度：$\ddot{q}_u = \ddot{q}_c + \ddot{q}_s$

动力学方程：
$$M(q_u)\ddot{q}_u + C(q_u, \dot{q}_u)\dot{q}_u + G(q_u) = \tau_u$$

展开：
$$M(q_u)(\ddot{q}_c + \ddot{q}_s) + C(q_u, \dot{q}_c + \dot{q}_s)(\dot{q}_c + \dot{q}_s) + G = \tau_u$$

求解 $\ddot{q}_c$：
$$\ddot{q}_c = M^{-1}(\tau_u - C\dot{q}_u - G) - \ddot{q}_s$$

### 7.2 简化步骤流程

```
┌─────────────────────────────────────────────────────────────┐
│                    仿真循环                                   │
├─────────────────────────────────────────────────────────────┤
│  1. 输入：当前时刻 t                                        │
│       ↓                                                    │
│  2. 获取船体运动（RAO）：                                  │
│       q_s(t), qd_s(t), qdd_s(t)                          │
│       ↓                                                    │
│  3. 计算上平台状态：                                        │
│       q_u = q_c + q_s                                      │
│       qd_u = qd_c + qd_s                                  │
│       ↓                                                    │
│  4. 求解动力学方程：                                         │
│       qdd_c = M⁻¹(τ_u - C·qd_u - G) - qdd_s             │
│       ↓                                                    │
│  5. 数值积分：                                              │
│       qd_c += qdd_c · dt                                   │
│       q_c  += qd_c  · dt                                   │
└─────────────────────────────────────────────────────────────┘
```

### 7.3 船体运动生成步骤

**在仿真前预计算**：

1. 读取RAO数据（频率响应曲线）
2. 生成频率谱（ITTC双参数谱）
3. 叠加N个频率分量：
   - 位置：$q_s(t) = \sum_k \text{RAO}(\omega_k) \cdot A_k \cos(\omega_k t + \phi_k)$
   - 速度：$\dot{q}_s(t) = \sum_k \text{RAO}(\omega_k) \cdot A_k \cdot (-\omega_k) \sin(\omega_k t + \phi_k)$
   - 加速度：$\ddot{q}_s(t) = \sum_k \text{RAO}(\omega_k) \cdot A_k \cdot (-\omega_k^2) \cos(\omega_k t + \phi_k)$

**每仿真步计算**：

1. 采样：$q_s = q_s(t)$, $qd_s = qd_s(t)$, $qdd_s = qdd_s(t)$
2. 上平台状态：$q_u = q_c + q_s$, $qd_u = qd_c + qd_s$
3. 质量矩阵：$M = M(q_u)$
4. 科里奥利矩阵：$C = C(q_u, qd_u)$
5. 重力向量：$G = G(q_u)$
6. 求解加速度：$
qdd_c = M^{-1}(\tau_u - C \cdot qd_u - G) - qdd_s$
7. 数值积分：$qd_c += qdd_c \cdot dt$, $q_c += qd_c \cdot dt$

### 7.4 参数说明

| 参数 | 符号 | 说明 |
|------|------|------|
| $q_s$ | 船体位移 | $z_s, \alpha_s, \beta_s$ |
| $qd_s$ | 船体速度 | $\dot{z}_s, \dot{\alpha}_s, \dot{\beta}_s$ |
| $qdd_s$ | 船体加速度 | $\ddot{z}_s, \ddot{\alpha}_s, \ddot{\beta}_s$ |
| $q_c$ | 平台控制量 | 主动支链产生的位移 |
| $q_u$ | 上平台总位移 | $q_c + q_s$ |
| $M$ | 质量矩阵 | 与上平台状态相关 |
| $C$ | 科里奥利矩阵 | 与上平台状态和速度相关 |
| $G$ | 重力向量 | $[m_{total}g, 0, 0]^T$ |
PK|| $\tau_u$ | 控制力 | $J^T \tau_{legs}$ |

---

## 8. 与简化版本的对比

| 版本 | 扰动表达式 | 适用场景 |
|------|------------|----------|
| **完整版** | $\tau_{dist} = -M\ddot{q}_s - C\dot{q}_s$ | 高速、大幅运动 |
| **简化版** | 直接叠加 $q_s, \dot{q}_s, \ddot{q}_s$ | 推荐使用 |

---

## 9. 参考文献

1. Fossen, T. I. (2021). Handbook of Marine Craft Hydrodynamics and Motion Control. Wiley.
2. MSS (Marine Systems Simulator) - 运动RAO数据
3. Spong, M. W., & Vidyasagar, M. (2008). Robot Dynamics and Control
4. 本项目: simulation/common/platform_dynamics.py
5. 本项目: simulation/common/wave_disturbance.py

