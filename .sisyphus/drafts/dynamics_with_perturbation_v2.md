# 船载Stewart平台动力学方程（含扰动）

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

### 1.3 运动传递关系

$$
\begin{aligned}
q_u &= q_s + q_c \\
\dot{q}_u &= \dot{q}_s + \dot{q}_c \\
\ddot{q}_u &= \ddot{q}_s + \ddot{q}_c
\end{aligned}
$$

---

## 2. 上平台动力学方程

### 2.1 标准拉格朗日形式

$$M(q_u)\ddot{q}_u + C(q_u, \dot{q}_u)\dot{q}_u + G(q_u) = \tau_u$$

其中 $\tau_u = J^T \tau_{legs}$ 是等效到任务空间的驱动力矩。

### 2.2 雅可比矩阵 $J$

#### 2.2.1 物理意义

雅可比矩阵 $J$ 描述了**任务空间速度**到**杆伸缩速度**的映射：

$$\boxed{\dot{l} = J \cdot \dot{q}}$$

其中：
- $\dot{l} = [\dot{l}_1, \dot{l}_2, \dot{l}_3]^T$ - 三个杆的伸缩速度
- $\dot{q} = [\dot{z}, \dot{\alpha}, \dot{\beta}]^T$ - 任务空间速度

同时描述了**虚功原理**下的力映射关系：

$$\boxed{\tau_u = J^T \cdot \tau_{legs}}$$

其中：
- $\tau_{legs} = [F_1, F_2, F_3]^T$ - 三个电动缸的推力
- $\tau_u = [F_z, M_\alpha, M_\beta]^T$ - 任务空间等效力/力矩

#### 2.2.2 计算公式

**几何关系**：

$$l_i = ||P_i - B_i||$$

其中：
- $P_i$ - 动平台铰链在惯性系中的位置
- $B_i$ - 基座铰链在惯性系中的位置
- $l_i$ - 第 $i$ 根杆的长度

**雅可比矩阵元素**：

$$J_{i,j} = \frac{\partial l_i}{\partial q_j} = \frac{\partial ||P_i - B_i||}{\partial q_j} = \frac{(P_i - B_i) \cdot \partial P_i / \partial q_j}{||P_i - B_i||}$$

展开为：

$$J = \begin{bmatrix}
\frac{\partial l_1}{\partial z} & \frac{\partial l_1}{\partial \alpha} & \frac{\partial l_1}{\partial \beta} \\
\frac{\partial l_2}{\partial z} & \frac{\partial l_2}{\partial \alpha} & \frac{\partial l_2}{\partial \beta} \\
\frac{\partial l_3}{\partial z} & \frac{\partial l_3}{\partial \alpha} & \frac{\partial l_3}{\partial \beta}
\end{bmatrix}$$

#### 2.2.3 具体计算

**第 $i$ 根杆的方向向量**（惯性系）：

$$e_i = \frac{P_i - B_i}{||P_i - B_i||} = [e_{ix}, e_{iy}, e_{iz}]^T$$

**雅可比矩阵元素**：

$$\begin{aligned}
J_{i,0} &= e_{iz} & &(\text{对 } z \text{ 的偏导})\\
J_{i,1} &= e_i \cdot (R \cdot \frac{\partial r_{local}}{\partial \alpha}) & &(\text{对 } \alpha \text{ 的偏导})\\
J_{i,2} &= e_i \cdot (R \cdot \frac{\partial r_{local}}{\partial \beta}) & &(\text{对 } \beta \text{ 的偏导})
\end{aligned}$$

其中 $R$ 是旋转矩阵，$r_{local}$ 是动平台铰链的局部坐标。

#### 2.2.4 $P_i$ 对 $q_j$ 的偏导说明

**问题**：$P_i$ 明明是3维向量，怎么对标量 $q_j$ 求偏导？

**回答**：
- $P_i$ = $[P_{ix}, P_{iy}, P_{iz}]^T$ 是**3维向量**
- $q_j$ = $z, \alpha, \beta$ 是**标量**
- $\frac{\partial P_i}{\partial q_j}$ 是**3维向量**（每个分量对标量的偏导）

**使用链式法则**：

$$\frac{\partial ||P_i - B_i||}{\partial q_j} = \frac{(P_i - B_i) \cdot \frac{\partial P_i}{\partial q_j}}{||P_i - B_i||}$$

其中：
- $P_i - B_i$ 是杆矢量（3维）
- $\frac{\partial P_i}{\partial q_j}$ 也是3维
- 点积 → 标量

**$P_i$ 对各自由度的偏导**：

| 偏导 | 含义 | 计算 |
|------|------|------|
| $\frac{\partial P_i}{\partial z}$ | 升沉方向的变化 | $\begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$ |
| $\frac{\partial P_i}{\partial \alpha}$ | 横滚角引起的变化 | $\frac{\partial R}{\partial \alpha} \cdot r_{local}$ |
| $\frac{\partial P_i}{\partial \beta}$ | 俯仰角引起的变化 | $\frac{\partial R}{\partial \beta} \cdot r_{local}$ |

**旋转矩阵的导数**：

$$R = R_y(\beta)R_x(\alpha)$$

对 $\alpha$ 求导：

$$\frac{\partial R}{\partial \alpha} = R_y(\beta) \cdot \frac{\partial R_x(\alpha)}{\partial \alpha}$$

其中 $R_x(\alpha)$ 是绕X轴的旋转矩阵。

#### 2.2.5 坐标系说明

**$J$ 是在惯性系（全局坐标系）下计算的**：
- 基座铰链位置 $B_i$：在惯性系中（$z=0$）
- 动平台铰链位置 $P_i$：在惯性系中（随平台运动）
- 杆矢量 $P_i - B_i$：在惯性系中

#### 2.2.6 代码实现

```python
def jacobian(self, q):
    """
    计算雅可比矩阵 J: l_dot = J @ q_dot
    """
    # 获取动平台铰链的全局位置
    platform_joints = self.platform_joints_global(q)  # [3x3] 矩阵
    
    J = np.zeros((3, 3))
    
    for i in range(3):
        # 杆矢量（惯性系）
        leg_vec = platform_joints[i] - self.base_joints[i]
        leg_unit = leg_vec / np.linalg.norm(leg_vec)
        
        # J[i, 0] = ∂l_i/∂z（升沉方向的导数）
        J[i, 0] = leg_unit[2]
        
        # J[i, 1] = ∂l_i/∂α（横滚方向的导数）
        r_local = self.platform_joints_local[i]
        v_rot_alpha = np.array([0, -r_local[2], r_local[1]])
        v_rot_alpha_global = self.rotation_matrix(alpha, beta) @ v_rot_alpha
        J[i, 1] = np.dot(leg_unit, v_rot_alpha_global)
        
        # J[i, 2] = ∂l_i/∂β（俯仰方向的导数）
        v_rot_beta = np.array([r_local[2], 0, -r_local[0]])
        v_rot_beta_global = self.rotation_matrix(alpha, beta) @ v_rot_beta
        J[i, 2] = np.dot(leg_unit, v_rot_beta_global)
    
    return J
```

#### 2.2.7 力/力矩转换

```python
# 任务空间力 → 杆力（逆运动学）
tau_legs = np.linalg.solve(J.T, tau_u)

# 杆力 → 任务空间力（正运动学）  
tau_u = J.T @ tau_legs
```

### 2.3 质量矩阵 $M(q_u)$

$$M(q_u) = \begin{bmatrix} 
m_{total} & 0 & 0 \\
0 & I_{xx} & 0 \\
0 & 0 & I_{yy}\cos^2\alpha + I_{zz}\sin^2\alpha
\end{bmatrix}$$

其中：
- $m_{total}$ ：动平台总质量
- $I_{xx}$ ：绕X轴（roll轴）的转动惯量
- $I_{yy}$ ：绕Y轴的转动惯量
- $I_{zz}$ ：绕Z轴的转动惯量

### 2.4 科里奥利/离心力矩阵 $C(q_u, \dot{q}_u)$

令 $s_\alpha = \sin\alpha, c_\alpha = \cos\alpha, K = (I_{zz} - I_{yy})s_\alpha c_\alpha$：

$$C = \begin{bmatrix}
0 & 0 & 0 \\
0 & 0 & -K\dot{\beta} \\
0 & K\dot{\beta} & K\dot{\alpha}
\end{bmatrix}$$

### 2.5 重力向量 $G(q_u)$

重力作用在动平台上（垂直向下），通过被动支链传递给船体：

$$G(q_u) = \begin{bmatrix} m_{total} g \\ 0 \\ 0 \end{bmatrix}$$

> **物理解释**：被动支链不能产生主动力来平衡重力，但可以承受反作用力。

---

## 3. 扰动引入

### 3.1 扰动来源

扰动来自船体运动（下平台），通过运动学约束传递到上平台：

$$\ddot{q}_u = \ddot{q}_s + \ddot{q}_c$$

其中 $\ddot{q}_s$ 是船体加速度（扰动源），$\ddot{q}_c$ 是控制产生的加速度。

### 3.2 加速度 → 力/力矩 转换

**核心公式**：

$$\tau_d = M(q_u) \cdot \ddot{q}_s$$

**物理原理**：
- 平动：$F = m \cdot a$
- 转动：$M = I \cdot \alpha$

**展开形式**：

$$
\tau_d = M(q_u) \ddot{q}_s = \begin{bmatrix}
m_{total} & 0 & 0 \\
0 & I_{xx} & 0 \\
0 & 0 & I_{yy}\cos^2\alpha + I_{zz}\sin^2\alpha
\end{bmatrix} \begin{bmatrix}
\ddot{z}_s \\
\ddot{\alpha}_s \\
\ddot{\beta}_s
\end{bmatrix} = \begin{bmatrix}
m_{total} \ddot{z}_s \\
I_{xx} \ddot{\alpha}_s \\
(I_{yy}\cos^2\alpha + I_{zz}\sin^2\alpha) \ddot{\beta}_s
\end{bmatrix}
$$

> **关键**：只需将船体加速度乘以质量矩阵 $M$，就得到了等效的力/力矩 $\tau_d$。**不需要求逆矩阵**！

### 3.3 完整扰动项

包含三项：加速度项、速度项（科里奥利力）、重力项

$$\tau_{dist} = -\tau_d - C(q_u, \dot{q}_s)\dot{q}_s - G(q_u)$$

**展开**：

$$
\tau_{dist} = -\begin{bmatrix}
m_{total} \ddot{z}_s \\
I_{xx} \ddot{\alpha}_s \\
I_{eq} \ddot{\beta}_s
\end{bmatrix} - \begin{bmatrix}
0 \\
-K\dot{\beta}_s \\
K\dot{\beta}_s\dot{\alpha}_s
\end{bmatrix} - \begin{bmatrix}
m_{total}g \\
0 \\
0
\end{bmatrix}
$$

其中：
- $I_{eq} = I_{yy}\cos^2\alpha + I_{zz}\sin^2\alpha$
- $K = (I_{zz} - I_{yy})\sin\alpha\cos\alpha$

### 3.4 简化形式

忽略科里奥利速度项：

$$
\tau_{dist} \approx -M(q_u)\ddot{q}_s - G(q_u) = \begin{bmatrix}
-m_{total}(\ddot{z}_s + g) \\
-I_{xx} \ddot{\alpha}_s \\
-I_{eq} \ddot{\beta}_s
\end{bmatrix}
$$

---

## 4. 最终动力学方程

### 4.1 完整形式

$$\boxed{M(q_u)\ddot{q}_c + C(q_u, \dot{q}_u)\dot{q}_c + G(q_u) = \tau_u + \tau_{dist}}$$

其中 $\tau_{dist} = -M(q_u)\ddot{q}_s - C\dot{q}_s$。

### 4.2 物理意义

| 项 | 含义 |
|----|------|
| $M(q_u)\ddot{q}_c$ | 控制产生的惯性力 |
| $C(q_u, \dot{q}_u)\dot{q}_c$ | 控制产生的科里奥利力 |
| $G(q_u)$ | 重力 |
| $\tau_u = J^T\tau_{legs}$ | 主动支链驱动力 |
| $\tau_{dist}$ | 船体运动扰动力矩 |

### 4.3 分量形式

$$\boxed{
\begin{aligned}
m_{total} \ddot{z}_c &= \tau_z - m_{total}(\ddot{z}_s + g) \\
I_{xx} \ddot{\alpha}_c &= \tau_\alpha - I_{xx} \ddot{\alpha}_s \\
I_{eq} \ddot{\beta}_c &= \tau_\beta - I_{eq} \ddot{\beta}_s
\end{aligned}}
$$

---

## 5. 离散时间实现

```python
# 获取船体运动扰动
ship = wave.generate_ship_motion(np.array([t]))
q_s = ship['position']      # [z_s, alpha_s, beta_s]
qd_s = ship['velocity']     # [zd_s, alphad_s, betad_s]
qdd_s = ship['acceleration'] # [zdd_s, alphadd_s, betadd_s]

# 当前上平台状态
q_u = self.q + q_s         # q = q_c, 上平台 = 控制量 + 船体运动
qd_u = self.qd + qd_s

# 计算质量矩阵和科里奥利矩阵
M = self.platform.mass_matrix(q_u)
C = self.platform.coriolis_matrix(q_u, qd_u)
G = self.platform.gravity_vector(q_u)

# 计算扰动力矩
tau_dist = -M @ qdd_s - C @ qd_s - G

# 计算控制力矩（从动作空间）
tau_u = J.T @ tau_legs

# 动力学方程求解加速度
qdd_c = np.linalg.solve(M, tau_u + tau_dist - C @ qd_c)

# 数值积分
self.qd += qdd_c * dt
self.q += self.qd * dt

# 上平台实际运动
q_u = q_s + self.q
qd_u = qd_s + self.qd
```

---

## 6. 与参考文档对比

| 项 | 3dofPlatformSim | 本项目 |
|----|------------------|--------|
| 自由度 | $z, \alpha$(pitch), $\beta$(roll) | $z, \alpha$(roll), $\beta$(pitch) |
| 旋转顺序 | $R_y(\beta)R_x(\alpha)$ | 相同 |
| G向量 | $[m_{total}g, 0, 0]^T$ | $[m_{total}g, 0, 0]^T$ |

---

## 7. 总结

### 核心方程

$$\boxed{M(q_u)\ddot{q}_c + C(q_u, \dot{q}_u)\dot{q}_c = \tau_u - M(q_u)\ddot{q}_s - C\dot{q}_s - G(q_u)}$$

### 简化理解

1. **控制产生的加速度** + **船体扰动加速度** = **上平台实际加速度**
2. 扰动通过 **$-M\ddot{q}_s$** 和 **$-C\dot{q}_s$** 两项引入
3. 重力项 $G = [m_{total}g, 0, 0]^T$ 由被动支链承受反作用力

---

## 8. 参考文献

1. 3dofPlatformSim/models/dynamics_model.md
2. Spong, M. W., & Vidyasagar, M. (2008). Robot Dynamics and Control
3. Siciliano, B., et al. (2010). Robotics: Modelling, Planning and Control
