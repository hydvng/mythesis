# 3-UPS/PU 并联平台完整动力学方程（拉格朗日法推导）

---

## 0. 系统参数

### 0.1 几何参数

| 参数 | 符号 | 值 |
|------|------|-----|
| 平台半径 | $r_p$ | 0.58 m |
| 基座半径 | $r_b$ | 0.65 m |
| 铰链角度 | $\theta_i$ | $0, 2\pi/3, 4\pi/3$ |

### 0.2 质量参数

| 参数 | 符号 | 值 |
|------|------|-----|
| 平台质量 | $M$ | 347.54 kg |
| 单杆质量 | $m_i$ | 25.0 kg |
| 绕X轴惯量 | $I_{xx}$ | 60.64 kg·m² |
| 绕Y轴惯量 | $I_{yy}$ | 115.4 kg·m² |
| 绕Z轴惯量 | $I_{zz}$ | 80.0 kg·m² |

### 0.3 铰链本地坐标

**动平台铰链**（$p_{\text{local},i}$，i=1,2,3）：

$$
p_{\text{local},1} = \begin{bmatrix} r_p \\ 0 \\ 0 \end{bmatrix}
= \begin{bmatrix} 0.58 \\ 0 \\ 0 \end{bmatrix}
$$

$$
p_{\text{local},2} = \begin{bmatrix} -\frac{r_p}{2} \\ \frac{\sqrt{3}}{2}r_p \\ 0 \end{bmatrix}
= \begin{bmatrix} -0.29 \\ 0.502 \\ 0 \end{bmatrix}
$$

$$
p_{\text{local},3} = \begin{bmatrix} -\frac{r_p}{2} \\ -\frac{\sqrt{3}}{2}r_p \\ 0 \end{bmatrix}
= \begin{bmatrix} -0.29 \\ -0.502 \\ 0 \end{bmatrix}
$$

**基座铰链**（$B_i$，常数）：

$$
B_1 = \begin{bmatrix} r_b \\ 0 \\ 0 \end{bmatrix}
= \begin{bmatrix} 0.65 \\ 0 \\ 0 \end{bmatrix}
$$

$$
B_2 = \begin{bmatrix} -\frac{r_b}{2} \\ \frac{\sqrt{3}}{2}r_b \\ 0 \end{bmatrix}
= \begin{bmatrix} -0.325 \\ 0.563 \\ 0 \end{bmatrix}
$$

$$
B_3 = \begin{bmatrix} -\frac{r_b}{2} \\ -\frac{\sqrt{3}}{2}r_b \\ 0 \end{bmatrix}
= \begin{bmatrix} -0.325 \\ -0.563 \\ 0 \end{bmatrix}
$$

---

## 1. 坐标变换

### 1.1 旋转矩阵

采用 Z-Y-X 欧拉角（先 α-roll绕X轴，再 β-pitch绕Y轴）：

$$R(\alpha,\beta) = R_y(\beta) \cdot R_x(\alpha)$$

其中：

$$
R_x(\alpha) = \begin{bmatrix}
1 & 0 & 0 \\
0 & \cos\alpha & -\sin\alpha \\
0 & \sin\alpha & \cos\alpha
\end{bmatrix}
$$

$$
R_y(\beta) = \begin{bmatrix}
\cos\beta & 0 & \sin\beta \\
0 & 1 & 0 \\
-\sin\beta & 0 & \cos\beta
\end{bmatrix}
$$

乘积：

$$
R(\alpha,\beta) = \begin{bmatrix}
\cos\beta & \sin\alpha\sin\beta & \cos\alpha\sin\beta \\
0 & \cos\alpha & -\sin\alpha \\
-\sin\beta & \sin\alpha\cos\beta & \cos\alpha\cos\beta
\end{bmatrix}
$$

### 1.2 正运动学

动平台第 $i$ 个铰链在惯性系中的位置：

$$P_i(q) = \begin{bmatrix}0 \\ 0 \\ z\end{bmatrix} + R(\alpha,\beta) \cdot p_{\text{local},i}$$

其中 $q = [z, \alpha, \beta]^T$。

---

## 2. 雅可比矩阵 $J_{P_i}$（位置对广义坐标的偏导）

### 2.1 定义

$$\dot{P}_i = J_{P_i} \dot{q}$$

其中 $J_{P_i} = \frac{\partial P_i}{\partial q}$ 是 3×3 矩阵。

### 2.2 计算

**对 $z$ 的偏导**（平移）：

$$\frac{\partial P_i}{\partial z} = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$$

**对 $\alpha$ 的偏导**（绕X轴旋转）：

$$\frac{\partial P_i}{\partial \alpha} = \frac{\partial R}{\partial \alpha} \cdot p_{\text{local},i}$$

其中：

$$\frac{\partial R}{\partial \alpha} = R_y(\beta) \cdot \frac{\partial R_x(\alpha)}{\partial \alpha}$$

$$
\frac{\partial R_x(\alpha)}{\partial \alpha} = \begin{bmatrix}
0 & 0 & 0 \\
0 & -\sin\alpha & -\cos\alpha \\
0 & \cos\alpha & -\sin\alpha
\end{bmatrix}
$$

**对 $\beta$ 的偏导**（绕Y轴旋转）：

$$\frac{\partial P_i}{\partial \beta} = \frac{\partial R}{\partial \beta} \cdot p_{\text{local},i}$$

其中：

$$\frac{\partial R}{\partial \beta} = \frac{\partial R_y(\beta)}{\partial \beta} \cdot R_x(\alpha)$$

$$
\frac{\partial R_y(\beta)}{\partial \beta} = \begin{bmatrix}
-\sin\beta & 0 & \cos\beta \\
0 & 0 & 0 \\
-\cos\beta & 0 & -\sin\beta
\end{bmatrix}
$$

### 2.3 显式表达式（代入具体坐标）

$$
J_{P_1} = \begin{bmatrix}
0 & 0 & r_p\cos\beta \\
0 & 0 & r_p\sin\alpha\sin\beta \\
1 & r_p\cos\alpha & -r_p\sin\beta
\end{bmatrix}
$$

$$
J_{P_2} = \begin{bmatrix}
0 & -\frac{\sqrt{3}}{2}r_p\cos\beta & -\frac{1}{2}r_p\cos\beta \\
0 & -\frac{\sqrt{3}}{2}r_p\sin\alpha\sin\beta & -\frac{1}{2}r_p\sin\alpha\sin\beta \\
1 & -\frac{\sqrt{3}}{2}r_p\cos\alpha & \frac{\sqrt{3}}{2}r_p\sin\beta
\end{bmatrix}
$$

$$
J_{P_3} = \begin{bmatrix}
0 & \frac{\sqrt{3}}{2}r_p\cos\beta & -\frac{1}{2}r_p\cos\beta \\
0 & \frac{\sqrt{3}}{2}r_p\sin\alpha\sin\beta & -\frac{1}{2}r_p\sin\alpha\sin\beta \\
1 & \frac{\sqrt{3}}{2}r_p\cos\alpha & -\frac{\sqrt{3}}{2}r_p\sin\beta
\end{bmatrix}
$$

---

## 3. 杆长雅可比矩阵 $J$（用于力映射）

### 3.1 定义

$$\dot{l}_i = \frac{\partial l_i}{\partial q} \dot{q} = J_{i,:} \dot{q}$$

其中 $l_i = \|P_i - B_i\|$。

### 3.2 计算公式

$$J_{i,j} = \frac{\partial \|P_i - B_i\|}{\partial q_j} = \frac{(P_i - B_i) \cdot \partial P_i / \partial q_j}{\|P_i - B_i\|}$$

### 3.3 简化计算（代码实现）

使用方向向量 $e_i = (P_i - B_i) / l_i$：

$$J_{i,0} = e_{i,z}$$

$$J_{i,1} = e_i \cdot \frac{\partial P_i}{\partial \alpha}$$

$$J_{i,2} = e_i \cdot \frac{\partial P_i}{\partial \beta}$$

---

## 4. 动能推导

### 4.1 平台动能

$$
T_{\text{plate}} = \frac12 M \dot{z}^2 + \frac12 I_{xx}\dot{\alpha}^2 + \frac12 I_{eq}\dot{\beta}^2
$$

其中 $I_{eq} = I_{yy}\cos^2\alpha + I_{zz}\sin^2\alpha$。

矩阵形式：

$$
T_{\text{plate}} = \frac12 \dot{q}^T M_{\text{plate}} \dot{q}
$$

$$
M_{\text{plate}} = \begin{bmatrix}
M & 0 & 0 \\
0 & I_{xx} & 0 \\
0 & 0 & I_{yy}\cos^2\alpha + I_{zz}\sin^2\alpha
\end{bmatrix}
$$

#### 4.2 两段式主动杆构型

**系统构型**：3-UPS/PU 并联平台
- 中间：PU 支路（被动，只起约束，无驱动力）
- 外围 3 根：UPS 主动推杆（唯一驱动力）

**每根主动杆为两段式伸缩结构**：
- 上段（靠近平台）质量：$m_{\text{up}} = 9\ \text{kg}$
- 下段（靠近基座）质量：$m_{\text{down}} = 16\ \text{kg}$
- 单杆总质量：$m_i = m_{\text{up}} + m_{\text{down}} = 25\ \text{kg}$

---

**质心位置**（不再是中点）：

$$P_{c,i} = \frac{m_{\text{down}} \cdot B_i + m_{\text{up}} \cdot P_i}{m_{\text{up}} + m_{\text{down}}} = \frac{16\,B_i + 9\,P_i}{25}$$

**质心速度**：

$$\dot P_{c,i} = \frac{9}{25}\,\dot P_i$$

---

**单根主动杆动能**：

$$
T_{\text{rod},i} = \frac12 m_i \| \dot P_{c,i} \|^2
= \frac12 \cdot 25 \cdot \left(\frac{9}{25}\right)^2 \|\dot P_i\|^2
= \frac{81}{50}\, \|\dot P_i\|^2
$$

**等效质量**：写成统一格式 $T_{\text{rod},i} = \frac12 m_{\text{eq}} \|\dot P_i\|^2$

$$m_{\text{eq}} = \frac{2 \cdot 81}{50} = \frac{81}{25} = 3.24\ \text{kg}$$

---

### 4.3 总质量矩阵

$$
M(q) = M_{\text{plate}} + \frac{81}{25}\,\sum_{i=1}^3 J_{P_i}^T J_{P_i}
$$

**与均匀杆对比**：
- 均匀杆：$\frac14 m_i = \frac{25}{4} = 6.25$
- 两段杆：$\frac{81}{25} = 3.24$

**注意**：当前代码实现中 $M(q)$ 只包含平台质量，未包含杆质量的贡献！
---

## 5. 势能推导

### 5.1 平台势能

$$V_{\text{plate}} = M g z$$

（$z$ 向下为正，重力势能降低）

### 5.2 杆势能

第 $i$ 根杆的质心高度：

$$z_{c,i} = \frac{16\, z_{B_i} + 9\, z_{P_i}}{25}$$

杆势能：

$$V_{\text{rod},i} = m_i g \cdot z_{c,i} = 25g \cdot \frac{16\, z_{B_i} + 9\, z_{P_i}}{25} = g\left(16\,z_{B_i} + 9\,z_{P_i}\right)$$

### 5.3 总势能

$$V = M g z + \sum_{i=1}^3 g\left(16\,z_{B_i} + 9\,z_{P_i}\right)$$

由于 $z_{B_i} = 0$（基座在 z=0 平面），简化为：

$$V = M g z + 9g \sum_{i=1}^3 z_{P_i}$$

其中 $z_{P_i}$ 是 $P_i$ 的 z 坐标。
---

## 6. 拉格朗日方程

### 6.1 拉格朗日函数

$$L = T - V$$

### 6.2 标准方程

$$\frac{d}{dt}\left(\frac{\partial L}{\partial \dot{q}}\right) - \frac{\partial L}{\partial q} = \tau$$

### 6.3 展开

$$\frac{d}{dt}(M\dot{q}) - \frac12 \dot{q}^T \frac{\partial M}{\partial q} \dot{q} - \frac{\partial V}{\partial q} = \tau$$

整理为标准形式：

$$M\ddot{q} + C\dot{q} + G = \tau$$

其中：

- $M$：质量矩阵
- $C$：科里奥利/离心力矩阵（由 $M$ 求导得到）
- $G = \partial V / \partial q$：重力向量
- $\tau$：广义力

---

## 7. 科里奥利矩阵 $C$ 的计算

### 7.1 公式

$$C_{ij} = \sum_k \frac12 \left( \frac{\partial M_{ij}}{\partial q_k} + \frac{\partial M_{ik}}{\partial q_j} - \frac{\partial M_{kj}}{\partial q_i} \right) \dot{q}_k$$

```python
def compute_coriolis(M_func, q, qd):
    """计算科里奥利矩阵（Christoffel符号）"""
    n = len(q)
    C = np.zeros((n, n))
    
    eps = 1e-6
    
    for j in range(n):
        # 数值求导：∂M/∂q_j
        q_plus = q.copy()
        q_minus = q.copy()
        q_plus[j] += eps
        q_minus[j] -= eps
        
        M_plus = M_func(q_plus)
        M_minus = M_func(q_minus)
        dM_dqj = (M_plus - M_minus) / (2 * eps)
        
        # C_ij = 0.5 * (∂M_ij/∂q_k + ∂M_kj/∂q_i - ∂M_ik/∂q_j) * qd_k
        for i in range(n):
            for k in range(n):
                C[i, k] += 0.5 * (
                    dM_dqj[i, j] +      # ∂M_ij/∂q_k
                    dM_dqj[k, i] -      # ∂M_kj/∂q_i  
                    dM_dqj[i, k]        # ∂M_ik/∂q_j
                ) * qd[j]
    
    return C
```

---

## 8. 重力向量 $G$ 的计算

### 8.1 公式

$$G_i = \frac{\partial V}{\partial q_i}$$

### 8.2 平台重力项

$$G_z = \frac{\partial (Mgz)}{\partial z} = Mg$$

$$G_\alpha = \frac{\partial (Mgz)}{\partial \alpha} = 0$$

$$G_\beta = \frac{\partial (Mgz)}{\partial \beta} = 0$$

### 8.3 杆重力项（简化）

如果忽略杆质量对重力项的贡献（或将杆重力合并到等效平台质量中）：

$$G = \begin{bmatrix} Mg \\ 0 \\ 0 \end{bmatrix}$$

### 8.4 完整重力项（含两段杆质量）

根据新的势能公式 $V = Mgz + 9g \sum z_{P_i}$：

$$G_z = \frac{\partial (Mgz)}{\partial z} + 9g \sum_{i=1}^3 \frac{\partial z_{P_i}}{\partial z} = Mg + 9g \cdot 3 = (M + 27)g$$

$$G_\alpha = 9g \sum_{i=1}^3 \frac{\partial z_{P_i}}{\partial \alpha}$$

$$G_\beta = 9g \sum_{i=1}^3 \frac{\partial z_{P_i}}{\partial \beta}$$

> **注意**：系数从均匀杆的 $1/2$ 变成两段杆的 $9/25$。

## 9. 完整动力学方程

### 9.1 标准形式

$$\boxed{M(q)\ddot{q} + C(q,\dot{q})\dot{q} + G(q) = J^T f}$$

其中：
- $q = [z, \alpha, \beta]^T$：广义坐标
- $M(q)$：质量矩阵（含平台 + 杆）
- $C(q,\dot{q})$：科里奥利矩阵
- $G(q)$：重力向量
- $J$：杆长雅可比矩阵
- $f = [f_1, f_2, f_3]^T$：杆轴向推力

### 9.2 展开形式

$$M_{ij}\ddot{q}_j + C_{ij}\dot{q}_j + G_i = J^T f$$

$$

$$
\begin{bmatrix} M_{11} & M_{12} & M_{13} \\ M_{21} & M_{22} & M_{23} \\ M_{31} & M_{32} & M_{33} \end{bmatrix}\begin{bmatrix} \ddot{z} \\ \ddot{\alpha} \\ \ddot{\beta} \end{bmatrix} + \begin{bmatrix} C_{11} & C_{12} & C_{13} \\ C_{21} & C_{22} & C_{23} \\ C_{31} & C_{32} & C_{33} \end{bmatrix}\begin{bmatrix} \dot{z} \\ \dot{\alpha} \\ \dot{\beta} \end{bmatrix} + \begin{bmatrix} G_1 \\ G_2 \\ G_3 \end{bmatrix} = J^T\begin{bmatrix} f_1 \\ f_2 \\ f_3 \end{bmatrix}
$$

---

## 10. 与现有代码对比

| 项 | 现有代码 | 完整拉格朗日 |
|----|----------|--------------|
| $M$ 矩阵 | 只含平台质量 | 平台 + 杆质量贡献 |
| $C$ 矩阵 | 解析公式 | 与左一致 |
| $G$ 向量 | $[Mg, 0, 0]^T$ | 含杆质量贡献 |

---

## 11. 参考文献

1. Spong, M. W., & Vidyasagar, M. (2008). Robot Dynamics and Control
2. Siciliano, B., et al. (2010). Robotics: Modelling, Planning and Control
3. 本项目：`simulation/common/platform_dynamics.py`
