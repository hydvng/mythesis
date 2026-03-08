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

### 1.2 基座运动学（船体运动）

基座中心 $B_0$ 固定在船上，随船体运动：

**船体运动**（6自由度）：
- $q_s = [x_s, y_s, z_s, \phi_s, \theta_s, \psi_s]^T$
  - $[x_s, y_s, z_s]$：船体位置（平移）
  - $[\phi_s, \theta_s, \psi_s]$：船体姿态（横滚、俯仰、偏航）

**基座中心位置**：
$$B_0(q_s) = \underbrace{B_{0,0}}_{\text{固定参考点}} + \underbrace{R_s(\phi_s, \theta_s, \psi_s) \cdot B_{0,\text{offset}}}_{\text{安装偏移}} + \underbrace{\begin{bmatrix}x_s \\ y_s \\ z_s\end{bmatrix}}_{\text{船体平移}}$$

其中：
- $B_{0,0}$：固定在码头/惯性系的参考点
- $R_s(\phi_s, \theta_s, \psi_s)$：船体旋转矩阵（3个欧拉角）
- $B_{0,\text{offset}}$：基座相对于船体重心的安装偏移向量
- $[x_s, y_s, z_s]^T$：船体位置

**简化为3自由度（只考虑横滚、俯仰、升沉）**：
$$q_s = [z_s, \alpha_s, \beta_s]^T$$

$$B_0(q_s) = B_{0,0} + \begin{bmatrix}0 \\ 0 \\ z_s\end{bmatrix} + R_s(\alpha_s, \beta_s) \cdot B_{0,\text{offset}}$$

---

## 1. 坐标定义

### 1.1 广义坐标

- **$q_s = [z_s, \alpha_s, \beta_s]^T$**：船体运动（下平台/基座相对于惯性系）
- **$q_c = [z_c, \alpha_c, \beta_c]^T$**：平台控制量（上平台相对于下平台）
- **$q = q_s + q_c$**：上平台相对于惯性系的总位移

### 1.2 基座运动学（船体运动）

**船体运动**（3自由度）：
$$q_s = [z_s, \alpha_s, \beta_s]^T$$

**基座中心位置**：
$$B_0(q_s) = B_{0,0} + \begin{bmatrix}0 \\ 0 \\ z_s\end{bmatrix} + R_s(\alpha_s, \beta_s) \cdot B_{0,\text{offset}}$$

其中 $R_s$ 是船体姿态旋转矩阵。

---

### 1.3 动平台运动学

**控制量 $q_c$**：
$$q_c = [z_c, \alpha_c, \beta_c]^T$$

- $z_c$：上平台相对于下平台的升沉位移
- $\alpha_c$：上平台相对于下平台的横滚角
- $\beta_c$：上平台相对于下平台的俯仰角

**关键**：$q_c$ 定义在**船体坐标系**下，其方向受船体扰动影响。

**动平台中心位置**：

$$P_0(q, q_s) = B_0(q_s) + R_s(\alpha_s, \beta_s) \cdot \begin{bmatrix}0 \\ 0 \\ z_c\end{bmatrix}$$

解释：
- $B_0(q_s)$：基座中心（受扰动）
- $R_s(\alpha_s, \beta_s)$：将 $z_c$ 从船体坐标系转换到惯性系
- $z_c$：相对位移大小

**动平台姿态**：

$$\alpha = \alpha_s + \alpha_c$$
$$\beta = \beta_s + \beta_c$$

即：
$$R(\alpha, \beta) = R_s(\alpha_s, \beta_s) \cdot R_c(\alpha_c, \beta_c)$$

**动平台铰链 $P_i$**：

$$P_i(q, q_s) = P_0(q, q_s) + R(\alpha, \beta) \cdot p_{\text{local},i}$$

其中 $p_{\text{local},i}$ 是铰链在平台坐标系中的位置。

---

### 1.4 总结

**位置关系**：
$$q = q_s + q_c$$

**姿态关系**：
$$\alpha = \alpha_s + \alpha_c$$
$$\beta = \beta_s + \beta_c$$

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

### 3.4 基座雅可比矩阵 $J_B$（扰动贡献）

基座位置 $B_i$ 随船体运动 $q_s = [z_s, \alpha_s, \beta_s]$ 变化：

$$B_i(q_s) = B_{i,0} + R_s(\alpha_s, \beta_s) \cdot B_{i,\text{offset}}$$

**基座雅可比**：
$$\frac{\partial B_i}{\partial q_s}$$

**链式法则**：杆长变化由动平台和基座共同决定：
$$\frac{\partial l_i}{\partial q} = \frac{\partial \|P_i - B_i\|}{\partial q} = \frac{(P_i - B_i) \cdot (\partial P_i / \partial q - \partial B_i / \partial q)}{l_i}$$

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

### 4.2 两段式主动杆构型

**系统构型**：3-UPS/PU 并联平台
- 中间：PU 支路（被动，只起约束，无驱动力）
- 外围 3 根：UPS 主动推杆（唯一驱动力）

**每根主动杆为两段式伸缩结构**：
- 上段（靠近平台）质量：$m_u = 9\ \text{kg}$
- 下段（靠近基座）质量：$m_d = 16\ \text{kg}$
- 单杆总质量：$m_i = m_u + m_d = 25\ \text{kg}$

---

**铰链位置**（在杆件中间）：

$$J = \frac{B + P}{2}$$

---

**下杆质心速度**（$m_d = 16$ kg）：

$$\dot{B}_d = \frac{3\dot{B} + \dot{P}}{4}$$

---

**上杆质心速度**（$m_u = 9$ kg）：

$$\dot{P}_u = \frac{\dot{B} + 3\dot{P}}{4}$$

---

**总动能** = 平动动能 + 转动动能

### 4.2.1 平动动能

$$
T_{\text{平动}} = \frac{m_d}{32}(9\dot{B}^T\dot{B} + 6\dot{B}^T\dot{P} + \dot{P}^T\dot{P}) + \frac{m_u}{32}(\dot{B}^T\dot{B} + 6\dot{B}^T\dot{P} + 9\dot{P}^T\dot{P})
$$

整理：

$$
T_{\text{平动}} = \frac{9m_d+m_u}{32}\dot{B}^T\dot{B} + \frac{6(m_d+m_u)}{32}\dot{B}^T\dot{P} + \frac{m_d+9m_u}{32}\dot{P}^T\dot{P}
$$

其中 $m_d=16, m_u=9$：
- $A = \frac{9\times16+9}{32} = \frac{153}{32} = 4.78125$
- $B = \frac{6\times25}{32} = \frac{150}{32} = 4.6875$
- $C = \frac{16+9\times9}{32} = \frac{97}{32} = 3.03125$

### 4.2.2 转动动能

**几何关系**：
- 上杆：长度 $l_u$，上铰点 $P$（动平台），下铰点 $J$（中间）
- 下杆：长度 $l_d$，上铰点 $J$（中间），下铰点 $B$（基座，受船体扰动影响）

**受扰动的基座运动**：
基座 $B$ 随船体运动，引入扰动角度 $\alpha_s$（船体横滚）、$\beta_s$（船体俯仰）：
$$B = B_0 + \text{扰动}(\alpha_s, \beta_s)$$

---

**角速度定义**：
杆的方向向量：
$$\mathbf{e} = \frac{P - B}{\|P - B\|} = \frac{P - B}{L}$$

其中 $L = l_u + l_d$ 是总杆长。

**角速度大小**：
$$\omega = \|\dot{\mathbf{e}}\|$$

**转动惯量**（细杆绕质心横向转动）：
- 上杆长度 $l_u$：
$$I_u = \frac{1}{12}m_u l_u^2$$
- 下杆长度 $l_d$：
$$I_d = \frac{1}{12}m_d l_d^2$$

**转动动能**：
$$T_{\text{rot}} = \frac12 I_u \omega_u^2 + \frac12 I_d \omega_d^2$$

由于两段杆共线，角速度相同 $\omega_u = \omega_d = \|\dot{\mathbf{e}}\|$：
$$T_{\text{rot}} = \frac12 \left(\frac{m_u l_u^2 + m_d l_d^2}{12}\right) \omega^2 = \frac{m_u l_u^2 + m_d l_d^2}{24} \|\dot{\mathbf{e}}\|^2$$

---

### 4.2.3 $||\dot{\mathbf{e}}||^2$ 的计算

对 $\mathbf{e} = (P - B) / L$ 求导：
$$\dot{\mathbf{e}} = \frac{\dot{P} - \dot{B}}{L} - \frac{(P - B)(\dot{P} - \dot{B})^\top (P - B)}{L^3}$$

模长平方：
$$\|\dot{\mathbf{e}}\|^2 = \frac{\|\dot{P} - \dot{B}\|^2 - \left(\mathbf{e}^\top (\dot{P} - \dot{B})\right)^2}{L^2}$$

---

### 4.2.4 完整动能（平动 + 转动）

$$
\boxed{
\begin{aligned}
T &= \frac{m_d}{32}\left\|3\dot{B}+\dot{P}\right\|^2
   + \frac{m_u}{32}\left\|\dot{B}+3\dot{P}\right\|^2 \\[4pt]
  &+ \frac{m_u l_u^2 + m_d l_d^2}{24} \cdot
\frac{\|\dot{P}-\dot{B}\|^2 - \big[\mathbf{e}^\top(\dot{P}-\dot{B})\big]^2}{\|P-B\|^2}
\end{aligned}
}
$$

其中 $l_u$ 是上杆长度，$l_d$ 是下杆长度，$\dot{B}$ 包含船体扰动 $\alpha_s, \beta_s$ 的影响。

---

**重要**：基座 B 随船体运动，$\dot{B} \neq 0$，不能忽略！


---

## 5. 势能推导

### 5.1 方案一：两段式杆件（上下质量不同）

**杆件构型**：
- 下杆（靠近基座）：$m_d = 16$ kg, 长度 $l_d$
- 上杆（靠近平台）：$m_u = 9$ kg，长度 $l_u$
- 杆长范围：完全收缩 $l_{min}$，完全伸长 $l_{max} $

**定义**：
- $e_i = \frac{P_i - B_i}{\|P_i - B_i\|}$：杆的方向向量（单位向量）
- $l_i = \|P_i - B_i\|$：杆长

**质心位置**（两段杆的质心都在各自杆的中点）：
- 下杆质心：$C_{d,i} = B_i + e_i \cdot \frac{l_d}{2}$
- 上杆质心：$C_{u,i} = P_i - e_i \cdot \frac{l_u}{2}$（从平台往下算）

**质心高度**：
- $z_{c,d,i} = z_{B_i} + e_{i,z} \cdot \frac{l_d}{2}$
- $z_{c,u,i} = z_{P_i} - e_{i,z} \cdot \frac{l_u}{2}$

**总势能**：

$$V = M g z + \sum_{i=1}^3 \left( m_d g \cdot z_{c,d,i} + m_u g \cdot z_{c,u,i} \right)$$

展开：

$$V = M g z + \sum_{i=1}^3 \left[ m_d g \cdot (z_{B_i} + e_{i,z} \frac{l_d}{2}) + m_u g \cdot (z_{P_i} - e_{i,z} \frac{l_u}{2}) \right]$$

由于 $l_d = l_i - l_u$：

$$V = M g z + \sum_{i=1}^3 \left[ (m_d + m_u) g \cdot z_{B_i} + m_u g \cdot z_{P_i} + g \cdot e_{i,z} \cdot \left( m_d \frac{l_i - l_u}{2} - m_u \frac{l_u}{2} \right) \right]$$

整理：

$$V = M g z + \sum_{i=1}^3 \left[ 25 g \cdot z_{B_i} + 9 g \cdot z_{P_i} + g \cdot e_{i,z} \cdot \left( 16 \cdot \frac{l_i - l_u}{2} - 9 \cdot \frac{l_u}{2} \right) \right]$$

最终：

$$V = M g z + 9g \sum_{i=1}^3 z_{P_i} + 25g \sum_{i=1}^3 z_{B_i} + \frac{g}{2} \sum_{i=1}^3 e_{i,z} \cdot (16l_i - 25l_u)$$

其中 $l_i = \|P_i - B_i\|$，$l_u$ 是上杆长度（常数）。

### 5.2 方案二：均匀杆件（简化）

**假设**：单杆质量均匀分布，$m_{rod} = 25$ kg，质心在杆件中点

**约束关系**：杆长由运动学决定

$$l_i = \|P_i - B_i\|$$

**质心位置**（在杆件中点）：

$$C_i = \frac{P_i + B_i}{2}$$

**质心高度**：

$$z_{c,i} = \frac{z_{P_i} + z_{B_i}}{2}$$

注意：$z_{P_i}$ 和 $z_{B_i}$ 都是由运动学决定的函数。

**总势能**：

$$V = M g z + \sum_{i=1}^3 m_{rod} g \cdot \frac{z_{P_i} + z_{B_i}}{2}$$

其中 $z_{P_i}$ 是动平台铰链的 z 坐标（由 q = [z, α, β] 决定），$z_{B_i}$ 是基座铰链的 z 坐标。

**重要**：引入扰动以后 $z_{B_i}$ 是随船体运动变化的（不再是0）！

---

## 6. 拉格朗日方程

### 6.1 拉格朗日函数

$$L = T - V$$

### 6.2 标准方程

$$\frac{d}{dt}\left(\frac{\partial L}{\partial \dot{q}}\right) - \frac{\partial L}{\partial q} = \tau$$

### 6.3 展开

质量矩阵 M 的完整形式（平台 + 杆件平动 + 杆件转动）：

$$M = M_{\text{plate}} + M_{\text{rod,平动}} + M_{\text{rod,转动}}$$

其中：

**平台部分**：
$$M_{\text{plate}} = \begin{bmatrix} M & 0 & 0 \\ 0 & I_{xx} & 0 \\ 0 & 0 & I_{yy}\cos^2\alpha + I_{zz}\sin^2\alpha \end{bmatrix}$$

**杆件平动部分**（见4.2.1节）：
$$M_{\text{rod,平动}} = \frac{97}{32} \sum_{i=1}^3 J_{P_i}^T J_{P_i}$$

**杆件转动部分**：
$$M_{\text{rod,转动}} = \sum_{i=1}^3 \frac{\partial e_i}{\partial q}^T I_{\text{杆},i} \frac{\partial e_i}{\partial q}$$

其中 $e_i = (P_i - B_i)/l_i$ 是杆的方向向量。

$$\frac{d}{dt}(M\dot{q}) - \frac12 \dot{q}^T \frac{\partial M}{\partial q} \dot{q} - \frac{\partial V}{\partial q} = \tau$$

整理为标准形式：

$$M\ddot{q} + C\dot{q} + G = \tau$$

其中：

- $M$：完整质量矩阵（含平台 + 杆件平动 + 杆件转动）
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

### 8.2 方案一：两段式杆件势能

根据第5.1节的势能公式：

$$V = M g z + 9g \sum_{i=1}^3 z_{P_i} + 25g \sum_{i=1}^3 z_{B_i} + \frac{g}{2} \sum_{i=1}^3 e_{i,z} \cdot (16l_i - 25l_u)$$

其中 $l_i = \|P_i - B_i\|$，$l_u$ 是上杆长度（常数）。

**重力向量**：
- $G_z = \frac{\partial V}{\partial z} = Mg + 9g \sum \frac{\partial z_{P_i}}{\partial z}$
- $G_\alpha = \frac{\partial V}{\partial \alpha} = 9g \sum \frac{\partial z_{P_i}}{\partial \alpha} + \frac{g}{2} \sum e_{i,z} \cdot 16 \frac{\partial l_i}{\partial \alpha}$
- $G_\beta = \frac{\partial V}{\partial \beta} = 9g \sum \frac{\partial z_{P_i}}{\partial \beta} + \frac{g}{2} \sum e_{i,z} \cdot 16 \frac{\partial l_i}{\partial \beta}$

### 8.3 方案二：均匀杆件势能

根据第5.2节的势能公式：

$$V = M g z + \sum_{i=1}^3 \frac{m_{rod} g}{2} (z_{P_i} + z_{B_i})$$

**重力向量**：
- $G_z = Mg + \frac{m_{rod}g}{2} \sum_{i=1}^3 \frac{\partial z_{P_i}}{\partial z}$
- $G_\alpha = \frac{m_{rod}g}{2} \sum_{i=1}^3 \frac{\partial z_{P_i}}{\partial \alpha}$
- $G_\beta = \frac{m_{rod}g}{2} \sum_{i=1}^3 \frac{\partial z_{P_i}}{\partial \beta}$

> **注意**：引入扰动后 $z_{B_i}$ 也随时间变化，但作为已知输入处理。

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
