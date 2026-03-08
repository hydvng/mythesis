# 3-UPS/PU 并联平台动力学方程（均匀杆件）

---

## 1. 广义坐标

| 符号 | 含义 |
|------|------|
| $q_s = [z_s, \alpha_s, \beta_s]^T$ | 船体运动（下平台相对于惯性系） |
| $q_c = [z_c, \alpha_c, \beta_c]^T$ | 平台控制量（上平台相对于下平台） |
| $q = q_s + q_c$ | 上平台相对于惯性系的总位姿（时间域叠加关系） |

> 约定：本文将 $q_s(t),q_c(t)$ 视为外部给定信号；构造 $\partial(\cdot)/\partial q$（对 $q=[z,\alpha,\beta]^T$ 的偏导）时保持 $q_s$ 不变，仅在时间求导时使用 $q=q_s+q_c$（即 $\dot q=\dot q_s+\dot q_c$，$\ddot q=\ddot q_s+\ddot q_c$）。

---

## 2. 运动学

### 2.1 船体扰动（下平台）

$$q_s = [z_s, \alpha_s, \beta_s]^T$$

基座中心位置：
$$B_0(q_s) = B_{0,0} + \begin{bmatrix}0 \\ 0 \\ z_s\end{bmatrix} + R_s(\alpha_s, \beta_s) \cdot B_{0,\text{offset}}$$

基座第 $i$ 个铰点位置：令 $b_{\text{local},i}\in\mathbb{R}^3$ 为第 $i$ 个基座铰点在基座系 $\{B\}$ 下的固定局部坐标，则
$$\boxed{\ B_i(q_s)=B_0(q_s)+R_s(\alpha_s,\beta_s)\,b_{\text{local},i}\ }$$

并可用几何参数 $r_b,\theta_i$ 给出一个常用取法（等边三点分布，且局部坐标系 $\{B\}$ 的 $z$ 轴与惯性系一致）：
$$b_{\text{local},i}=\begin{bmatrix} r_b\cos\theta_i\\ r_b\sin\theta_i\\ 0\end{bmatrix},\qquad \theta_i\in\Big\{0,\tfrac{2\pi}{3},\tfrac{4\pi}{3}\Big\}$$

### 2.2 上平台控制量

$$q_c = [z_c, \alpha_c, \beta_c]^T$$

### 2.3 动平台中心位置

$$P_0(q, q_s) = B_0(q_s) + R_s(\alpha_s, \beta_s) \cdot \begin{bmatrix}0 \\ 0 \\ z_c\end{bmatrix}$$

### 2.4 动平台姿态

$$\alpha = \alpha_s + \alpha_c$$
$$\beta = \beta_s + \beta_c$$

### 2.5 动平台铰链位置

$$P_i(q, q_s) = P_0(q, q_s) + R(\alpha, \beta) \cdot p_{\text{local},i}$$

其中动平台局部铰链坐标也可用几何参数 $r_p,\theta_i$ 给出一个常用取法：
$$p_{\text{local},i}=\begin{bmatrix} r_p\cos\theta_i\\ r_p\sin\theta_i\\ 0\end{bmatrix},\qquad \theta_i\in\Big\{0,\tfrac{2\pi}{3},\tfrac{4\pi}{3}\Big\}$$

### 2.6 旋转矩阵（明确欧拉角序）

本文采用
$$R(\alpha,\beta)=R_y(\beta)R_x(\alpha)$$

并记其偏导为
$$\frac{\partial R}{\partial \alpha}=R_y(\beta)\frac{\partial R_x(\alpha)}{\partial \alpha},\qquad
\frac{\partial R}{\partial \beta}=\frac{\partial R_y(\beta)}{\partial \beta}R_x(\alpha)$$

---

## 3. 符号定义（补充维度）

| 符号 | 含义 |
|------|------|
| $B_i\in\mathbb{R}^3$ | 基座第 $i$ 个铰点在惯性系的位置（依赖 $q_s$） |
| $P_i\in\mathbb{R}^3$ | 动平台第 $i$ 个铰点在惯性系的位置 |
| $b_{\text{local},i}\in\mathbb{R}^3$ | 基座第 $i$ 个铰点在基座系 $\{B\}$ 下的局部坐标（常量） |
| $p_{\text{local},i}\in\mathbb{R}^3$ | 动平台第 $i$ 个铰点在平台系 $\{P\}$ 下的局部坐标（常量） |
| $R_s(\alpha_s,\beta_s)\in\mathrm{SO}(3)$ | 船体/基座姿态旋转矩阵（将 $\{B\}$ 向量旋到惯性系 $\{I\}$） |
| $R(\alpha,\beta)\in\mathrm{SO}(3)$ | 动平台姿态旋转矩阵（本文取 $R=R_y(\beta)R_x(\alpha)$） |
| $L_i=\|P_i-B_i\|$ | 第 $i$ 根杆长度（驱动量） |
| $\mathbf{e}_i=(P_i-B_i)/L_i\in\mathbb{R}^3$ | 第 $i$ 根杆方向单位向量 |
| $\mathbf I_3\in\mathbb{R}^{3\times 3}$ | $3\times 3$ 单位阵（为避免与转动惯量记号混淆，统一写作粗体 $\mathbf I_3$） |
| $J(q)=\partial L/\partial q\in\mathbb{R}^{3\times 3}$ | 杆长对广义坐标的雅可比 |
| $f=[f_1,f_2,f_3]^T\in\mathbb{R}^3$ | 三根杆的轴向推力 |

---

## 4. 质量参数

| 参数 | 符号 | 值 |
|------|------|-----|
| 平台质量 | $M$ | 347.54 kg |
| 单杆质量（均匀） | $m_{rod}$ | 25 kg |
| 杆长 | $l$ | - |
| 绕X轴惯量 | $I_{xx}$ | 60.64 kg·m² |
| 绕Y轴惯量 | $I_{yy}$ | 115.4 kg·m² |
| 绕Z轴惯量 | $I_{zz}$ | 80.0 kg·m² |

---

## 4'. 几何参数（补充）

| 参数 | 符号 | 值 |
|------|------|-----|
| 平台铰链圆半径 | $r_p$ | 0.58 m |
| 基座铰链圆半径 | $r_b$ | 0.65 m |
| 铰链周向角 | $\theta_i$ | $0,\ 2\pi/3,\ 4\pi/3$ |
| 名义杆长（零位参考/初始化） | $l_{nom}$ | 1.058 m |

> 注：本文件的均匀杆长度记为 $l$，若采用名义长度建模可取 $l=l_{nom}$；若后续引入杆长驱动量 $L_i$，则 $l$ 仅用于惯量项 $\tfrac{1}{12}m_{rod}l^2$ 的标称长度。

---

## 5. 动能推导

### 5.1 平台动能

$$T_{\text{plate}} = \frac12 M \dot{z}^2 + \frac12 I_{xx}\dot{\alpha}^2 + \frac12 I_{eq}\dot{\beta}^2$$

其中 $I_{eq} = I_{yy}\cos^2\alpha + I_{zz}\sin^2\alpha$。

$$M_{\text{plate}} = \begin{bmatrix} M & 0 & 0 \\ 0 & I_{xx} & 0 \\ 0 & 0 & I_{eq} \end{bmatrix}$$

### 5.2 杆件平动动能（均匀杆）

质心在杆件中点（对于第 $i$ 根杆）：
$$C_i = \frac{P_i + B_i}{2}$$

质心速度：
$$\dot{C}_i = \frac{\dot{P}_i + \dot{B}_i}{2}$$

平动动能（对 $i=1,2,3$ 求和）：
$$T_{\text{trans}} = \sum_{i=1}^3 \frac12 m_{rod} \|\dot{C}_i\|^2
= \sum_{i=1}^3 \frac{m_{rod}}{8} \left(\|\dot{P}_i\|^2 + 2\dot{P}_i^T\dot{B}_i + \|\dot{B}_i\|^2\right)$$

整理后系数：
- $\dot{P}_i^T\dot{P}_i$ 项：$\frac{m_{rod}}{8}$
- $\dot{B}_i^T\dot{B}_i$ 项：$\frac{m_{rod}}{8}$
- $\dot{P}_i^T\dot{B}_i$ 项：$\frac{m_{rod}}{4}$

### 5.3 杆件转动动能

方向向量（对于第 $i$ 根杆）：
$$\mathbf{e}_i = \frac{P_i - B_i}{\|P_i - B_i\|}$$

角速度：$\omega_i = \|\dot{\mathbf{e}}_i\|$

转动惯量（细杆绕质心）：$I_i = \frac{1}{12}m_{rod} l^2$

转动动能：
$$T_{\text{rot}} = \sum_{i=1}^3 \frac12 I_i \omega_i^2 = \sum_{i=1}^3 \frac{m_{rod} l^2}{24} \|\dot{\mathbf{e}}_i\|^2$$

### 5.4 $\|\dot{\mathbf{e}}_i\|^2$ 计算

$$\dot{\mathbf{e}}_i = \frac{\dot{P}_i - \dot{B}_i}{L_i} - \frac{(P_i - B_i)(\dot{P}_i - \dot{B}_i)^T (P_i - B_i)}{L_i^3}$$

$$\|\dot{\mathbf{e}}_i\|^2 = \frac{\|\dot{P}_i - \dot{B}_i\|^2 - \left(\mathbf{e}_i^T (\dot{P}_i - \dot{B}_i)\right)^2}{L_i^2}$$

其中 $L_i = \|P_i - B_i\|$。

---

## 6. 势能

### 6.1 质心位置

均匀杆质心在杆件中点（对于第 $i$ 根杆）：
$$C_i = \frac{P_i + B_i}{2}$$

### 6.2 势能公式

$$V = M g z + \sum_{i=1}^3 \frac{m_{rod} g}{2} (z_{P_i} + z_{B_i})$$

> 注：$z_{B_i}$ 随船体运动变化（依赖 $q_s$）。在本文偏导约定（构造 $\partial(\cdot)/\partial q$ 时保持 $q_s$ 不变）下，有 $\partial z_{B_i}/\partial q = 0$；但其时间变化（$\dot z_{B_i},\ddot z_{B_i}$）将对动能/广义力产生影响，本文将其通过等效扰动项 $\tau_s$ 体现。

---

## 7. 拉格朗日方程

名义动力学写作
$$\boxed{\ M(q)\ddot{q} + C(q,\dot{q})\dot{q} + G(q) = \tau\ }$$

其中执行器广义力
$$\tau = J(q)^T f$$

若采用分解 $q=q_s+q_c$，则可写成控制形式
$$\boxed{\ M(q)\ddot{q}_c + C(q,\dot{q})\dot{q}_c + G(q) = \tau + \tau_s\ }$$

并取等效扰动项的紧凑定义
$$\boxed{\ \tau_s \triangleq -M(q)\ddot q_s - C(q,\dot q)\dot q_s\ }$$

### 7.1 质量矩阵

$$M(q) = M_{\text{plate}}(q) + M_{\text{rod,trans}}(q) + M_{\text{rod,rot}}(q)$$

- 平台部分：$M_{\text{plate}}(q)=M_{\text{plate}}$。

- 杆件平动部分（仅保留 $\dot P_i$ 的纯二次项）：
$$\boxed{\ M_{\text{rod,trans}}(q)=\frac{m_{rod}}{8}\sum_{i=1}^3 J_{P_i}(q)^T J_{P_i}(q)\ }$$

- 杆件转动部分：令 $E_i(q)=\partial \mathbf{e}_i/\partial q\in\mathbb{R}^{3\times 3}$，则
$$\boxed{\ M_{\text{rod,rot}}(q)=\frac{m_{rod} l^2}{24}\sum_{i=1}^3 E_i(q)^T E_i(q)\ }$$

并且在本文偏导约定（保持 $q_s$ 不变，故 $\partial B_i/\partial q=0$）下，对任意 $q_k\in\{z,\alpha,\beta\}$
$$\boxed{\ \frac{\partial \mathbf{e}_i}{\partial q_k}=\frac{1}{L_i}(\mathbf I_3-\mathbf{e}_i\mathbf{e}_i^T)\frac{\partial P_i}{\partial q_k}\ }$$

### 7.2 重力向量

由 $G(q)=\partial V/\partial q$，在保持 $q_s$ 不变的偏导约定下（因此 $\partial z_{B_i}/\partial q=0$），可取

$$G_z = \frac{\partial V}{\partial z} = Mg + \frac{m_{rod}g}{2} \sum_{i=1}^3 \frac{\partial z_{P_i}}{
\partial z}$$

$$G_\alpha = \frac{m_{rod}g}{2} \sum_{i=1}^3 \frac{\partial z_{P_i}}{\partial \alpha}$$

$$G_\beta = \frac{m_{rod}g}{2} \sum_{i=1}^3 \frac{\partial z_{P_i}}{\partial \beta}$$

---

## 8. 动力学方程

$$\boxed{M(q)\ddot{q} + C(q,\dot{q})\dot{q} + G(q) = J(q)^T f + \tau_s}$$

其中 $f = [f_1, f_2, f_3]^T$ 是杆轴向推力。
