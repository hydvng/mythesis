# 3-UPS/PU 并联平台动力学方程（两段式杆件）

---

## 0. 建模约定与参数

### 0.1 坐标系与正方向

- 惯性系 $\{I\}$：$z$ 轴竖直向上。
- 基座系 $\{B\}$：固定在下平台（随船体运动）。
- 动平台系 $\{P\}$：固定在上平台。

角度约定：本文仅考虑横滚/俯仰两自由度，记
- $\alpha$：绕 $x$ 轴的滚转（roll）
- $\beta$：绕 $y$ 轴的俯仰（pitch）

> 若后续需要写出 $R(\alpha,\beta)$ 的闭式，请在全文统一采用同一欧拉角序（例如 $R=R_y(\beta)R_x(\alpha)$）。本文符号推导对任意光滑 $R$ 成立。

### 0.2 几何与质量参数（给定数值）

**几何参数（与构型相关）**：

- 动平台铰链圆半径：$r_p=0.58\,\mathrm{m}$
- 基座铰链圆半径：$r_b=0.65\,\mathrm{m}$
- 三个支链的周向角：$\theta_i\in\{0,\tfrac{2\pi}{3},\tfrac{4\pi}{3}\}$
- 名义杆长（用于初始化/零位参考）：$l_{nom}=1.058\,\mathrm{m}$

**质量参数（本章使用）**：

- 动平台质量：$M=347.54\,\mathrm{kg}$
- 下杆质量：$m_d=16\,\mathrm{kg}$（每根）
- 上杆质量：$m_u=9\,\mathrm{kg}$（每根）
- 下杆长度：$l_d=0.90\,\mathrm{m}$
- 上杆长度：$l_u=0.85\,\mathrm{m}$
- 重力加速度：$g=9.81\,\mathrm{m/s^2}$

动平台转动惯量（需由结构给定/测量）：$I_{xx},I_{yy},I_{zz}$。

两段杆关于自身质心、垂直杆轴的转动惯量（细长杆假设）：
$$I_{d}=\frac{1}{12}m_d l_d^2,\qquad I_{u}=\frac{1}{12}m_u l_u^2$$

---

## 1. 广义坐标与状态变量

系统仅考虑 3 个自由度（升沉 + 两转角）：
$$q=\begin{bmatrix}z\\\alpha\\\beta\end{bmatrix}\in\mathbb{R}^3,\qquad \dot{q}=\begin{bmatrix}\dot z\\\dot\alpha\\\dot\beta\end{bmatrix},\qquad \ddot{q}=\begin{bmatrix}\ddot z\\\ddot\alpha\\\ddot\beta\end{bmatrix}$$

船体扰动与平台控制叠加：
$$q=q_s+q_c$$
其中
$$q_s=[z_s,\alpha_s,\beta_s]^T,\qquad q_c=[z_c,\alpha_c,\beta_c]^T$$

---

## 2. 运动学

### 2.1 基座（受船体扰动）

- $q_s = [z_s, \alpha_s, \beta_s]^T$：船体升沉、横滚、俯仰
- $B_0$：基座中心
- $B_{0,0}$：基座 nominal 位置（固定参考点）
- $B_{0,offset}$：基座相对于船的安装偏移

**基座中心位置**：
$$B_0(q_s) = B_{0,0} + \begin{bmatrix}0 \\ 0 \\ z_s\end{bmatrix} + R_s(\alpha_s, \beta_s) \cdot B_{0,offset}$$

其中 $R_s(\alpha_s, \beta_s)$ 是船体姿态旋转矩阵。

**基座第 $i$ 个铰点位置**（闭环几何中的下平台端点）：

令 $b_{local,i}\in\mathbb{R}^3$ 为第 $i$ 个基座铰点在基座坐标系 $\{B\}$ 下的固定局部坐标，则其在惯性系 $\{I\}$ 中的位置为
$$\boxed{\ B_i(q_s)=B_0(q_s)+R_s(\alpha_s,\beta_s)\,b_{local,i}\ }$$

若采用等边三点分布的默认几何（也便于数值初始化），可取
$$b_{local,i}=\begin{bmatrix} r_b\cos\theta_i\\ r_b\sin\theta_i\\ 0\end{bmatrix},\qquad \theta_i\in\Big\{0,\tfrac{2\pi}{3},\tfrac{4\pi}{3}\Big\}$$

> 上式仅提供一种常用参数化；若你的 CAD/安装坐标不同，请以实际 $b_{local,i}$ 为准。

### 2.2 动平台

- $q_c = [z_c, \alpha_c, \beta_c]^T$：控制位移和角度
- $P_0$：动平台中心
- $p_{local,i}$：第 $i$ 个铰链在平台坐标系中的位置

**动平台中心**：
$$P_0 = B_0 + R_s(\alpha_s, \beta_s) \cdot \begin{bmatrix}0 \\ 0 \\ z_c\end{bmatrix}$$

**姿态叠加**：
$$\alpha = \alpha_s + \alpha_c$$
$$\beta = \beta_s + \beta_c$$

**铰链位置**：
$$P_i = P_0 + R(\alpha, \beta) \cdot p_{local,i}$$

同理，动平台局部铰链坐标也可用等边三点分布写成
$$p_{local,i}=\begin{bmatrix} r_p\cos\theta_i\\ r_p\sin\theta_i\\ 0\end{bmatrix},\qquad \theta_i\in\Big\{0,\tfrac{2\pi}{3},\tfrac{4\pi}{3}\Big\}$$

> 上式同样仅为默认几何参数化，用于把 $r_p,\theta_i$ 与 $p_{local,i}$ 联系起来。

### 2.3 旋转矩阵（明确欧拉角序）

本文采用
$$R(\alpha,\beta)=R_y(\beta)R_x(\alpha)$$

其中
$$
R_x(\alpha)=
\begin{bmatrix}
1&0&0\\
0&\cos\alpha&-\sin\alpha\\
0&\sin\alpha&\cos\alpha
\end{bmatrix},\qquad
R_y(\beta)=
\begin{bmatrix}
\cos\beta&0&\sin\beta\\
0&1&0\\
-\sin\beta&0&\cos\beta
\end{bmatrix}
$$

从而
$$
R(\alpha,\beta)=
\begin{bmatrix}
\cos\beta & \sin\alpha\sin\beta & \cos\alpha\sin\beta\\
0 & \cos\alpha & -\sin\alpha\\
-\sin\beta & \sin\alpha\cos\beta & \cos\alpha\cos\beta
\end{bmatrix}
$$

并且其偏导为
$$\frac{\partial R}{\partial \alpha}=R_y(\beta)\frac{\partial R_x(\alpha)}{\partial \alpha},\qquad
\frac{\partial R}{\partial \beta}=\frac{\partial R_y(\beta)}{\partial \beta}R_x(\alpha)$$

其中
$$
\frac{\partial R_x}{\partial \alpha}=
\begin{bmatrix}
0&0&0\\
0&-\sin\alpha&-\cos\alpha\\
0&\cos\alpha&-\sin\alpha
\end{bmatrix},\qquad
\frac{\partial R_y}{\partial \beta}=
\begin{bmatrix}
-\sin\beta&0&\cos\beta\\
0&0&0\\
-\cos\beta&0&-\sin\beta
\end{bmatrix}
$$

---

## 3. 符号定义（补充维度）

| 符号 | 含义 |
|------|------|
| $B_i\in\mathbb{R}^3$ | 基座第 $i$ 个铰点在惯性系的位置 |
| $P_i\in\mathbb{R}^3$ | 动平台第 $i$ 个铰点在惯性系的位置 |
| $b_{local,i}\in\mathbb{R}^3$ | 基座第 $i$ 个铰点在基座系 $\{B\}$ 下的局部坐标（常量） |
| $R_s(\alpha_s,\beta_s)\in\mathrm{SO}(3)$ | 船体/基座姿态旋转矩阵（将 $\{B\}$ 向量旋到惯性系 $\{I\}$） |
| $L_i=\|P_i-B_i\|$ | 第 $i$ 根杆长度（驱动量） |
| $e_i=(P_i-B_i)/L_i\in\mathbb{R}^3$ | 第 $i$ 根杆方向单位向量 |
| $J(q)=\partial L/\partial q\in\mathbb{R}^{3\times 3}$ | 杆长对广义坐标雅可比 |
| $f=[f_1,f_2,f_3]^T\in\mathbb{R}^3$ | 三根杆的轴向推力 |

---

## 4. 动能

### 4.1 平台动能

$$T_{plate} = \frac{1}{2} M \dot{z}^2 + \frac{1}{2} I_{xx} \dot{\alpha}^2 + \frac{1}{2} I_{eq} \dot{\beta}^2$$

其中 $I_{eq} = I_{yy} \cos^2\alpha + I_{zz} \sin^2\alpha$。

### 4.2 杆件平动动能

**质心位置**（两段杆独立计算）：

下杆质心（第 $i$ 根）：
$$C_{d,i} = B_i + e_i \cdot \frac{l_d}{2}$$

上杆质心：
$$C_{u,i} = P_i - e_i \cdot \frac{l_u}{2}$$

**质心速度**（对时间求导）：

$$\dot{e}_i = \frac{\dot{P}_i - \dot{B}_i}{L_i} - \frac{(P_i - B_i)(\dot{P}_i - \dot{B}_i)^T(P_i - B_i)}{L_i^3}$$

下杆质心速度：
$$\dot{C}_{d,i} = \dot{B}_i + \dot{e}_i \cdot \frac{l_d}{2}$$

上杆质心速度：
$$\dot{C}_{u,i} = \dot{P}_i - \dot{e}_i \cdot \frac{l_u}{2}$$

**平动动能**：
$$T_{trans} = \sum_{i=1}^3 \left( \frac{1}{2} m_d \|\dot{C}_{d,i}\|^2 + \frac{1}{2} m_u \|\dot{C}_{u,i}\|^2 \right)$$

### 4.3 杆件转动动能

角速度（杆的方向向量变化率）：
$$\omega_i = \|\dot{e}_i\|$$

转动惯量：
- 上杆：$I_{u,i} = \frac{1}{12} m_u l_u^2$
- 下杆：$I_{d,i} = \frac{1}{12} m_d l_d^2$

**转动动能**：
$$T_{rot} = \sum_{i=1}^3 \left( \frac{1}{2} I_{u,i} \omega_i^2 + \frac{1}{2} I_{d,i} \omega_i^2 \right)$$

其中：
$$\|\dot{e}_i\|^2 = \frac{\|\dot{P}_i - \dot{B}_i\|^2 - (e_i^T(\dot{P}_i - \dot{B}_i))^2}{L_i^2}$$

### 4.4 总动能

$$T = T_{plate} + T_{trans} + T_{rot}$$

---

## 5. 势能

### 5.1 质心位置

下杆：$C_{d,i} = B_i + e_i \cdot \frac{l_d}{2}$

上杆：$C_{u,i} = P_i - e_i \cdot \frac{l_u}{2}$

### 5.2 势能表达（更严格）

仅考虑重力势能（不含弹性/阻尼势能），令 $e_z=[0,0,1]^T$，有
$$V(q)=g\Big(M\,e_z^T P_c(q)+\sum_{i=1}^3 m_d\,e_z^T C_{d,i}(q)+\sum_{i=1}^3 m_u\,e_z^T C_{u,i}(q)\Big)$$

其中 $P_c$ 为动平台质心位置。若动平台质心与 $P_0$ 重合，可令 $P_c\equiv P_0$。

---

## 6. 动力学方程（最终标准形式）

为避免与平台质量 $M$ 混淆，本节将质量矩阵仍写为 $\mathcal{M}(q)$。

### 6.1 拉格朗日方程

拉格朗日函数：
$$\mathcal{L}(q,\dot{q})=T(q,\dot{q})-V(q)$$

系统动力学满足
$$\frac{\mathrm{d}}{\mathrm{d}t}\left(\frac{\partial \mathcal{L}}{\partial \dot{q}}\right)-\frac{\partial \mathcal{L}}{\partial q}=\tau+\tau_s$$

其中：
- $\tau\in\mathbb{R}^3$：执行器（3 根杆）对 $q$ 的广义力
- $\tau_s\in\mathbb{R}^3$：船体扰动（基座运动）等效到 $q$ 的广义力/等效激励

### 6.2 执行器广义力 $\tau$

若将杆长 $L=[L_1,L_2,L_3]^T$ 视为驱动量，则速度关系为
$$\dot{L}=J(q)\dot{q}$$

虚功一致性给出
$$\tau = J(q)^T f$$

### 6.3 扰动广义力 $\tau_s$（用 $\ddot q_s,\dot q_s$ 的紧凑表示）

本文采用分解 $q=q_s+q_c$，因此
$$\dot q=\dot q_s+\dot q_c,\qquad \ddot q=\ddot q_s+\ddot q_c$$

将其代入名义动力学
$$\mathcal{M}(q)\ddot q+\mathcal{C}(q,\dot q)\dot q+G(q)=\tau$$

并把所有含 $(q_s,\dot q_s,\ddot q_s)$ 的项移到右端，可得到等效扰动广义力的一个标准写法：

$$\boxed{\ \mathcal{M}(q)\ddot q_c+\mathcal{C}(q,\dot q)\dot q_c+G(q)=\tau+\tau_s\ }$$

其中
$$\boxed{\ \tau_s\ \triangleq\ -\mathcal{M}(q)\ddot q_s-\mathcal{C}(q,\dot q)\dot q_s\ }$$

> 说明：该定义把船体运动对平台动力学的影响视为“已知输入激励”，并等效为外广义力 $\tau_s$。若进一步考虑 $\mathcal{C}(q,\dot q)$ 中 $\dot q=\dot q_s+\dot q_c$ 带来的交叉项，也可统一保持在 $\mathcal{C}(q,\dot q)$ 中（本式仍成立，因为 $\tau_s$ 只负责将与 $\dot q_s,\ddot q_s$ 线性相关的项移到右端）。

### 6.4 标准矩阵形式

将第 4、5 节得到的 $T,V$ 代入拉格朗日方程，最终可写为
$$\boxed{\ \mathcal{M}(q)\,\ddot{q}+\mathcal{C}(q,\dot{q})\,\dot{q}+G(q)=\tau+\tau_s\ }$$

其中各项定义为：
- $\mathcal{M}(q)\in\mathbb{R}^{3\times 3}$：由 $T=\frac12\dot q^T\mathcal{M}(q)\dot q$ 唯一确定
- $\mathcal{C}(q,\dot q)\dot q$：由 $\mathcal{M}(q)$ 的 Christoffel 符号构造（见 6.6）
- $G(q)=\partial V/\partial q\in\mathbb{R}^3$（见 6.7）
- $\tau=J(q)^T f$
- $\tau_s=-\mathcal{M}(q)\ddot q_s-\mathcal{C}(q,\dot q)\dot q_s$

### 6.6 科里奥利/离心矩阵 $\mathcal{C}(q,\dot q)$ 的 Christoffel 构造

令 $\mathcal{M}(q)=[\mathcal{M}_{ij}(q)]\in\mathbb{R}^{3\times 3}$，对 $q=[q_1,q_2,q_3]^T=[z,\alpha,\beta]^T$。

定义 Christoffel 符号
$$\Gamma_{ijk}(q)=\frac{1}{2}\left(\frac{\partial \mathcal{M}_{ij}}{\partial q_k}+\frac{\partial \mathcal{M}_{ik}}{\partial q_j}-\frac{\partial \mathcal{M}_{jk}}{\partial q_i}\right)$$

则速度二次项向量 $h(q,\dot q)\triangleq \mathcal{C}(q,\dot q)\dot q$ 满足
$$h_i(q,\dot q)=\sum_{j=1}^{3}\sum_{k=1}^{3}\Gamma_{ijk}(q)\,\dot q_j\,\dot q_k$$

若希望给出 $\mathcal{C}$ 的矩阵元，一个常用的取法为
$$\boxed{\ \mathcal{C}_{ij}(q,\dot q)=\sum_{k=1}^{3}\Gamma_{ijk}(q)\,\dot q_k\ }$$

该取法保证 $\mathcal{C}(q,\dot q)\dot q = h(q,\dot q)$。

### 6.7 重力向量 $G(q)$ 的三维分量表达

由势能 $V(q)$ 定义
$$G(q)=\frac{\partial V(q)}{\partial q}=\begin{bmatrix}G_z\\G_\alpha\\G_\beta\end{bmatrix}
=\begin{bmatrix}\frac{\partial V}{\partial z}\\\frac{\partial V}{\partial \alpha}\\\frac{\partial V}{\partial \beta}\end{bmatrix}
$$

结合第 5.2 节
$$V(q)=g\Big(M\,e_z^T P_c(q)+\sum_{i=1}^3 m_d\,e_z^T C_{d,i}(q)+\sum_{i=1}^3 m_u\,e_z^T C_{u,i}(q)\Big)$$

对每个点 $X\in\{P_c,C_{d,i},C_{u,i}\}$，有 $\partial(e_z^T X)/\partial q = e_z^T\,\partial X/\partial q$。

因此三维分量可写为
$$\boxed{\ G_z = g\Big(M\,e_z^T\frac{\partial P_c}{\partial z}+\sum_{i=1}^3 m_d\,e_z^T\frac{\partial C_{d,i}}{\partial z}+\sum_{i=1}^3 m_u\,e_z^T\frac{\partial C_{u,i}}{\partial z}\Big)\ }$$

$$\boxed{\ G_{\alpha} = g\Big(M\,e_z^T\frac{\partial P_c}{\partial \alpha}+\sum_{i=1}^3 m_d\,e_z^T\frac{\partial C_{d,i}}{\partial \alpha}+\sum_{i=1}^3 m_u\,e_z^T\frac{\partial C_{u,i}}{\partial \alpha}\Big)\ }$$

$$\boxed{\ G_{\beta} = g\Big(M\,e_z^T\frac{\partial P_c}{\partial \beta}+\sum_{i=1}^3 m_d\,e_z^T\frac{\partial C_{d,i}}{\partial \beta}+\sum_{i=1}^3 m_u\,e_z^T\frac{\partial C_{u,i}}{\partial \beta}\Big)\ }$$

其中
$$\frac{\partial C_{d,i}}{\partial q}=\frac{\partial B_i}{\partial q}+\frac{l_d}{2}\frac{\partial e_i}{\partial q},\qquad
\frac{\partial C_{u,i}}{\partial q}=\frac{\partial P_i}{\partial q}-\frac{l_u}{2}\frac{\partial e_i}{\partial q}$$

并约定：本文将 $q_s(t),q_c(t)$ 视为外部给定信号，$q=q_s+q_c$ 仅用于时间域的叠加关系（即 $\dot q=\dot q_s+\dot q_c$，$\ddot q=\ddot q_s+\ddot q_c$）。因此在构造 $\partial(\cdot)/\partial q$（对广义坐标 $q=[z,\alpha,\beta]^T$ 的偏导）时，保持 $q_s$ 不随 $q$ 变化，有
$$\boxed{\ \frac{\partial B_i(q_s)}{\partial q}=0\ }$$

在该偏导约定下
$$\frac{\partial e_i}{\partial q_k}=\frac{1}{L_i}(\mathbf I_3-e_ie_i^T)\frac{\partial P_i}{\partial q_k},\qquad q_k\in\{z,\alpha,\beta\}$$

> 若改用另一种等价参数化（例如把 $q$ 视为独立变量并用 $q_s=q-q_c$ 消元），则会产生 $\partial B_i/\partial q\neq 0$ 的链式项；这对应“将船体运动嵌入坐标变换”的扩展建模，与本文将 $\dot B_i,\ddot B_i$ 影响归并到扰动项 $\tau_s$ 的处理方式不同。

---

### 6.5 质量矩阵 $\mathcal{M}(q)$ 的显式表达（3×3）

令 $q=[z,\alpha,\beta]^T$。在本文建模中，总质量矩阵由“平台 + 三根两段式杆”的动能叠加得到：
$$\mathcal{M}(q)=\mathcal{M}_{plate}(q)+\mathcal{M}_{rod,trans}(q)+\mathcal{M}_{rod,rot}(q)$$

#### 6.5.1 平台部分 $\mathcal{M}_{plate}(q)$

采用你在第 4.1 节的动能写法（仅保留 3 个自由度），平台部分对应
$$
\mathcal{M}_{plate}(q)=
\begin{bmatrix}
M & 0 & 0\\
0 & I_{xx} & 0\\
0 & 0 & I_{yy}\cos^2\alpha+I_{zz}\sin^2\alpha
\end{bmatrix}
$$

> 注：这等价于在 3-DOF 情况下采用一个“等效”转动惯量 $I_{eq}(\alpha)$ 来描述绕 $y$ 轴方向的动能项。

#### 6.5.2 杆件平动部分 $\mathcal{M}_{rod,trans}(q)$

你在两段式杆的平动动能中，已经把单根杆的平动动能写成关于 $\dot P_i,\dot B_i$ 的二次型。

若在构造质量矩阵时将基座扰动速度视为外部输入，并取 $\dot B_i$ 的项归并到 $\tau_s$（即只保留關於 $\dot q$ 的二次型），則每根杆平動動能對 $\dot P_i$ 的“純二次項”系數為
$$c_P=\frac{m_d+9m_u}{32}$$

而 $\dot P_i = J_{P_i}(q)\,\dot q$，其中 $J_{P_i}=\frac{\partial P_i}{\partial q}\in\mathbb{R}^{3\times 3}$。

因此杆件平动质量矩阵可写为
$$\boxed{\ \mathcal{M}_{rod,trans}(q)=c_P\sum_{i=1}^3 J_{P_i}(q)^T J_{P_i}(q)\ }$$

其中（由 $P_i=P_0+R(\alpha,\beta)p_{local,i}$）
$$
J_{P_i}(q)=\left[\ \frac{\partial P_i}{\partial z}\ \ \frac{\partial P_i}{\partial \alpha}\ \ \frac{\partial P_i}{\partial \beta}\ \right]
=\left[\ e_z\ \ \frac{\partial R}{\partial \alpha}p_{local,i}\ \ \frac{\partial R}{\partial \beta}p_{local,i}\ \right]
$$

（其中 $e_z=[0,0,1]^T$。）

> 若不把 $\dot B_i$ 归并到扰动项，而是希望把 $\dot q_s$ 也作为扩展状态，则会出现 $\dot q` 与 $\dot q_s$ 的耦合质量项；本文采用 $\tau_s$ 吸收扰动的写法，因此这里保留關於 $\dot q$ 的有效質量矩陣即可。

#### 6.5.3 杆件转动部分 $\mathcal{M}_{rod,rot}(q)$

对第 $i$ 根杆，令
$$r_i(q)=P_i(q)-B_i,\qquad L_i=\|r_i\|,\qquad e_i=\frac{r_i}{L_i}$$

转动动能采用细杆近似、且两段共线（角速度相同），则第 $i$ 根杆的等效转动惯量为
$$I_{\perp,rod}=I_d+I_u=\frac{1}{12}m_d l_d^2+\frac{1}{12}m_u l_u^2$$

并用方向向量的变化率构造角速度大小：$\omega_i\approx\|\dot e_i\|$。

当 $B_i$ 作为常量（或其变化项归入 $\tau_s$）时，有
$$\dot e_i = \frac{\partial e_i}{\partial q}\,\dot q$$

其中对任意 $q_k\in\{z,\alpha,\beta\}$：
$$
\frac{\partial e_i}{\partial q_k}=\frac{1}{L_i}\left(\mathbf I_3-e_ie_i^T\right)\frac{\partial P_i}{\partial q_k}
$$

将 $\frac{\partial e_i}{\partial q}=[\partial e_i/\partial z\ \partial e_i/\partial \alpha\ \partial e_i/\partial \beta]\in\mathbb{R}^{3\times 3}$ 记为 $E_i(q)$，则
$$\|\dot e_i\|^2=\dot q^T\,E_i(q)^T E_i(q)\,\dot q$$

故杆件转动质量矩阵为
$$\boxed{\ \mathcal{M}_{rod,rot}(q)=I_{\perp,rod}\sum_{i=1}^3 E_i(q)^T E_i(q)\ }$$

> 注：本文中 $\mathbf I_3$ 表示 $3\times 3$ 单位阵；$I_{\perp,rod}$ 表示细杆关于任意垂直于杆轴的转动惯量（标量）。为避免与单位阵混淆，不再使用 $I_{eq,rod}$ 这一记号。

---

以上三部分相加即得到 $\mathcal{M}(q)$ 的显式矩阵形式。
