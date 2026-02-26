# Chapter 4 数学表达与实现对照（`rl_env_chapter4.py`）

本文档用于把 `simulation/chapter4/env/rl_env_chapter4.py` 中的离散时间更新与“实际总扰动”反推写成可用于论文的数学表达，并明确代码中各项取值时刻（$k$ 或 $k+1$）。

> 约定：离散时间 $t_k = k\Delta t$，其中 $\Delta t = \texttt{dt}$。

## 1. 连续时间动力学模型（用于定义符号）

以 3 自由度平台的广义坐标 $q\in\mathbb{R}^3$、速度 $\dot q\in\mathbb{R}^3$ 表示系统。平台动力学写成：

$$
M(q)\,\ddot q + C(q,\dot q)\,\dot q + G(q) = u + d,
$$

其中：

- $M(q)\in\mathbb{R}^{3\times 3}$：质量矩阵；
- $C(q,\dot q)\in\mathbb{R}^{3\times 3}$：科氏/离心项矩阵（实现中以 $C\dot q$ 的形式出现）；
- $G(q)\in\mathbb{R}^{3}$：重力项；
- $u\in\mathbb{R}^{3}$：控制输入（代码中为 `u_legs`）；
- $d\in\mathbb{R}^{3}$：总扰动（代码中由波浪生成的 `disturbance`，以及后文反推的“实际总扰动” `d_actual`）。

## 2. 代码中的离散更新（含外部扰动）

### 2.1 名义加速度（无扰动）

在第 $k$ 步，代码调用正向动力学得到名义加速度：

$$
\ddot q_k^{\mathrm{nom}} = f\bigl(q_k,\dot q_k,u_k\bigr).
$$

对应实现：

- `qdd = platform.forward_dynamics(self.q, self.qd, u_legs)`

### 2.2 扰动等效到加速度通道

代码把波浪扰动视为加在广义力通道的扰动 $d_k$，并用 $M^{-1}$ 等效为加速度增量：

$$
\ddot q_k^{\mathrm{dist}} = \ddot q_k^{\mathrm{nom}} + M(q_k)^{-1} d_k.
$$

对应实现：

- `M = platform.mass_matrix(self.q)`（此处的 $M$ 按 **更新前** 的 $q_k$ 计算）
- `qdd_disturbed = qdd + M_inv @ disturbance`

### 2.3 半隐式欧拉积分（semi-implicit Euler）

代码先更新速度，再用更新后的速度更新位置：

$$
\dot q_{k+1} = \dot q_k + \ddot q_k^{\mathrm{dist}}\,\Delta t,
$$

$$
q_{k+1} = q_k + \dot q_{k+1}\,\Delta t.
$$

对应实现：

- `self.qd = self.qd + qdd_disturbed * self.dt`
- `self.q  = self.q  + self.qd * self.dt`

## 3. 实际加速度与“实际总扰动”反推

### 3.1 实际加速度的离散估计

代码用一步速度差分估计该步“实际加速度”（更精确地说，是区间上的平均加速度）：

$$
\ddot q_k^{\mathrm{actual}} \approx \frac{\dot q_{k+1} - \dot q_k}{\Delta t}.
$$

对应实现：

- 记 `qd_before = self.qd.copy()` 为更新前 $\dot q_k$
- 更新后 `self.qd` 为 $\dot q_{k+1}$
- `qdd_actual = (self.qd - qd_before) / self.dt`

### 3.2 “实际总扰动”反推公式（理想一致形式）

如果在同一时刻 $(q,\dot q,\ddot q)$ 上严格满足连续模型

$$
M(q)\,\ddot q + C(q,\dot q)\,\dot q + G(q) = u + d,
$$

则扰动可由

$$
d = M(q)\,\ddot q + C(q,\dot q)\,\dot q + G(q) - u
$$

反推出。

### 3.3 代码中的时刻取值（需要在论文里说明的“近似不一致”）

`rl_env_chapter4.py` 里的 `d_actual` 计算并非严格使用同一时刻的 $(q,\dot q)$：

- $M$ 使用更新前的 $q_k$（因为 `M` 在积分前计算）；
- $C$、$G$ 使用更新后的 $(q_{k+1},\dot q_{k+1})$（因为它们在积分后计算）；
- 速度项乘的是更新前速度 $\dot q_k$（代码用 `qd_before`）。

因此，代码严格对应的离散表达应写为：

$$
\boxed{
\;d_k^{\mathrm{actual}}
= M(q_k)\,\ddot q_k^{\mathrm{actual}}
+ C(q_{k+1},\dot q_{k+1})\,\dot q_k
+ G(q_{k+1})
- u_k.\;}
$$

对应实现：

- `d_actual = M @ qdd_actual + C @ qd_before + G - u_legs`

> 说明：从数值分析角度，这是一种“混合取点”的离散近似。若希望表达更简洁、与连续模型更一致，论文中可写理想一致形式 $d\!=\!M(q_{k+1})\ddot q_k^{actual}+C(q_{k+1},\dot q_{k+1})\dot q_{k+1}+G(q_{k+1})-u_k$，并在实现细节处注明代码实际采用的取点方式。

## 4. 误差、积分量与 reward（避免 ISE 概念歧义）

令期望轨迹为 $q_k^{\mathrm{des}}$，则跟踪误差：

$$
e_k = q_k - q_k^{\mathrm{des}}.
$$

代码中每步的“ISE”记录量为：

$$
\mathrm{ise}_k = \|e_k\|_2^2 = e_k^\top e_k,
$$

并采用逐步 reward：

$$
r_k = -\mathrm{ise}_k.
$$

同时，代码维护了误差积分（用于 `info` 记录，而 **不** 进入 reward）：

$$
E_{k+1} = E_k + e_{k+1}\,\Delta t.
$$

这意味着：

- 训练优化的目标主要是最小化逐步误差平方（running cost）；
- episode 级的离散 ISE 可由

$$
J_{\mathrm{ISE}} \approx \sum_{k=0}^{T-1} \|e_k\|_2^2\,\Delta t
$$

给出。

---

**文件关联**：
- 实现：`simulation/chapter4/env/rl_env_chapter4.py`（`step()` 中扰动反推与状态更新部分）
