# 海浪扰动建模方法

## 1. 基本思路

船载平台受到的扰动是**惯性力**，由船体运动加速度引起：

$$
\vec{F} = m\vec{a}, \quad \vec{M} = I\vec{\alpha}
$$

## 2. 计算流程

### Step 1: 波面合成

ITTC谱描述波能分布：

$$
S(\omega) = \frac{173H_s^2}{T_1^4\omega^5}\exp\left(-\frac{691}{T_1^4\omega^4}\right)
$$

波面高度：

$$
\eta(t) = \sum_{k=1}^N A_k\cos(\omega_k t + \phi_k), \quad A_k = \sqrt{2S(\omega_k)\Delta\omega}
$$

### Step 2: 船体运动 (RAO)

响应幅值算子描述船体对波浪的响应：

$$
X_k = \text{RAO}(\omega_k) \cdot A_k
$$

RAO采用钟形曲线模型：

$$
\text{RAO}(\omega) = \frac{\text{RAO}_{\max}}{\sqrt{1+((\omega-\omega_n)/(\zeta\omega_n))^2}}
$$

### Step 3: 加速度

简谐运动加速度公式：

$$
a_k = \omega_k^2 X_k = \omega_k^2 \cdot \text{RAO}(\omega_k) \cdot A_k
$$

### Step 4: 惯性力

牛顿第二定律：

$$
F_k = m \cdot a_k = m\omega_k^2 \cdot \text{RAO}(\omega_k) \cdot A_k
$$

力矩：

$$
M_{\alpha,k} = I_{xx}\omega_k^2 \cdot \text{RAO}_{\text{roll}}(\omega_k) \cdot A_k
$$

$$
M_{\beta,k} = I_{yy}\omega_k^2 \cdot \text{RAO}_{\text{pitch}}(\omega_k) \cdot A_k
$$

### Step 5: 时域合成

$$
\tau_{\text{wave}}(t) = \sum_{k=1}^N [F_k, M_{\alpha,k}, M_{\beta,k}]^T \cos(\omega_k t + \phi_k)
$$

## 3. 参数表

### 平台参数（大尺度平台）
- $m = 347.54$ kg（质量×10）
- $I_{xx} = 60.64$ kg·m², $I_{yy} = 115.4$ kg·m²（惯性矩×40）
- 平台半径 $r = 0.58$ m（尺寸×2）

### 船舶参数（50吨级）
- $L=30$m, $B=8$m, $T=2.5$m, $GM=1.0$m

### 船舶固有频率
- 垂荡：$\omega_{heave} = \sqrt{g/T} = 1.98$ rad/s
- 横摇：$\omega_{roll} = \sqrt{gGM/(B/2)^2} = 0.78$ rad/s
- 纵摇：$\omega_{pitch} = \sqrt{gGM/(L/2)^2} = 0.21$ rad/s

### RAO峰值
| 自由度 | RAO_max | ζ |
|--------|---------|---|
| 垂荡 | 0.8 | 0.15 |
| 横摇 | 0.6 | 0.20 |
| 纵摇 | 0.4 | 0.20 |

## 4. 物理意义

1. **RAO**：船体运动响应特性，无量纲
2. **$a = \omega^2 X$**：简谐运动加速度公式
3. **$F = ma$**：牛顿第二定律，惯性力来源

扰动力本质是平台跟随船体运动产生的**惯性效应**。
