# 海浪扰动模型符号系统规范

## 一、符号命名原则

1. **理论公式**：使用希腊字母（符合Fossen和海洋工程惯例）
2. **代码实现**：使用英文字母（便于编程）
3. **统一映射**：希腊字母与代码变量一一对应

---

## 二、核心符号对照表

### 波浪参数

| 希腊符号 | 英文代码 | LaTeX | 含义 | 单位 |
|---------|---------|-------|------|------|
| Hs | `Hs` | `H_s` | 有义波高 | m |
| T0 | `T0` | `T_0` | 峰值周期 | s |
| ω0 | `omega_0` | `\omega_0` | 主导频率 | rad/s |
| S(ω) | `S` | `S(\omega)` | 波能谱 | m²·s |
| η(t) | `eta` | `\eta(t)` | 波面高度 | m |

### 响应参数

| 希腊符号 | 英文代码 | LaTeX | 含义 | 单位 |
|---------|---------|-------|------|------|
| λ | `lambda_d` | `\lambda` | 阻尼比(Fossen) | - |
| ζ | `zeta` | `\zeta` | 阻尼比(3dof) | - |
| ωn | `omega_n` | `\omega_n` | 固有频率 | rad/s |
| σ | `sigma` | `\sigma` | 波强度参数 | m |
| Kw | `Kw` | `K_w` | 波浪增益 | varies |

### 扰动力

| 希腊符号 | 英文代码 | LaTeX | 含义 | 单位 |
|---------|---------|-------|------|------|
| τ_wave | `tau_wave` | `\tau_{wave}` | 波浪扰动 | N, N·m |
| τ_wave1 | `tau_wave1` | `\tau_{wave1}` | 一阶波浪力 | N, N·m |
| τ_wave2 | `tau_wave2` | `\tau_{wave2}` | 二阶波浪力 | N, N·m |
| d(t) | `disturbance` | `d(t)` | 扰动向量 | N, N·m |

### 力和力矩分量

| 符号 | 英文代码 | 含义 | 单位 |
|------|---------|------|------|
| Fz | `Fz` | 垂荡力 | N |
| Mα | `M_alpha` | 横摇力矩 | N·m |
| Mβ | `M_beta` | 纵摇力矩 | N·m |
| X, Y, Z | `X, Y, Z` | 纵向/横向/垂向力 | N |
| K, M, N | `K, M, N` | 横摇/纵摇/艏摇力矩 | N·m |

### 运动学参数

| 符号 | 英文代码 | LaTeX | 含义 | 单位 |
|------|---------|-------|------|------|
| z | `z` | `z` | 升沉位移 | m |
| α | `alpha` | `\alpha` | 横摇角 | rad |
| β | `beta` | `\beta` | 纵摇角 | rad |

---

## 三、参数对应关系

### 3dofPlatformSim ↔ Fossen 参数映射

| 3dof参数 | Fossen参数 | 关系式 | 含义 |
|---------|-----------|--------|------|
| `gain` | `2*lambda*omega_0*sigma` | K = 2λω₀σ | 增益系数 |
| `wn` | `omega_0` | ωn = ω₀ | 固有频率 |
| `zeta` | `lambda` | ζ = λ | 阻尼比 |

---

## 四、代码命名规范

### 类名
```python
class WaveDisturbance          # 基础波浪扰动
class WaveDisturbanceRAO       # 基于RAO的扰动  
class WaveDisturbanceLinear    # 二阶线性近似
class ITTCWaveModel            # ITTC波面模型
```

### 方法名
```python
def compute_ittc_spectrum(omega)     # ITTC谱计算
def generate_wave_elevation(t)       # 生成波面
def generate_disturbance(t)          # 生成扰动力
def state_space_step(dt, w)          # 状态空间单步
```

### 变量名
```python
# 波浪参数
Hs, T0, omega_0, lambda_d, sigma, Kw

# 频域
omega, domega, S, freqs

# 时域
t, dt, total_time

# 波面
eta, A_k, epsilon_k, phases

# 扰动
tau_wave, tau_wave1, tau_wave2
disturbance, d_z, d_alpha, d_beta

# 状态空间
x_w, A_w, B_w, C_w, w_noise
```

---

## 五、推荐参数值

### Sea State 4（中等海况）
```python
# 海况参数
Hs = 2.0          # m
T0 = 8.0          # s
omega_0 = 0.785   # rad/s (2*pi/T0)

# 响应参数（垂荡）
lambda_d = 0.15   # 或 zeta = 0.15
omega_n = 1.0     # rad/s
Kw = 2*lambda_d*omega_0*sigma  # 自动计算

# 响应参数（横摇/纵摇）
lambda_d = 0.20
omega_n = 0.9
```

### 希腊字母输入参考
- ω: `\omega`
- λ: `\lambda`  
- ζ: `\zeta`
- σ: `\sigma`
- η: `\eta`
- ε: `\epsilon`
- τ: `\tau`
- α: `\alpha`
- β: `\beta`

---

## 六、LaTeX公式模板

### 波能谱（ITTC）
```latex
S(\omega) = \frac{173H_s^2}{T_1^4\omega^5}
\exp\left(-\frac{691}{T_1^4\omega^4}\right)
```

### 二阶响应函数
```latex
|H(j\omega)| = \frac{K_w}{\sqrt{(\omega_0^2-\omega^2)^2+(2\lambda\omega_0\omega)^2}}
```

### 状态空间
```latex
\dot{x}_w = A_w x_w + B_w w
y_w = C_w^T x_w
```

### 扰动力
```latex
\tau_{wave}(t) = \sum_{k=1}^{N} |H(j\omega_k)| A_k \cos(\omega_k t + \epsilon_k)
```

---

**注意**：
1. `lambda_d` 中的 `_d` 是为了避免与Python关键字 `lambda` 冲突
2. 所有角度单位默认为 **弧度(rad)**
3. 时间单位默认为 **秒(s)**
4. 长度单位默认为 **米(m)**
