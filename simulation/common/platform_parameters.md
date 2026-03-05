# Parallel Platform (3-UPS/PU) Parameters

This document summarizes the **structural (geometric)** and **mass/inertia** parameters used by the 3-DOF parallel platform model implemented in `simulation/common/platform_dynamics.py` (`ParallelPlatform3DOF`).

All values below are the **default constructor values** in code unless otherwise stated.

---

## Coordinate system and DOFs (model reminder)

The model uses 3 DOFs:

- $z$: heave translation along global $Z$ (m)
- $\alpha$: roll about $X$ (rad)
- $\beta$: pitch about $Y$ (rad)

State:

- $q = [z,\ \alpha,\ \beta]^T$
- $\dot q = [\dot z,\ \dot \alpha,\ \dot \beta]^T$

Actuation:

- $u = [F_1, F_2, F_3]^T$ (three leg/cylinder axial forces)

---

## 1) Structural / geometric parameters

### 1.1 Main geometry

| Parameter | Symbol | Default | Unit | Meaning |
|---|---:|---:|---|---|
| Base joint radius | $r_{base}$ | 0.65 | m | Radius of the 3 base joints (equally spaced by 120°) |
| Platform joint radius | $r_{platform}$ | 0.58 | m | Radius of the 3 moving-platform joints (equally spaced by 120°) |
| Nominal leg length | $l_{leg,nom}$ | 1.058 | m | Nominal/neutral leg length |
| Minimum leg length | $l_{leg,min}$ | 0.808 | m | Leg stroke lower limit |
| Maximum leg length | $l_{leg,max}$ | 1.308 | m | Leg stroke upper limit |

### 1.2 Motion constraints (limits)

| Constraint | Symbol | Default | Unit | Meaning |
|---|---:|---:|---|---|
| Roll limit | $|\alpha| \le \alpha_{max}$ | $\pi/6 \approx 0.5236$ | rad | Max roll angle |
| Pitch limit | $|\beta| \le \beta_{max}$ | $\pi/6 \approx 0.5236$ | rad | Max pitch angle |
| Heave lower bound | $z \ge z_{min}$ | 0.808 | m | Lower bound used in `check_constraints()` |
| Heave upper bound | $z \le z_{max}$ | 1.308 | m | Upper bound used in `check_constraints()` |

> Note: In the current implementation, `z_min/z_max` numerically match the leg length limits. If you interpret $z$ as a *relative displacement* (instead of an *absolute height/leg-length proxy*), you may want to redefine these bounds accordingly.

---

## 2) Mass and inertia parameters

### 2.1 Mass properties

| Parameter | Symbol | Default | Unit | Meaning |
|---|---:|---:|---|---|
| Platform mass | $m$ | 347.54 | kg | `m_platform` |
| Cylinder mass (parameter) | $m_{cyl}$ | 25.0 | kg | `m_cylinder` (kept as a parameter; not explicitly added into $M(q)$ in current code) |
| Gravity | $g$ | 9.81 | m/s² | gravitational acceleration |

### 2.2 Moments of inertia

| Parameter | Symbol | Default | Unit | Meaning |
|---|---:|---:|---|---|
| Roll inertia | $I_{xx}$ | 60.64 | kg·m² | about the body/global x-axis used for roll dynamics |
| Pitch inertia | $I_{yy}$ | 115.4 | kg·m² | about the body/global y-axis |
| Yaw inertia (auxiliary) | $I_{zz}$ | 80.0 | kg·m² | used in $M(q)$ and Coriolis coupling terms |

---

## 3) Actuator / input limits and friction parameters

### 3.1 Input limits

| Parameter | Symbol | Default | Unit | Meaning |
|---|---:|---:|---|---|
| Force upper limit | $u_{max}$ | 30000 | N | per-leg maximum axial force |
| Force lower limit | $u_{min}$ | -30000 | N | per-leg minimum axial force |
| Velocity limit | $v_{max}$ | 0.2 | m/s | velocity limit parameter (used by controllers/constraints) |

### 3.2 Friction model (smooth approximation)

The friction vector is modeled as:

$$F(\dot q) = F_c\,\tanh(k\dot q) + F_v\,\dot q,$$

with:

| Parameter | Symbol | Default | Units | Meaning |
|---|---:|---:|---|---|
| Coulomb friction amplitude | $F_c$ | 100.0 | (model units) | magnitude of Coulomb-like friction |
| Viscous coefficient | $F_v$ | 500.0 | (model units) | viscous friction coefficient |
| Smoothness factor | $k$ | 50.0 | – | higher is closer to `sign(·)` |

---

## 4) Mapping to the wave disturbance model (`wave_disturbance.py`)

`simulation/disturbance/wave_disturbance.py` accepts a `platform_params` dictionary.

Current fields used by `WaveDisturbance`:

| `platform_params` key | Meaning | Typical source in `ParallelPlatform3DOF` |
|---|---|---|
| `m` | platform mass (kg) | `m_platform` |
| `Ixx` | roll inertia (kg·m²) | `Ixx` |
| `Iyy` | pitch inertia (kg·m²) | `Iyy` |
| `r` | platform joint radius (m) | `r_platform` |

Recommended consistent export (example):

- `m = 347.54`
- `Ixx = 60.64`
- `Iyy = 115.4`
- `r = 0.58`

---

## Source

- Parameters extracted from: `simulation/common/platform_dynamics.py` → `ParallelPlatform3DOF.__init__()` default arguments.
