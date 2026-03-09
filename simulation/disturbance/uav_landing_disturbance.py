"""uav_landing_disturbance.py

UAV 在平台上“接住/降落”造成的突变扰动模型（3DOF: [z, alpha, beta]）。

坐标与符号约定
- 广义坐标顺序：q = [z, alpha, beta]
  - z：竖直平移（m）
  - alpha：roll，绕 X 轴（rad）
  - beta：pitch，绕 Y 轴（rad）
- 注意：本文件按你的要求采用：z 轴正向为重力方向。
  因此 UAV 重量对应的外力为 +Fz，其中 Fz = m_uav * g。

落点与力矩
- 设 UAV 落点相对平台中心的偏心距 r = [r_x, r_y]（单位 m），范围要求在半径 R=0.3m 圆内。
- 纯竖直力 F = [0,0,Fz] 作用在 r 处，对平台中心产生力矩：
    tau_rot = r x F
  即：
    Mx = r_y * Fz
    My = -r_x * Fz

映射到 3DOF 广义力
- tau = [tau_z, tau_alpha, tau_beta]
    tau_z     = Fz
    tau_alpha = Mx
    tau_beta  = My

时间剖面（突变 + 冲量）
- 静载：使用平滑阶跃（线性 ramp）避免数值抖动。
- 冲量：额外叠加一个短时半正弦脉冲，面积满足：∫F_imp dt = impulse_Iz (N·s)。
  相应的力矩冲量按同一偏心距比例映射到 alpha/beta 通道。

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class UavLandingParams:
    m_uav: float = 20.0
    g: float = 9.81
    t0: float = 20.0
    ramp: float = 0.2
    duration: Optional[float] = None

    # impulse (N·s) carried at landing (additional to static weight)
    impulse_Iz: float = 0.0
    # pulse duration for impulse (s)
    impulse_duration: float = 0.05

    # landing offset (m)
    r_x: Optional[float] = None
    r_y: Optional[float] = None
    radius_limit: float = 0.3
    random_seed: Optional[int] = 0


def sample_landing_offset(
    *,
    radius_limit: float = 0.3,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """在半径 radius_limit 的圆盘内均匀采样 (r_x, r_y)。"""
    if rng is None:
        rng = np.random.default_rng()

    # uniform disk: r = sqrt(u) * R
    u = rng.random()
    theta = 2 * np.pi * rng.random()
    r = np.sqrt(u) * radius_limit
    return float(r * np.cos(theta)), float(r * np.sin(theta))


def _ramp_profile(t: np.ndarray, *, t0: float, ramp: float, duration: Optional[float]) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    ramp = float(max(ramp, 0.0))

    if ramp <= 0:
        on = (t >= t0).astype(float)
    else:
        on = np.clip((t - t0) / ramp, 0.0, 1.0)

    if duration is None:
        return on

    t1 = t0 + float(duration)
    if ramp <= 0:
        off = (t < t1).astype(float)
        return on * off

    off = np.clip((t1 + ramp - t) / ramp, 0.0, 1.0)
    return on * off


def _half_sine_pulse(t: np.ndarray, *, t0: float, duration: float) -> np.ndarray:
    r"""半正弦脉冲，支撑区间 [t0, t0+duration]，峰值为 1。

    p(t) = sin(pi * (t-t0)/duration), t in [t0,t0+duration]
    面积: \int p dt = 2*duration/pi
    """
    t = np.asarray(t, dtype=float)
    duration = float(max(duration, 1e-9))
    x = (t - t0) / duration
    p = np.zeros_like(t)
    mask = (x >= 0.0) & (x <= 1.0)
    p[mask] = np.sin(np.pi * x[mask])
    return p


def generate_uav_landing_tau(
    t: np.ndarray,
    *,
    params: UavLandingParams,
) -> tuple[np.ndarray, dict]:
    """生成 UAV 降落扰动的广义力 tau_uav(t)。

    Returns:
        tau_uav: shape (N,3) for [z, alpha, beta]
        meta: includes chosen (r_x, r_y), Fz, moments, impulse info
    """
    t = np.asarray(t, dtype=float)

    rng = np.random.default_rng(params.random_seed)
    if params.r_x is None or params.r_y is None:
        r_x, r_y = sample_landing_offset(radius_limit=params.radius_limit, rng=rng)
    else:
        r_x, r_y = float(params.r_x), float(params.r_y)
        if (r_x**2 + r_y**2) > params.radius_limit**2 + 1e-12:
            raise ValueError(f"landing offset ({r_x},{r_y}) exceeds radius_limit={params.radius_limit}")

    Fz = float(params.m_uav) * float(params.g)  # +z is gravity direction
    Mx = r_y * Fz
    My = -r_x * Fz

    s = _ramp_profile(t, t0=params.t0, ramp=params.ramp, duration=params.duration)

    # impulse pulse (area equals impulse_Iz)
    p = _half_sine_pulse(t, t0=params.t0, duration=params.impulse_duration)
    pulse_area = 2.0 * float(params.impulse_duration) / np.pi
    Fz_imp_amp = 0.0
    if abs(float(params.impulse_Iz)) > 0:
        Fz_imp_amp = float(params.impulse_Iz) / pulse_area

    # corresponding moment impulse shares same force profile scaled by lever arm
    Mx_imp_amp = r_y * Fz_imp_amp
    My_imp_amp = -r_x * Fz_imp_amp

    tau = np.zeros((t.size, 3), dtype=float)
    # static weight (step/ramp)
    tau[:, 0] = Fz * s
    tau[:, 1] = Mx * s
    tau[:, 2] = My * s

    # impulse part (pulse)
    tau[:, 0] += Fz_imp_amp * p
    tau[:, 1] += Mx_imp_amp * p
    tau[:, 2] += My_imp_amp * p

    meta = {
        "m_uav": float(params.m_uav),
        "g": float(params.g),
        "t0": float(params.t0),
        "ramp": float(params.ramp),
        "duration": None if params.duration is None else float(params.duration),
        "radius_limit": float(params.radius_limit),
        "r_x": float(r_x),
        "r_y": float(r_y),
        "Fz": float(Fz),
        "Mx": float(Mx),
        "My": float(My),
        "impulse_Iz": float(params.impulse_Iz),
        "impulse_duration": float(params.impulse_duration),
        "impulse_peak_Fz": float(Fz_imp_amp),
    }
    return tau, meta
