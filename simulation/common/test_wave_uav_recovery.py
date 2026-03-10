"""test_wave_uav_recovery.py

快速回归测试：联合海浪+UAV冲击工况下，平台总位姿 q=q_s+q_c 在 touchdown 后应回到 0 附近。

这不是严格的控制性能评测，只做“不会长期偏置/发散”的健康检查。
"""

from __future__ import annotations

import numpy as np

from simulation.common.sim_uniform_rod_with_wave_and_uav_impact_demo import simulate


def test_uav_impact_recovers_near_zero():
    sim = simulate(t_end=20.0, dt=0.01, uav_touchdown=6.0)
    t = sim["t"]
    q = sim["q_s"] + sim["q_c"]

    # check a window sufficiently after touchdown
    mask = (t >= 12.0) & (t <= 20.0)
    q_tail = q[mask]

    # z in meters; angles in radians
    # allow small residual due to persistent wave excitation + numerical discretization
    z_mean = float(np.mean(q_tail[:, 0]))
    a_mean = float(np.mean(q_tail[:, 1]))
    b_mean = float(np.mean(q_tail[:, 2]))

    assert abs(z_mean) < 5e-3  # 5 mm
    assert abs(a_mean) < np.deg2rad(0.15)
    assert abs(b_mean) < np.deg2rad(0.15)
