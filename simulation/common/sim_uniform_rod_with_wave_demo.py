"""sim_uniform_rod_with_wave_demo.py

更详细的 uniform-rod 3DOF 时域仿真与绘图 demo。

仿真方程（控制形式，来自 dynamics_3ups_uniform.md）：
    M(q) qdd_c + C(q, qd) qd_c + G(q) = tau + tau_s
其中
    q = q_s + q_c
    qd = qd_s + qd_c
    tau_s = -M(q) qdd_s - C(q, qd) qd_s

控制器：简单 PD（示例用，不代表最优控制）。

输出：生成多张图（每张图 3 个自由度 z/alpha/beta）：
- 位置：q_s / q_c / q = q_s+q_c
- 速度：qd_s / qd_c / qd
- 加速度：qdd_s / qdd_c / qdd
- 力：tau（控制）、tau_s（扰动）、tau_total=tau+tau_s

默认输出目录：`simulation/common/figures_uniform/`
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# 允许直接从任意 cwd 运行
_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation.common.uniform_rod_platform_dynamics import UniformRodPlatform3DOF
from simulation.common.controllers import (
    ComputedTorqueTrackingController3DOF,
    SimplePDGravityController3DOF,
)
from simulation.disturbance.wave_disturbance import WaveDisturbance


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _plot_3dof(
    t: np.ndarray,
    series: dict,
    *,
    title: str,
    ylabels: tuple[str, str, str],
    out_path: Path,
):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    dof_names = ("z", "alpha", "beta")

    for r in range(3):
        ax = axes[r]
        for name, y in series.items():
            ax.plot(t, y[:, r], label=name, alpha=0.9)
        ax.set_ylabel(ylabels[r])
        ax.set_title(f"{title} / {dof_names[r]}")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=3, fontsize=9)

    axes[-1].set_xlabel("t (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def simulate(t_end: float = 20.0, dt: float = 0.01):
    t = np.arange(0.0, t_end + 1e-12, dt)

    # 平台模型
    platform = UniformRodPlatform3DOF(diff_eps=1e-6)

    # 波浪扰动（输出 ship_state，内部单位 rad；这里按 3DOF 用 heave/roll/pitch）
    wd = WaveDisturbance()
    ship = wd.generate_ship_motion(t)

    # ship motion -> 3DOF: [z, alpha, beta] 对应 [heave, roll, pitch]
    q_s = ship["q_s"][:, [2, 3, 4]]
    qd_s = ship["qd_s"][:, [2, 3, 4]]
    qdd_s = ship["qdd_s"][:, [2, 3, 4]]

    n = t.size
    q_c = np.zeros((n, 3))
    qd_c = np.zeros((n, 3))
    qdd_c_hist = np.zeros((n, 3))

    # 跟踪目标：默认希望上平台总位姿 q = q_s + q_c 跟踪到 0
    # 你也可以改成常值，比如 q_ref = [1.05, 0, 0] 等
    q_ref = np.zeros((n, 3))
    qd_ref = np.zeros((n, 3))
    qdd_ref = np.zeros((n, 3))

    # 控制器：未知扰动下的简单 PD + 重力补偿（不使用 qdd_s）
    # 说明：这里 Kp/Kd 是“外环误差动力学”增益，先给一个稳一点的默认值。
    # 经验上：z 的单位是 m，角度是 rad；所以角度通道的 Kp/Kd 看起来会更大。
    # 这些参数偏“稳”，优先降低高频抖动。
    Kp = np.diag([2.0e4, 4.0e4, 4.0e4])
    Ki = np.diag([3.0e3, 1.5e3, 1.5e3])
    Kd = np.diag([2.2e4, 1.8e4, 1.8e4])
    controller = SimplePDGravityController3DOF(
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        use_gravity_compensation=True,
        use_coriolis_compensation=False,
        err_filter_T=0.05,
        integral_limit=np.array([0.5, 0.1, 0.1]),
        tau_limit=np.array([5e4, 5e3, 5e3]),
    )

    tau_hist = np.zeros((n, 3))
    tau_s_hist = np.zeros((n, 3))

    for k in range(n - 1):
        q = q_s[k] + q_c[k]
        qd = qd_s[k] + qd_c[k]

        M = platform.mass_matrix(q)
        C = platform.coriolis_matrix(q, qd)
        G = platform.gravity_vector(q)

        tau = controller.compute(
            t=t[k],
            platform=platform,
            q_c=q_c[k],
            qd_c=qd_c[k],
            q_s=q_s[k],
            qd_s=qd_s[k],
            q_ref=q_ref[k],
            qd_ref=qd_ref[k],
        )
        tau_s = -(M @ qdd_s[k] + C @ qd_s[k])

        # M qdd_c = tau + tau_s - C qd_c - G
        rhs = tau + tau_s - C @ qd_c[k] - G
        qdd_c = np.linalg.solve(M, rhs)
        qdd_c_hist[k] = qdd_c

        qd_c[k + 1] = qd_c[k] + qdd_c * dt
        q_c[k + 1] = q_c[k] + qd_c[k] * dt

        tau_hist[k] = tau
        tau_s_hist[k] = tau_s

    # 末端补齐
    qdd_c_hist[-1] = qdd_c_hist[-2]
    tau_hist[-1] = tau_hist[-2]
    tau_s_hist[-1] = tau_s_hist[-2]

    return (
        t,
        q_c,
        qd_c,
        qdd_c_hist,
        q_s,
        qd_s,
        qdd_s,
        tau_hist,
        tau_s_hist,
        q_ref,
        qd_ref,
    )


def main():
    (
        t,
        q_c,
        qd_c,
        qdd_c,
        q_s,
        qd_s,
        qdd_s,
        tau,
        tau_s,
        q_ref,
        qd_ref,
    ) = simulate(t_end=100.0, dt=0.01)

    figs_dir = _mkdir(Path(__file__).with_name("figures_uniform"))

    # 合成量
    q = q_s + q_c
    qd = qd_s + qd_c
    qdd = qdd_s + qdd_c
    tau_total = tau + tau_s

    e = q_ref - q
    ed = qd_ref - qd

    # 单位转换：角度(rad->deg)，位移(m->mm)
    def _to_plot_units(x: np.ndarray, kind: str) -> np.ndarray:
        x = np.asarray(x)
        y = x.copy()
        if kind in ("q", "qd", "qdd"):
            y[:, 0] *= 1000.0  # z: mm
        if kind in ("q", "qd", "qdd"):
            y[:, 1:] = np.rad2deg(y[:, 1:])  # angles: deg
        return y

    q_s_p = _to_plot_units(q_s, "q")
    q_c_p = _to_plot_units(q_c, "q")
    q_p = _to_plot_units(q, "q")

    qd_s_p = _to_plot_units(qd_s, "qd")
    qd_c_p = _to_plot_units(qd_c, "qd")
    qd_p = _to_plot_units(qd, "qd")

    qdd_s_p = _to_plot_units(qdd_s, "qdd")
    qdd_c_p = _to_plot_units(qdd_c, "qdd")
    qdd_p = _to_plot_units(qdd, "qdd")

    _plot_3dof(
        t,
        {"q_s": q_s_p, "q_c": q_c_p, "q": q_p},
        title="position",
        ylabels=("z (mm)", "alpha (deg)", "beta (deg)"),
        out_path=figs_dir / "uniform_position.png",
    )

    _plot_3dof(
        t,
        {"qd_s": qd_s_p, "qd_c": qd_c_p, "qd": qd_p},
        title="velocity",
        ylabels=("z_dot (mm/s)", "alpha_dot (deg/s)", "beta_dot (deg/s)"),
        out_path=figs_dir / "uniform_velocity.png",
    )

    _plot_3dof(
        t,
        {"qdd_s": qdd_s_p, "qdd_c": qdd_c_p, "qdd": qdd_p},
        title="acceleration",
        ylabels=("z_ddot (mm/s^2)", "alpha_ddot (deg/s^2)", "beta_ddot (deg/s^2)"),
        out_path=figs_dir / "uniform_acceleration.png",
    )

    _plot_3dof(
        t,
        {"tau": tau, "tau_s": tau_s, "tau_total": tau_total},
        title="generalized forces",
        ylabels=("tau_z", "tau_alpha", "tau_beta"),
        out_path=figs_dir / "uniform_forces.png",
    )

    # disturbance only
    _plot_3dof(
        t,
        {"tau_s": tau_s},
        title="disturbance (tau_s)",
        ylabels=("tau_s_z", "tau_s_alpha", "tau_s_beta"),
        out_path=figs_dir / "uniform_disturbance_tau_s.png",
    )

    # tracking error
    e_p = _to_plot_units(e, "q")
    ed_p = _to_plot_units(ed, "qd")
    _plot_3dof(
        t,
        {"e": e_p, "ed": ed_p},
        title="tracking error (q_ref - q)",
        ylabels=("e_z (mm)", "e_alpha (deg)", "e_beta (deg)"),
        out_path=figs_dir / "uniform_tracking_error.png",
    )

    print(f"saved figures under: {figs_dir}")


if __name__ == "__main__":
    main()
