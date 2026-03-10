"""sim_uniform_rod_with_wave_and_uav_impact_demo.py

在 `simulation/common/` 下提供一个“海浪基座扰动 + UAV 着舰冲击扰动”联合仿真 demo。

- 平台模型：UniformRodPlatform3DOF（3DOF: q=[z, alpha, beta]）
- 海浪扰动：WaveDisturbance 生成 ship 6DOF 运动，并取 [heave, roll, pitch] 映射为 q_s
  等效广义扰动：tau_wave = -M(q) qdd_s - C(q,qd) qd_s
- UAV 冲击扰动：generate_uav_landing_tau(t, params) 直接生成 tau_uav(t)

闭环仿真（控制形式，来自 common 的 wave demo 约定）：
    q = q_s + q_c
    qd = qd_s + qd_c
    M(q) qdd_c = tau_ctrl + tau_wave + tau_uav - C(q,qd) qd_c - G(q)

输出：
- 位置/速度/加速度
- 广义力：tau_ctrl / tau_wave / tau_uav / tau_total
- 局部放大图：UAV touchdown 附近窗口；海浪扰动固定 [5,8]s 窗口

默认输出目录：`simulation/common/figures_uniform/wave_uav_impact/`

注意：这是“common 侧”的可复现仿真脚本，便于论文附录/实验复现引用。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation.common.uniform_rod_platform_dynamics import UniformRodPlatform3DOF
from simulation.common.controllers import SimplePDGravityController3DOF
from simulation.disturbance.wave_disturbance import WaveDisturbance
from simulation.disturbance.uav_landing_disturbance import (
    UavLandingParams,
    generate_uav_landing_tau,
)


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


def _plot_3dof_window(
    t: np.ndarray,
    series: dict,
    *,
    title: str,
    ylabels: tuple[str, str, str],
    out_path: Path,
    t0: float,
    t1: float,
):
    mask = (t >= t0) & (t <= t1)
    _plot_3dof(
        t[mask],
        {k: v[mask] for k, v in series.items()},
        title=f"{title} (zoom {t0:.2f}-{t1:.2f}s)",
        ylabels=ylabels,
        out_path=out_path,
    )


def simulate(*, t_end: float = 20.0, dt: float = 0.01, uav_touchdown: float = 6.0):
    t = np.arange(0.0, t_end + 1e-12, dt)

    platform = UniformRodPlatform3DOF(diff_eps=1e-6)

    # wave
    wd = WaveDisturbance()
    ship = wd.generate_ship_motion(t)
    q_s = ship["q_s"][:, [2, 3, 4]]
    qd_s = ship["qd_s"][:, [2, 3, 4]]
    qdd_s = ship["qdd_s"][:, [2, 3, 4]]

    # uav
    uav_params = UavLandingParams(
        m_uav=500.0,
        t0=uav_touchdown,
        ramp=0.4,
        impulse_Iz=120.0,
        impulse_duration=0.08,
        r_x=0.25,
        r_y=-0.15,
    )
    tau_uav, meta = generate_uav_landing_tau(t, params=uav_params)

    n = t.size
    q_c = np.zeros((n, 3))
    qd_c = np.zeros((n, 3))
    qdd_c_hist = np.zeros((n, 3))

    # reference (total q track to 0)
    q_ref = np.zeros((n, 3))
    qd_ref = np.zeros((n, 3))

    # controller (same style as wave-only demo)
    # 说明：为提升UAV冲击后的回零能力，这里适当增强积分（消除稳态偏差）并提高阻尼。
    # 同时提高tau_limit并启用 back-calculation 抗积分饱和，避免冲击瞬间积分“卡死”。
    Kp = np.diag([2.2e4, 8.0e4, 8.0e4])
    Ki = np.diag([3.0e4, 1.0e4, 1.0e4])
    Kd = np.diag([3.0e4, 3.0e4, 3.0e4])
    controller = SimplePDGravityController3DOF(
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        # 纯PID：不使用除M之外的模型信息（不做G/C补偿）
        use_gravity_compensation=False,
        use_coriolis_compensation=False,
        err_filter_T=0.05,
    integral_limit=np.array([2.0, 0.25, 0.25]),
    tau_limit=np.array([2.0e5, 1.2e4, 1.2e4]),
        anti_windup_mode="backcalc",
        anti_windup_gain=20.0,
    )

    tau_ctrl_hist = np.zeros((n, 3))
    tau_wave_hist = np.zeros((n, 3))

    for k in range(n - 1):
        q = q_s[k] + q_c[k]
        qd = qd_s[k] + qd_c[k]

        M = platform.mass_matrix(q)
        C = platform.coriolis_matrix(q, qd)
        G = platform.gravity_vector(q)

        tau_ctrl = controller.compute(
            t=t[k],
            platform=platform,
            q_c=q_c[k],
            qd_c=qd_c[k],
            q_s=q_s[k],
            qd_s=qd_s[k],
            q_ref=q_ref[k],
            qd_ref=qd_ref[k],
        )
        tau_wave = -(M @ qdd_s[k] + C @ qd_s[k])
        tau_u = tau_uav[k]

        rhs = tau_ctrl + tau_wave + tau_u - C @ qd_c[k] - G
        qdd_c = np.linalg.solve(M, rhs)

        qdd_c_hist[k] = qdd_c
        tau_ctrl_hist[k] = tau_ctrl
        tau_wave_hist[k] = tau_wave

        qd_c[k + 1] = qd_c[k] + qdd_c * dt
        q_c[k + 1] = q_c[k] + qd_c[k] * dt

    qdd_c_hist[-1] = qdd_c_hist[-2]
    tau_ctrl_hist[-1] = tau_ctrl_hist[-2]
    tau_wave_hist[-1] = tau_wave_hist[-2]
    tau_uav[-1] = tau_uav[-2]

    return {
        "t": t,
        "q_s": q_s,
        "qd_s": qd_s,
        "qdd_s": qdd_s,
        "q_c": q_c,
        "qd_c": qd_c,
        "qdd_c": qdd_c_hist,
        "tau_ctrl": tau_ctrl_hist,
        "tau_wave": tau_wave_hist,
        "tau_uav": tau_uav,
        "uav_meta": meta,
        "uav_params": uav_params,
    }


def main():
    out_dir = _mkdir(Path(__file__).with_name("figures_uniform") / "wave_uav_impact")

    sim = simulate(t_end=20.0, dt=0.01, uav_touchdown=6.0)
    t = sim["t"]

    q_s = sim["q_s"]
    qd_s = sim["qd_s"]
    qdd_s = sim["qdd_s"]
    q_c = sim["q_c"]
    qd_c = sim["qd_c"]
    qdd_c = sim["qdd_c"]

    q = q_s + q_c
    qd = qd_s + qd_c
    qdd = qdd_s + qdd_c

    tau_ctrl = sim["tau_ctrl"]
    tau_wave = sim["tau_wave"]
    tau_uav = sim["tau_uav"]
    tau_total = tau_ctrl + tau_wave + tau_uav

    # unit conversion for states
    def _to_plot_units(x: np.ndarray, kind: str) -> np.ndarray:
        y = np.asarray(x).copy()
        if kind in ("q", "qd", "qdd"):
            y[:, 0] *= 1000.0
            y[:, 1:] = np.rad2deg(y[:, 1:])
        return y

    _plot_3dof(
        t,
        {"q_s": _to_plot_units(q_s, "q"), "q_c": _to_plot_units(q_c, "q"), "q": _to_plot_units(q, "q")},
        title="position",
        ylabels=("z (mm)", "alpha (deg)", "beta (deg)"),
        out_path=out_dir / "position.png",
    )

    _plot_3dof(
        t,
        {"qd_s": _to_plot_units(qd_s, "qd"), "qd_c": _to_plot_units(qd_c, "qd"), "qd": _to_plot_units(qd, "qd")},
        title="velocity",
        ylabels=("z_dot (mm/s)", "alpha_dot (deg/s)", "beta_dot (deg/s)"),
        out_path=out_dir / "velocity.png",
    )

    _plot_3dof(
        t,
        {
            "qdd_s": _to_plot_units(qdd_s, "qdd"),
            "qdd_c": _to_plot_units(qdd_c, "qdd"),
            "qdd": _to_plot_units(qdd, "qdd"),
        },
        title="acceleration",
        ylabels=("z_ddot (mm/s^2)", "alpha_ddot (deg/s^2)", "beta_ddot (deg/s^2)"),
        out_path=out_dir / "acceleration.png",
    )

    _plot_3dof(
        t,
        {"tau_ctrl": tau_ctrl, "tau_wave": tau_wave, "tau_uav": tau_uav, "tau_total": tau_total},
        title="generalized forces",
        ylabels=("tau_z", "tau_alpha", "tau_beta"),
        out_path=out_dir / "forces.png",
    )

    # zoom windows
    td = sim["uav_params"].t0
    _plot_3dof_window(
        t,
        {"tau_uav": tau_uav, "tau_total": tau_total},
        title="uav impact forces",
        ylabels=("tau_z", "tau_alpha", "tau_beta"),
        out_path=out_dir / "forces_uav_zoom.png",
        t0=td - 0.5,
        t1=td + 1.0,
    )

    _plot_3dof_window(
        t,
        {"tau_wave": tau_wave},
        title="wave disturbance",
        ylabels=("tau_z", "tau_alpha", "tau_beta"),
        out_path=out_dir / "tau_wave_zoom_5_8s.png",
        t0=5.0,
        t1=8.0,
    )

    print(f"saved figures under: {out_dir}")


if __name__ == "__main__":
    main()
