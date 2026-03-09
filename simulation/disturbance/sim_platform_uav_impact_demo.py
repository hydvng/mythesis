"""sim_platform_uav_impact_demo.py

把 UAV 降落（静载 + 冲量）当作外部扰动注入 3DOF 平台模型，观察冲击后的
位置/速度/加速度/广义力等状态曲线。

说明
- 复用 uniform-rod 3DOF 平台动力学与现有控制器。
- 这里默认不叠加波浪扰动（以便单独看 UAV 冲击效果）。如果你想要同时叠加波浪，
  可以把 wave_disturbance 的 q_s/qd_s/qdd_s 加进来。

输出目录
- `simulation/disturbance/figures/uav_impact/`

"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from simulation.common.uniform_rod_platform_dynamics import UniformRodPlatform3DOF
from simulation.common.controllers import SimplePDGravityController3DOF
from simulation.disturbance.uav_landing_disturbance import UavLandingParams, generate_uav_landing_tau
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


def _slice_window(t: np.ndarray, y: np.ndarray, *, t0: float, t_pre: float, t_post: float):
    """截取 [t0-t_pre, t0+t_post] 时间窗。"""
    mask = (t >= (t0 - t_pre)) & (t <= (t0 + t_post))
    return t[mask], y[mask]


def simulate(
    *,
    t_end: float = 30.0,
    dt: float = 0.001,
    impulse_check_dt: float = 1e-4,
    use_wave: bool = False,
):
    t = np.arange(0.0, t_end + 1e-12, dt)

    if use_wave:
        wd = WaveDisturbance()
        ship = wd.generate_ship_motion(t)
        # ship motion -> 3DOF: [z, alpha, beta] 对应 [heave, roll, pitch]
        q_s = ship["q_s"][:, [2, 3, 4]]
        qd_s = ship["qd_s"][:, [2, 3, 4]]
        qdd_s = ship["qdd_s"][:, [2, 3, 4]]
    else:
        # 不叠加波浪时：ship motion = 0
        q_s = np.zeros((t.size, 3))
        qd_s = np.zeros((t.size, 3))
        qdd_s = np.zeros((t.size, 3))

    platform = UniformRodPlatform3DOF(diff_eps=1e-6)

    # 目标：总位姿 q = q_s + q_c 跟踪 0
    q_ref = np.zeros((t.size, 3))
    qd_ref = np.zeros((t.size, 3))

    # 控制器（偏稳健参数；后续可按需要调）
    Kp = np.diag([2.0e4, 4.0e4, 4.0e4])
    Ki = np.diag([0.0, 0.0, 0.0])
    Kd = np.diag([2.2e4, 1.8e4, 1.8e4])
    controller = SimplePDGravityController3DOF(
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        use_gravity_compensation=True,
        use_coriolis_compensation=False,
        err_filter_T=0.02,
        integral_limit=np.array([0.5, 0.1, 0.1]),
        tau_limit=np.array([8e4, 8e3, 8e3]),
    )

    # UAV 扰动
    uav_params = UavLandingParams(
        m_uav=20.0,
        t0=10.0,
        ramp=0.2,
        duration=None,
        impulse_Iz=120.0,
        impulse_duration=0.05,
        radius_limit=0.3,
        random_seed=3,
    )
    tau_uav, meta = generate_uav_landing_tau(t, params=uav_params)

    # 更精确的冲量校验（仅用于校验，不影响主仿真步长）
    if abs(meta["impulse_Iz"]) > 0 and impulse_check_dt is not None and impulse_check_dt < dt:
        t_chk = np.arange(meta["t0"], meta["t0"] + meta["impulse_duration"], impulse_check_dt)
        tau_chk, meta_chk = generate_uav_landing_tau(t_chk, params=uav_params)
        # 静载 profile s(t)
        s_chk = np.clip((t_chk - meta_chk["t0"]) / meta_chk["ramp"], 0.0, 1.0)
        Iz_est = np.trapz(tau_chk[:, 0] - meta_chk["Fz"] * s_chk, t_chk)
        print(
            f"impulse Iz check (dense): target={meta_chk['impulse_Iz']:.3f} N·s, est={Iz_est:.3f} N·s"
        )

    n = t.size
    q_c = np.zeros((n, 3))
    qd_c = np.zeros((n, 3))
    qdd_c_hist = np.zeros((n, 3))

    tau_ctrl_hist = np.zeros((n, 3))
    tau_total_hist = np.zeros((n, 3))
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

        # 总外力：UAV 扰动（正方向按约定 z 轴向下/重力方向）
        tau_ext = tau_uav[k]

        # 波浪等效扰动（仅当 use_wave=True 时 q_s/qd_s/qdd_s 非零）
        # 形式与之前的 sim_uniform_rod_with_wave_demo 保持一致：
        #   tau_wave = -M(q) qdd_s - C(q, qd) qd_s
        tau_wave = -(M @ qdd_s[k] + C @ qd_s[k])
        tau_wave_hist[k] = tau_wave

        # M qdd_c = tau_ctrl + tau_ext + tau_wave - C qd_c - G
        rhs = tau_ctrl + tau_ext + tau_wave - C @ qd_c[k] - G
        qdd_c = np.linalg.solve(M, rhs)
        qdd_c_hist[k] = qdd_c

        qd_c[k + 1] = qd_c[k] + qdd_c * dt
        q_c[k + 1] = q_c[k] + qd_c[k] * dt

        tau_ctrl_hist[k] = tau_ctrl
        tau_total_hist[k] = tau_ctrl + tau_ext + tau_wave

    qdd_c_hist[-1] = qdd_c_hist[-2]
    tau_ctrl_hist[-1] = tau_ctrl_hist[-2]
    tau_total_hist[-1] = tau_total_hist[-2]
    tau_wave_hist[-1] = tau_wave_hist[-2]

    q = q_s + q_c
    qd = qd_s + qd_c
    qdd = qdd_s + qdd_c_hist

    return t, q, qd, qdd, tau_ctrl_hist, tau_uav, tau_wave_hist, tau_total_hist, meta, use_wave


def main():
    # 生成两组：仅 UAV / UAV + wave（方便对比）
    for use_wave in (False, True):
        t, q, qd, qdd, tau_ctrl, tau_uav, tau_wave, tau_total, meta, _ = simulate(
            t_end=30.0, dt=0.001, use_wave=use_wave
        )

        tag = "uav_plus_wave" if use_wave else "uav_only"
        out_dir = _mkdir(_THIS.parent / "figures" / "uav_impact" / tag)

        # 单位转换：z(m->mm), 角(rad->deg)
        def _to_plot_units(x: np.ndarray, kind: str) -> np.ndarray:
            y = np.asarray(x, dtype=float).copy()
            if kind in ("q", "qd", "qdd"):
                y[:, 0] *= 1000.0
                y[:, 1:] = np.rad2deg(y[:, 1:])
            return y

        q_p = _to_plot_units(q, "q")
        qd_p = _to_plot_units(qd, "qd")
        qdd_p = _to_plot_units(qdd, "qdd")

        _plot_3dof(
            t,
            {"q": q_p},
            title=f"position (platform total) [{tag}]",
            ylabels=("z (mm)", "alpha (deg)", "beta (deg)"),
            out_path=out_dir / "position.png",
        )
        _plot_3dof(
            t,
            {"qd": qd_p},
            title=f"velocity (platform total) [{tag}]",
            ylabels=("z_dot (mm/s)", "alpha_dot (deg/s)", "beta_dot (deg/s)"),
            out_path=out_dir / "velocity.png",
        )
        _plot_3dof(
            t,
            {"qdd": qdd_p},
            title=f"acceleration (platform total) [{tag}]",
            ylabels=("z_ddot (mm/s^2)", "alpha_ddot (deg/s^2)", "beta_ddot (deg/s^2)"),
            out_path=out_dir / "acceleration.png",
        )
        _plot_3dof(
            t,
            {"tau_ctrl": tau_ctrl, "tau_uav": tau_uav, "tau_wave": tau_wave, "tau_total": tau_total},
            title=f"generalized forces [{tag}]",
            ylabels=("tau_z (N)", "tau_alpha (N·m)", "tau_beta (N·m)"),
            out_path=out_dir / "forces.png",
        )

        _plot_3dof(
            t,
            {"tau_uav": tau_uav},
            title=f"uav disturbance (static + impulse) [{tag}]",
            ylabels=("tau_z (N)", "tau_alpha (N·m)", "tau_beta (N·m)"),
            out_path=out_dir / "disturbance.png",
        )

        if use_wave:
            _plot_3dof(
                t,
                {"tau_wave": tau_wave},
                title=f"wave disturbance (equivalent generalized) [{tag}]",
                ylabels=("tau_z (N)", "tau_alpha (N·m)", "tau_beta (N·m)"),
                out_path=out_dir / "wave_disturbance.png",
            )

        # 冲击窗口放大图（默认：t0 前 1s，后 3s）
        t0 = float(meta["t0"])
        t_pre, t_post = 1.0, 3.0

        tw, qw = _slice_window(t, q_p, t0=t0, t_pre=t_pre, t_post=t_post)
        _, qdw = _slice_window(t, qd_p, t0=t0, t_pre=t_pre, t_post=t_post)
        _, qddw = _slice_window(t, qdd_p, t0=t0, t_pre=t_pre, t_post=t_post)
        _, tau_uav_w = _slice_window(t, tau_uav, t0=t0, t_pre=t_pre, t_post=t_post)
        _, tau_wave_w = _slice_window(t, tau_wave, t0=t0, t_pre=t_pre, t_post=t_post)
        _, tau_ctrl_w = _slice_window(t, tau_ctrl, t0=t0, t_pre=t_pre, t_post=t_post)
        _, tau_total_w = _slice_window(t, tau_total, t0=t0, t_pre=t_pre, t_post=t_post)

        _plot_3dof(
            tw,
            {"q": qw},
            title=f"position zoom (t0±[{t_pre},{t_post}]s) [{tag}]",
            ylabels=("z (mm)", "alpha (deg)", "beta (deg)"),
            out_path=out_dir / "position_zoom.png",
        )
        _plot_3dof(
            tw,
            {"qd": qdw},
            title=f"velocity zoom (t0±[{t_pre},{t_post}]s) [{tag}]",
            ylabels=("z_dot (mm/s)", "alpha_dot (deg/s)", "beta_dot (deg/s)"),
            out_path=out_dir / "velocity_zoom.png",
        )
        _plot_3dof(
            tw,
            {"qdd": qddw},
            title=f"acceleration zoom (t0±[{t_pre},{t_post}]s) [{tag}]",
            ylabels=("z_ddot (mm/s^2)", "alpha_ddot (deg/s^2)", "beta_ddot (deg/s^2)"),
            out_path=out_dir / "acceleration_zoom.png",
        )
        _plot_3dof(
            tw,
            {"tau_uav": tau_uav_w},
            title=f"uav disturbance zoom (t0±[{t_pre},{t_post}]s) [{tag}]",
            ylabels=("tau_z (N)", "tau_alpha (N·m)", "tau_beta (N·m)"),
            out_path=out_dir / "disturbance_zoom.png",
        )
        _plot_3dof(
            tw,
            {"tau_ctrl": tau_ctrl_w, "tau_uav": tau_uav_w, "tau_total": tau_total_w},
            title=f"forces zoom (t0±[{t_pre},{t_post}]s) [{tag}]",
            ylabels=("tau_z (N)", "tau_alpha (N·m)", "tau_beta (N·m)"),
            out_path=out_dir / "forces_zoom.png",
        )

        # 按你的要求：海浪扰动局部放大图（固定时间窗，便于看清小扰动）
        if use_wave:
            t1, t2 = 5.0, 8.0
            mask = (t >= t1) & (t <= t2)
            _plot_3dof(
                t[mask],
                {"tau_wave": tau_wave[mask]},
                title=f"wave disturbance zoom (t in [{t1:.1f},{t2:.1f}]s) [{tag}]",
                ylabels=("tau_z (N)", "tau_alpha (N·m)", "tau_beta (N·m)"),
                out_path=out_dir / "wave_disturbance_zoom_5_8s.png",
            )

        print(f"saved to: {out_dir}")
        print(
            f"[{tag}] uav meta: "
            f"m={meta['m_uav']}kg, r=({meta['r_x']:.3f},{meta['r_y']:.3f})m, "
            f"t0={meta['t0']}s, impulse_Iz={meta['impulse_Iz']} N·s"
        )


if __name__ == "__main__":
    main()
