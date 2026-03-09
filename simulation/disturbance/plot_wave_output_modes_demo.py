"""Quick visualization for WaveDisturbance output modes.

This script generates a single PNG under:
  simulation/disturbance/figures/wave_output_modes_demo.png

It plots:
  - ship motion: q_s, qd_s, qdd_s (z, roll, pitch)
  - equivalent generalized disturbance: tau_dist

Run from anywhere:
  python3 /abs/path/to/plot_wave_output_modes_demo.py
"""

import os
import sys

import numpy as np

# Allow running from any CWD by injecting the project root
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from simulation.disturbance.wave_disturbance import WaveDisturbance
from simulation.common.platform_dynamics import ParallelPlatform3DOF


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_end = 100.0
    dt = 0.01
    t = np.arange(0.0, t_end + 1e-12, dt)

    wave = WaveDisturbance(
        Hs=2.0,
        T1=8.0,
        vessel_file="supply.mat",
        wave_heading=45.0,
        n_components=30,
        random_seed=0,
        enable_burst_step=False,
    )

    ship = wave.generate(t, output="ship_state", angle_unit="deg")

    platform = ParallelPlatform3DOF(
        m_platform=float(wave.platform.get("m", 347.54)),
        Ixx=float(wave.platform.get("Ixx", 60.64)),
        Iyy=float(wave.platform.get("Iyy", 115.4)),
        Izz=float(wave.platform.get("Izz", 80.0)),
    )

    q_u = np.zeros((len(t), 3))
    qd_u = np.zeros((len(t), 3))
    q_u[:, 0] = 1.058  # nominal heave offset

    out = wave.generate(t, output="tau_dist", q_u=q_u, qd_u=qd_u, platform=platform)

    q_s, qd_s, qdd_s = ship["q_s"], ship["qd_s"], ship["qdd_s"]
    tau = out["tau_dist"]

    figures_dir = os.path.join(_THIS_DIR, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Figure A: legacy 3DOF channels (heave/roll/pitch) + 3DOF tau_dist
    # Ship motion ordering: [surge, sway, heave, roll, pitch, yaw]
    idx3 = [2, 3, 4]
    labels3 = ["heave z (m)", "roll alpha (deg)", "pitch beta (deg)"]
    tau_labels = ["Fz (N)", "M_alpha (N·m)", "M_beta (N·m)"]

    fig, axes = plt.subplots(4, 3, figsize=(14, 10), sharex=True)
    for i, k in enumerate(idx3):
        axes[0, i].plot(t, q_s[:, k], lw=1.2)
        axes[0, i].set_title(f"q_s: {labels3[i]}")
        axes[0, i].grid(True, alpha=0.3)

        axes[1, i].plot(t, qd_s[:, k], lw=1.2, color="tab:orange")
        axes[1, i].set_title(f"qd_s: d/dt {labels3[i]}")
        axes[1, i].grid(True, alpha=0.3)

        axes[2, i].plot(t, qdd_s[:, k], lw=1.2, color="tab:green")
        axes[2, i].set_title(f"qdd_s: d²/dt² {labels3[i]}")
        axes[2, i].grid(True, alpha=0.3)

        axes[3, i].plot(t, tau[:, i], lw=1.4, color="tab:red")
        axes[3, i].set_title(f"tau_dist (3DOF): {tau_labels[i]}")
        axes[3, i].grid(True, alpha=0.3)
        axes[3, i].set_xlabel("Time (s)")

    fig.suptitle(
        "WaveDisturbance output demo (6DOF ship_state, 3DOF tau_dist from ParallelPlatform3DOF)\n"
        "tau_dist = -M(q_u) qdd_s - C(q_u,qd_u) qd_s",
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path_a = os.path.join(figures_dir, "wave_output_modes_demo.png")
    fig.savefig(out_path_a, dpi=180)
    plt.close(fig)

    # Figure B: full 6DOF ship motion only (no tau)
    dof_names = ["surge", "sway", "heave", "roll", "pitch", "yaw"]
    fig2, axes2 = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    axes2 = axes2.reshape(3, 2)
    for i, name in enumerate(dof_names):
        r = i // 2
        c = i % 2
        axes2[r, c].plot(t, q_s[:, i], lw=1.2)
        axes2[r, c].set_title(f"q_s: {name}")
        axes2[r, c].grid(True, alpha=0.3)
        if r == 2:
            axes2[r, c].set_xlabel("Time (s)")
    fig2.suptitle("WaveDisturbance ship_state (6DOF) — q_s", fontweight="bold")
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    out_path_b = os.path.join(figures_dir, "wave_ship_state_6dof.png")
    fig2.savefig(out_path_b, dpi=180)
    plt.close(fig2)

    print(out_path_a)
    print(out_path_b)


if __name__ == "__main__":
    main()
