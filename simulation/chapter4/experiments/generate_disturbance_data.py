"""Generate disturbance data (actual vs ESO estimate) for Chapter 4.

This script runs the Chapter 4 environment for a short horizon and saves an NPZ file containing:
- time: (T,)
- q, q_des, error: (T,3)
- eso_disturbance: (T,3)
- actual_disturbance: (T,3)
- region: (T,)

By default it uses random actions (no trained policy required).

Output:
- simulation/chapter4/figures/chapter4_disturbance_data.npz
"""

from __future__ import annotations

import os
import sys
import argparse
import numpy as np

_HERE = os.path.dirname(__file__)
_CH4_DIR = os.path.abspath(os.path.join(_HERE, ".."))
_CH4_ENV_DIR = os.path.join(_CH4_DIR, "env")
sys.path.insert(0, _CH4_ENV_DIR)

from rl_env_chapter4 import PlatformRLEnvChapter4


def main():
    parser = argparse.ArgumentParser(
        description="Generate Chapter 4 disturbance dataset (actual vs ESO estimate)"
    )
    parser.add_argument("--steps", type=int, default=1000, help="Number of env steps")
    parser.add_argument(
        "--mode",
        type=str,
        default="zero",
        choices=["zero", "constant", "seeded", "pid"],
        help=(
            "Action mode: zero=always [0,0,0]; "
            "constant=always use --action; "
            "seeded=env.action_space.sample() with fixed RNG seed; "
            "pid=use simple PID tracking to keep the platform near q_des"
        ),
    )
    parser.add_argument(
        "--action",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help="Constant action in [-1,1]^3 (only for --mode=constant)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for --mode=seeded")
    parser.add_argument(
        "--pid-kp",
        type=float,
        nargs=3,
        default=[0.10, 0.08, 0.08],
        help="PID Kp in action units (for --mode=pid)",
    )
    parser.add_argument(
        "--pid-ki",
        type=float,
        nargs=3,
        default=[0.00, 0.00, 0.00],
        help="PID Ki in action units (for --mode=pid)",
    )
    parser.add_argument(
        "--pid-kd",
        type=float,
        nargs=3,
        default=[0.03, 0.02, 0.02],
        help="PID Kd in action units (for --mode=pid)",
    )
    parser.add_argument(
        "--pid-action-limit",
        type=float,
        default=0.9,
        help="Clamp PID action to [-limit, +limit] (for --mode=pid)",
    )
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_dir = os.path.join(repo_root, "chapter4", "figures")
    os.makedirs(out_dir, exist_ok=True)

    env = PlatformRLEnvChapter4(
        use_model_compensation=True,
        use_eso=True,
        use_steso=True,
        use_hard_switch=True,
        max_episode_steps=args.steps,
        Hs=2.0,
        T1=8.0,
        q_des_type="sinusoidal",
        diverge_threshold=0.9,
        eso_omega=20.0,
        switch_threshold=0.6,
        switch_beta=0.5,
    )

    _ = env.reset()

    # Deterministic action generation
    if args.mode == "seeded":
        np.random.seed(args.seed)

    if args.mode == "zero":
        action = np.zeros(env.action_dim, dtype=np.float32)
    elif args.mode == "constant":
        action = np.array(args.action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
    else:
        action = None

    # PID state (in action units)
    kp = np.asarray(args.pid_kp, dtype=np.float64)
    ki = np.asarray(args.pid_ki, dtype=np.float64)
    kd = np.asarray(args.pid_kd, dtype=np.float64)
    pid_i = np.zeros(env.action_dim, dtype=np.float64)
    pid_prev_e = None

    for _ in range(args.steps):
        if args.mode == "seeded":
            action = env.action_space.sample()
        elif args.mode == "pid":
            # Use internal env state for tracking error (in physical units)
            e = env.q_des - env.q
            if pid_prev_e is None:
                pid_prev_e = e.copy()
            de = (e - pid_prev_e) / env.dt
            pid_prev_e = e.copy()

            pid_i = pid_i + e * env.dt

            # PID computes RL residual action v_RL (model compensation remains enabled in env)
            v = kp * e + ki * pid_i + kd * de

            # Convert to normalized action in [-1,1]
            action = (v / env.action_scale).astype(np.float32)
            action = np.clip(action, -args.pid_action_limit, args.pid_action_limit)

        _, _, done, _ = env.step(action)
        if done:
            break

    hist = env.get_history()

    time = np.asarray(hist.get("time", []), dtype=np.float64)
    q = np.asarray(hist.get("q", []), dtype=np.float64)
    qd = np.asarray(hist.get("qd", []), dtype=np.float64)
    q_des = np.asarray(hist.get("q_des", []), dtype=np.float64)
    qd_des = np.asarray(hist.get("qd_des", []), dtype=np.float64) if "qd_des" in hist else None
    error = q - q_des if (len(q) and len(q_des)) else np.zeros((0, 3))

    eso = np.asarray(hist.get("eso_disturbance", []), dtype=np.float64)
    actual = np.asarray(hist.get("actual_disturbance", []), dtype=np.float64)
    region_raw = hist.get("region", [])
    region = np.asarray(region_raw, dtype=object)
    steso = np.asarray(hist.get("steso_disturbance", []), dtype=np.float64)
    wave = np.asarray(hist.get("wave_disturbance", []), dtype=np.float64)
    residual = (actual - wave) if (len(actual) and len(wave)) else np.zeros((0, 3), dtype=np.float64)

    # Diagnostics (norms & estimation errors)
    e_norm = np.linalg.norm(error, axis=1) if len(error) else np.zeros((0,), dtype=np.float64)
    d_actual_norm = np.linalg.norm(actual, axis=1) if len(actual) else np.zeros((0,), dtype=np.float64)

    eso_err = (eso - actual) if (len(eso) and len(actual)) else np.zeros((0, 3), dtype=np.float64)
    steso_err = (steso - actual) if (len(steso) and len(actual)) else np.zeros((0, 3), dtype=np.float64)

    eso_err_norm = np.linalg.norm(eso_err, axis=1) if len(eso_err) else np.zeros((0,), dtype=np.float64)
    steso_err_norm = np.linalg.norm(steso_err, axis=1) if len(steso_err) else np.zeros((0,), dtype=np.float64)

    suffix = args.mode
    if args.mode == "constant":
        suffix += f"_{action[0]:+.2f}_{action[1]:+.2f}_{action[2]:+.2f}"
    if args.mode == "pid":
        suffix += "_kp" + "_".join(f"{x:.3g}" for x in kp)
        suffix += "_kd" + "_".join(f"{x:.3g}" for x in kd)
    out_path = os.path.join(out_dir, f"chapter4_disturbance_data_{suffix}.npz")

    np.savez(
        out_path,
        time=time,
        q=q,
        qd=qd,
        q_des=q_des,
        qd_des=qd_des if qd_des is not None else np.zeros((0, 3), dtype=np.float64),
        error=error,
        error_norm=e_norm,
        eso_disturbance=eso,
        actual_disturbance=actual,
        wave_disturbance=wave,
        residual_disturbance=residual,
        steso_disturbance=steso,
        eso_error=eso_err,
        steso_error=steso_err,
        d_actual_norm=d_actual_norm,
        eso_error_norm=eso_err_norm,
        steso_error_norm=steso_err_norm,
        region=region,
    )

    print(f"Saved: {out_path}")
    print(f"mode={args.mode}; steps={len(time)}; actual={actual.shape}; eso={eso.shape}")


if __name__ == "__main__":
    main()
