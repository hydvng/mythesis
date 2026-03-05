"""Plot tracking results (q vs q_des and error) for Chapter 4.

Inputs (NPZ):
- time: (T,)
- q: (T,3)
- q_des: (T,3)
Optional:
- qd: (T,3)
- qd_des: (T,3)
- region: (T,)

Usage:
- Set env var CH4_NPZ to choose file, else uses chapter4/figures/chapter4_disturbance_data_pid*.npz

Outputs:
- simulation/chapter4/figures/chapter4_tracking_q.png
- simulation/chapter4/figures/chapter4_tracking_error.png
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


DOF_LABELS = ["z", "alpha", "beta"]


@dataclass
class TrackingData:
    time: np.ndarray
    q: np.ndarray
    q_des: np.ndarray
    qd: Optional[np.ndarray] = None
    qd_des: Optional[np.ndarray] = None
    region: Optional[np.ndarray] = None


def _load_npz(path: str) -> TrackingData:
    d = np.load(path, allow_pickle=True)
    files = set(d.files)

    for k in ("time", "q", "q_des"):
        if k not in files:
            raise KeyError(f"Missing key '{k}' in {path}. Keys: {sorted(files)}")

    time = np.asarray(d["time"], dtype=np.float64)
    q = np.asarray(d["q"], dtype=np.float64)
    q_des = np.asarray(d["q_des"], dtype=np.float64)

    qd = np.asarray(d["qd"], dtype=np.float64) if "qd" in files else None
    qd_des = np.asarray(d["qd_des"], dtype=np.float64) if "qd_des" in files else None

    region = None
    if "region" in files:
        region = np.asarray(d["region"], dtype=object)

    return TrackingData(time=time, q=q, q_des=q_des, qd=qd, qd_des=qd_des, region=region)


def plot_q_tracking(data: TrackingData, out_path: str, title: str = "") -> None:
    if data.q.ndim != 2 or data.q.shape[1] != 3:
        raise ValueError(f"q must be (T,3), got {data.q.shape}")
    if data.q_des.ndim != 2 or data.q_des.shape[1] != 3:
        raise ValueError(f"q_des must be (T,3), got {data.q_des.shape}")

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    for i, ax in enumerate(axes):
        ax.plot(data.time, data.q[:, i], label="q", linewidth=1.5)
        ax.plot(data.time, data.q_des[:, i], label="q_des", linewidth=1.2)
        ax.set_ylabel(DOF_LABELS[i])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Time (s)")
    if title:
        fig.suptitle(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_error(data: TrackingData, out_path: str, title: str = "") -> None:
    e = data.q - data.q_des

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    for i, ax in enumerate(axes):
        ax.plot(data.time, e[:, i], label="e = q - q_des", linewidth=1.5)
        ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
        ax.set_ylabel(f"e_{DOF_LABELS[i]}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Time (s)")
    if title:
        fig.suptitle(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _pick_default_npz(fig_dir: str) -> str:
    # Prefer PID-generated datasets
    candidates = sorted(glob.glob(os.path.join(fig_dir, "chapter4_disturbance_data_pid_*.npz")))
    if candidates:
        return candidates[-1]
    candidates = sorted(glob.glob(os.path.join(fig_dir, "chapter4_disturbance_data_*.npz")))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"No chapter4_disturbance_data_*.npz found in {fig_dir}")


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    fig_dir = os.path.join(repo_root, "chapter4", "figures")

    npz_path = os.environ.get("CH4_NPZ")
    if not npz_path:
        npz_path = _pick_default_npz(fig_dir)

    os.makedirs(fig_dir, exist_ok=True)

    data = _load_npz(npz_path)

    out_q = os.path.join(fig_dir, "chapter4_tracking_q.png")
    out_e = os.path.join(fig_dir, "chapter4_tracking_error.png")

    base = os.path.basename(npz_path)
    plot_q_tracking(data, out_path=out_q, title=f"Chapter 4 tracking (q vs q_des)\n{base}")
    plot_error(data, out_path=out_e, title=f"Chapter 4 tracking error\n{base}")

    print(f"NPZ: {npz_path}")
    print(f"Saved: {out_q}")
    print(f"Saved: {out_e}")


if __name__ == "__main__":
    main()
