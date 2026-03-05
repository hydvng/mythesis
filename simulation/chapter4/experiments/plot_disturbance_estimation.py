"""Plot disturbance estimation results for Chapter 4.

This script plots, for each DOF (z/alpha/beta):
- actual total disturbance (computed from dynamics) vs.
- ESO estimated disturbance

Data sources:
1) Preferred: an NPZ file containing keys:
   - time: (T,)
   - actual_disturbance: (T, 3)
   - eso_disturbance: (T, 3)

2) Backward-compatible: existing `chapter4_data.npz` which contains:
   - time: (T,)
   - eso_dist: (T, 3)
   In this case, the script will only plot ESO disturbance.

Outputs are saved to: simulation/chapter4/figures/
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


DOF_LABELS = ["z", "alpha", "beta"]


@dataclass
class DisturbanceData:
    time: np.ndarray  # (T,)
    eso: np.ndarray  # (T, 3)
    steso: Optional[np.ndarray] = None  # (T, 3)
    actual: Optional[np.ndarray] = None  # (T, 3)
    wave: Optional[np.ndarray] = None  # (T, 3)
    residual: Optional[np.ndarray] = None  # (T, 3)


def _load_npz(path: str) -> DisturbanceData:
    d = np.load(path, allow_pickle=True)

    if "time" not in d.files:
        raise KeyError(f"Missing key 'time' in {path}. Keys: {d.files}")

    time = np.asarray(d["time"], dtype=np.float64)

    eso_key = None
    for k in ("eso_disturbance", "eso_dist", "eso_disturb", "eso"):
        if k in d.files:
            eso_key = k
            break
    if eso_key is None:
        raise KeyError(
            f"Missing ESO disturbance key in {path}. Expected one of: "
            f"eso_disturbance/eso_dist/eso_disturb/eso. Keys: {d.files}"
        )

    eso = np.asarray(d[eso_key], dtype=np.float64)

    steso_key = None
    for k in ("steso_disturbance", "steso_dist", "steso"):
        if k in d.files:
            steso_key = k
            break
    steso = None
    if steso_key is not None:
        steso = np.asarray(d[steso_key], dtype=np.float64)

    actual_key = None
    for k in ("actual_disturbance", "actual_dist", "d_actual"):
        if k in d.files:
            actual_key = k
            break

    actual = None
    if actual_key is not None:
        actual = np.asarray(d[actual_key], dtype=np.float64)

    wave = np.asarray(d["wave_disturbance"], dtype=np.float64) if "wave_disturbance" in d.files else None
    residual = (
        np.asarray(d["residual_disturbance"], dtype=np.float64)
        if "residual_disturbance" in d.files
        else ((actual - wave) if (actual is not None and wave is not None) else None)
    )

    return DisturbanceData(time=time, eso=eso, steso=steso, actual=actual, wave=wave, residual=residual)


def plot_disturbance(data: DisturbanceData, out_path: str, title: str = "") -> None:
    if data.eso.ndim != 2 or data.eso.shape[1] != 3:
        raise ValueError(f"ESO array must be (T,3), got {data.eso.shape}")
    if data.steso is not None and (data.steso.ndim != 2 or data.steso.shape[1] != 3):
        raise ValueError(f"STESO array must be (T,3), got {data.steso.shape}")
    if data.actual is not None and (data.actual.ndim != 2 or data.actual.shape[1] != 3):
        raise ValueError(f"Actual array must be (T,3), got {data.actual.shape}")
    if data.wave is not None and (data.wave.ndim != 2 or data.wave.shape[1] != 3):
        raise ValueError(f"Wave array must be (T,3), got {data.wave.shape}")
    if data.residual is not None and (data.residual.ndim != 2 or data.residual.shape[1] != 3):
        raise ValueError(f"Residual array must be (T,3), got {data.residual.shape}")

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    for i, ax in enumerate(axes):
        # 只绘制: 实际扰动, ESO
        if data.actual is not None:
            ax.plot(data.time, data.actual[:, i], label="Actual Disturbance", linewidth=1.5, color="black")
        ax.plot(data.time, data.eso[:, i], label="ESO Estimate", linewidth=1.5, color="blue")

        ax.set_ylabel(f"d_{DOF_LABELS[i]}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Time (s)")
    if title:
        fig.suptitle(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_disturbance_error(data: DisturbanceData, out_path: str, title: str = "") -> None:
    """Plot d_hat - d_actual per DOF for ESO/STESO (requires actual)."""
    if data.actual is None:
        raise ValueError("Actual disturbance is required to plot estimation error.")
    if data.eso.ndim != 2 or data.eso.shape[1] != 3:
        raise ValueError(f"ESO array must be (T,3), got {data.eso.shape}")
    if data.actual.ndim != 2 or data.actual.shape[1] != 3:
        raise ValueError(f"Actual array must be (T,3), got {data.actual.shape}")
    if data.steso is not None and (data.steso.ndim != 2 or data.steso.shape[1] != 3):
        raise ValueError(f"STESO array must be (T,3), got {data.steso.shape}")

    eso_err = data.eso - data.actual
    steso_err = (data.steso - data.actual) if data.steso is not None else None

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(data.time, eso_err[:, i], label="ESO error (d_hat - d)", linewidth=1.2)
        if steso_err is not None:
            ax.plot(data.time, steso_err[:, i], label="STESO error (d_hat - d)", linewidth=1.2)
        ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
        ax.set_ylabel(f"err_{DOF_LABELS[i]}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Time (s)")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_disturbance_error_norm(data: DisturbanceData, out_path: str, title: str = "") -> None:
    """Plot ||d_hat - d|| and ||d|| over time (requires actual)."""
    if data.actual is None:
        raise ValueError("Actual disturbance is required to plot estimation error norm.")

    d_norm = np.linalg.norm(data.actual, axis=1)
    eso_err_norm = np.linalg.norm(data.eso - data.actual, axis=1)
    steso_err_norm = (
        np.linalg.norm(data.steso - data.actual, axis=1) if data.steso is not None else None
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(data.time, d_norm, label="||d_actual||", linewidth=1.2)
    ax.plot(data.time, eso_err_norm, label="||ESO error||", linewidth=1.2)
    if steso_err_norm is not None:
        ax.plot(data.time, steso_err_norm, label="||STESO error||", linewidth=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Norm")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    default_npz = os.path.join(repo_root, "chapter4", "figures", "chapter4_data.npz")

    npz_path = os.environ.get("CH4_NPZ", default_npz)
    out_dir = os.path.join(repo_root, "chapter4", "figures")
    os.makedirs(out_dir, exist_ok=True)

    data = _load_npz(npz_path)

    out_path = os.path.join(out_dir, "chapter4_disturbance_tracking.png")
    title = "Chapter 4 Disturbance: estimates vs actual"
    if data.actual is None:
        title += " (actual not available in NPZ)"

    plot_disturbance(data, out_path=out_path, title=title)
    print(f"Saved: {out_path}")

    if data.actual is not None:
        out_err = os.path.join(out_dir, "chapter4_disturbance_error.png")
        out_err_norm = os.path.join(out_dir, "chapter4_disturbance_error_norm.png")
        plot_disturbance_error(
            data,
            out_path=out_err,
            title="Chapter 4 Disturbance estimation error (d_hat - d_actual)",
        )
        plot_disturbance_error_norm(
            data,
            out_path=out_err_norm,
            title="Chapter 4 Disturbance estimation error norms",
        )
        print(f"Saved: {out_err}")
        print(f"Saved: {out_err_norm}")


if __name__ == "__main__":
    main()
