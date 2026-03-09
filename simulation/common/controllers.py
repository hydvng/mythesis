"""controllers.py

简单但实用的 3DOF 平台跟踪控制器。

控制目标
- 令上平台相对惯性系的总位姿 q = q_s + q_c 跟踪期望 q_ref。

动力学（控制形式）
    M(q) qdd_c + C(q, qd) qd_c + G(q) = tau + tau_s
其中
    q = q_s + q_c
    qd = qd_s + qd_c
    tau_s = -M(q) qdd_s - C(q, qd) qd_s

计算力矩/力（广义力）
- 采用“计算力矩法”风格的跟踪控制（feedback linearization 的简化版）：

    tau = M(q) * v + C(q,qd) * qd + G(q) + tau_s_ff

  其中 v 设计为总位姿误差的二阶稳定系统：

    e = q_ref - q
    ed = qd_ref - qd
    v = qdd_ref + Kd * ed + Kp * e

- 前馈项 tau_s_ff（可选）：如果能获得 qdd_s，则用完整扰动补偿（抵消右端的 τ_s）：

    tau_s_ff = +M(q) qdd_s + C(q,qd) qd_s

  若拿不到 qdd_s（仅有 q_s, qd_s），则也可以只补偿 -C*qd_s 或直接关闭。

注意
- 这里的 q_ref, qd_ref, qdd_ref 都是“总位姿”的参考（即惯性系下的上平台期望）。
- 若你希望平台相对船体保持某姿态（即 q_c 跟踪某参考），只需把 q_ref 改成 q_s + q_c_ref。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SimplePDGravityController3DOF:
    """未知扰动下的简单控制器（不使用 qdd_s）。

    思路：把扰动当作外部未知输入，用 PD + 重力补偿（可选加上 C(q,qd)qd）实现稳定与一定的跟踪能力。

    适用：你不想/不能用 qdd_s 做 τ_s 前馈时。

    控制目标：默认对“总位姿” q=q_s+q_c 做跟踪：e=q_ref-q。
    如果你想控制 q_c，只需要把 q_ref 改成 q_s + q_c_ref。
    """

    Kp: np.ndarray  # (3,3)
    Kd: np.ndarray  # (3,3)
    use_coriolis_compensation: bool = False
    tau_limit: Optional[np.ndarray] = None  # (3,) or scalar

    def compute(
        self,
        *,
        t: float,
        platform,
        q_c: np.ndarray,
        qd_c: np.ndarray,
        q_s: np.ndarray,
        qd_s: np.ndarray,
        q_ref: np.ndarray,
        qd_ref: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        q_c = np.asarray(q_c, dtype=float).reshape(3)
        qd_c = np.asarray(qd_c, dtype=float).reshape(3)
        q_s = np.asarray(q_s, dtype=float).reshape(3)
        qd_s = np.asarray(qd_s, dtype=float).reshape(3)
        q_ref = np.asarray(q_ref, dtype=float).reshape(3)

        if qd_ref is None:
            qd_ref = np.zeros(3)
        else:
            qd_ref = np.asarray(qd_ref, dtype=float).reshape(3)

        q = q_s + q_c
        qd = qd_s + qd_c

        C = platform.coriolis_matrix(q, qd)
        G = platform.gravity_vector(q)

        e = q_ref - q
        ed = qd_ref - qd

        tau = self.Kp @ e + self.Kd @ ed 
        if self.use_coriolis_compensation:
            tau = tau + C @ qd

        if self.tau_limit is not None:
            lim = np.asarray(self.tau_limit, dtype=float)
            if lim.size == 1:
                lim = np.full(3, float(lim))
            lim = lim.reshape(3)
            tau = np.clip(tau, -lim, lim)

        return tau


@dataclass
class ComputedTorqueTrackingController3DOF:
    """计算力矩法（computed torque）跟踪控制器。"""

    Kp: np.ndarray  # shape (3,3)
    Kd: np.ndarray  # shape (3,3)
    enable_disturbance_feedforward: bool = True
    tau_limit: Optional[np.ndarray] = None  # shape (3,) or scalar

    def compute(
        self,
        *,
        t: float,
        platform,
        q_c: np.ndarray,
        qd_c: np.ndarray,
        q_s: np.ndarray,
        qd_s: np.ndarray,
        qdd_s: Optional[np.ndarray],
        q_ref: np.ndarray,
        qd_ref: Optional[np.ndarray] = None,
        qdd_ref: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        q_c = np.asarray(q_c, dtype=float).reshape(3)
        qd_c = np.asarray(qd_c, dtype=float).reshape(3)
        q_s = np.asarray(q_s, dtype=float).reshape(3)
        qd_s = np.asarray(qd_s, dtype=float).reshape(3)
        q_ref = np.asarray(q_ref, dtype=float).reshape(3)

        if qd_ref is None:
            qd_ref = np.zeros(3)
        else:
            qd_ref = np.asarray(qd_ref, dtype=float).reshape(3)

        if qdd_ref is None:
            qdd_ref = np.zeros(3)
        else:
            qdd_ref = np.asarray(qdd_ref, dtype=float).reshape(3)

        # Total state
        q = q_s + q_c
        qd = qd_s + qd_c

        # Model terms
        M = platform.mass_matrix(q)
        C = platform.coriolis_matrix(q, qd)
        G = platform.gravity_vector(q)

        # Tracking errors on total pose
        e = q_ref - q
        ed = qd_ref - qd

        v = qdd_ref + self.Kd @ ed + self.Kp @ e

        tau_s_ff = np.zeros(3)
        if self.enable_disturbance_feedforward and (qdd_s is not None):
            qdd_s = np.asarray(qdd_s, dtype=float).reshape(3)
            # cancel tau_s on the right-hand side
            tau_s_ff = M @ qdd_s + C @ qd_s

        # Control law: cancel nominal dynamics and enforce 2nd-order error dynamics
        tau = M @ v + C @ qd + G + tau_s_ff

        # Optional saturation
        if self.tau_limit is not None:
            lim = np.asarray(self.tau_limit, dtype=float)
            if lim.size == 1:
                lim = np.full(3, float(lim))
            lim = lim.reshape(3)
            tau = np.clip(tau, -lim, lim)

        return tau
