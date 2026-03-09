r"""uniform_rod_platform_dynamics.py

实现 `dynamics_3ups_uniform.md` 中的“均匀杆件”3-UPS/PU 平台动力学（3DOF: [z, alpha, beta]）。

目标：提供与现有 `ParallelPlatform3DOF` 一致的接口：
- `mass_matrix(q)`
- `coriolis_matrix(q, qd)`  (先复用 plate 的解析形式)
- `gravity_vector(q)`

并补充：
- 均匀杆件导致的额外质量矩阵项：M_rod_trans、M_rod_rot
- 均匀杆件导致的额外重力项：G_rod

说明与建模约定
- 广义坐标 q = [z, alpha, beta]^T 表示“上平台相对惯性系”的总位姿（论文中 q=q_s+q_c）。
- 在构造偏导（例如对 q 求导）的过程中，视 q_s 为常量；这里我们直接对 q 做偏导。

数值实现选择
- 文档中的 M_rod_trans 需要 J_{P_i}(q) = \partial P_i / \partial q。
  由于现有 `ParallelPlatform3DOF` 没有显式提供 \partial R/\partial alpha,beta，
  这里采用中心差分对 P_i(q) 做数值雅可比（3x3）。
- 文档中的 M_rod_rot 需要 E_i = \partial e_i/\partial q，其中
  \partial e_i/\partial q_k = (1/L_i) * (I - e e^T) * \partial P_i/\partial q_k。

这版实现优先保证：接口一致、可仿真、数值稳定；后续如需加速，可以把数值雅可比替换为解析偏导。
"""

from __future__ import annotations

import numpy as np

from .platform_dynamics import ParallelPlatform3DOF


class UniformRodPlatform3DOF(ParallelPlatform3DOF):
    """带“均匀杆件”质量/重力项的 3DOF 平台模型。"""

    def __init__(
        self,
        *,
        m_platform: float = 347.54,
        Ixx: float = 60.64,
        Iyy: float = 115.4,
        Izz: float = 80.0,
        r_base: float = 0.65,
        r_platform: float = 0.58,
        l_leg_nominal: float = 1.058,
        m_rod: float = 25.0,
        rod_length_nominal: float | None = None,
        g: float = 9.81,
        # 数值微分步长
        diff_eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(
            m_platform=m_platform,
            Ixx=Ixx,
            Iyy=Iyy,
            Izz=Izz,
            r_base=r_base,
            r_platform=r_platform,
            l_leg_nominal=l_leg_nominal,
            g=g,
            **kwargs,
        )

        self.m_rod = float(m_rod)
        if rod_length_nominal is None:
            rod_length_nominal = float(l_leg_nominal)
        self.rod_length_nominal = float(rod_length_nominal)
        self.diff_eps = float(diff_eps)

    # -------------------------
    # 基础几何量
    # -------------------------

    def platform_joints_global(self, q: np.ndarray) -> np.ndarray:
        """沿用父类：P_i(q)。"""
        return super().platform_joints_global(q)

    def _leg_unit_vectors(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回 (L, e) ，其中
        - L shape (3,)
        - e shape (3,3) 每行是 e_i^T
        """
        P = self.platform_joints_global(q)
        B = self.base_joints
        d = P - B
        L = np.linalg.norm(d, axis=1)
        L = np.maximum(L, 1e-9)
        e = d / L[:, None]
        return L, e

    def _J_Pi_numeric(self, q: np.ndarray, i: int) -> np.ndarray:
        """数值计算第 i 个铰点位置 P_i 对 q 的雅可比 J_{P_i} = dP_i/dq。

        返回 shape (3,3)，列对应 [z, alpha, beta]。
        """
        q = np.asarray(q, dtype=float).reshape(3)
        eps = self.diff_eps

        def P_i(q_):
            return self.platform_joints_global(q_)[i].astype(float)

        J = np.zeros((3, 3), dtype=float)
        for k in range(3):
            dq = np.zeros(3)
            dq[k] = eps
            fp = P_i(q + dq)
            fm = P_i(q - dq)
            J[:, k] = (fp - fm) / (2.0 * eps)

        return J

    # -------------------------
    # 动力学项
    # -------------------------

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """M(q) = M_plate + M_rod_trans + M_rod_rot"""
        q = np.asarray(q, dtype=float).reshape(3)

        # 5.1 平台动能对应的质量矩阵
        M_plate = super().mass_matrix(q)

        # 5.2 杆件平动动能：M_rod_trans = (m_rod/8) * sum J_Pi^T J_Pi
        M_rod_trans = np.zeros((3, 3), dtype=float)
        for i in range(3):
            JPi = self._J_Pi_numeric(q, i)
            M_rod_trans += JPi.T @ JPi
        M_rod_trans *= self.m_rod / 8.0

        # 5.3 杆件转动动能：M_rod_rot = (m_rod*l^2/24) * sum E_i^T E_i
        L, e = self._leg_unit_vectors(q)
        I3 = np.eye(3)
        M_rod_rot = np.zeros((3, 3), dtype=float)
        for i in range(3):
            JPi = self._J_Pi_numeric(q, i)  # 3x3
            proj = (I3 - np.outer(e[i], e[i])) / L[i]
            # E_i = d e_i / d q = proj @ dP_i/dq
            Ei = proj @ JPi
            M_rod_rot += Ei.T @ Ei
        M_rod_rot *= self.m_rod * (self.rod_length_nominal**2) / 24.0

        M = M_plate + M_rod_trans + M_rod_rot
        # 保证对称性（数值误差）
        M = 0.5 * (M + M.T)
        return M

    def coriolis_matrix(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """暂复用 plate 的解析 C(q,qd)。

        严格来说 rod 项也会引入额外 C，但实现会显著复杂；
        这版先用于“uniform 仿真可跑 + τ_s 计算一致”。
        """
        q = np.asarray(q, dtype=float).reshape(3)
        qd = np.asarray(qd, dtype=float).reshape(3)
        return super().coriolis_matrix(q, qd)

    def gravity_vector(self, q: np.ndarray) -> np.ndarray:
        """G(q) = dV/dq。

        文档给出的保持 q_s 不变的偏导约定下：
        - 平台项: [M*g, 0, 0]
        - 杆件项: (m_rod*g/2) * sum d z_{P_i} / d q

        这里用数值微分对 z_{P_i}(q) 求偏导。
        """
        q = np.asarray(q, dtype=float).reshape(3)

        # 平台重力项
        G_plate = np.array([self.m_platform * self.g, 0.0, 0.0], dtype=float)

        # 杆件重力项
        eps = self.diff_eps
        P = self.platform_joints_global(q)
        zP = P[:, 2]

        dz_dq = np.zeros(3, dtype=float)
        for k in range(3):
            dq = np.zeros(3)
            dq[k] = eps
            zP_p = self.platform_joints_global(q + dq)[:, 2]
            zP_m = self.platform_joints_global(q - dq)[:, 2]
            dz_dq[k] = np.sum((zP_p - zP_m) / (2.0 * eps))

        G_rods = (self.m_rod * self.g / 2.0) * dz_dq

        return G_plate + G_rods


def _quick_sanity():
    platform = UniformRodPlatform3DOF()
    q = np.array([1.058, 0.05, 0.03])
    qd = np.array([0.0, 0.0, 0.0])

    M = platform.mass_matrix(q)
    C = platform.coriolis_matrix(q, qd)
    G = platform.gravity_vector(q)

    print("M=\n", M)
    print("C=\n", C)
    print("G=", G)


if __name__ == "__main__":
    _quick_sanity()
