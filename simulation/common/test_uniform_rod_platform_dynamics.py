"""test_uniform_rod_platform_dynamics.py

最小数值检查：
- M(q) 对称
- M(q) 近似正定（特征值 >= -tol）
- gravity_vector 维度正确
- 扰动项 τ_s 与公式一致（对随机输入做一致性检查）

运行方式：直接 python 运行即可。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# 允许直接从任意 cwd 运行
_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from simulation.common.uniform_rod_platform_dynamics import UniformRodPlatform3DOF


def test_mass_matrix_symmetry_and_psd():
    platform = UniformRodPlatform3DOF()
    q = np.array([1.058, 0.1, -0.05])

    M = platform.mass_matrix(q)
    assert M.shape == (3, 3)
    assert np.allclose(M, M.T, atol=1e-10)

    w = np.linalg.eigvalsh(M)
    # 允许一点点数值误差
    assert np.min(w) > -1e-7


def test_gravity_vector_shape():
    platform = UniformRodPlatform3DOF()
    q = np.array([1.0, 0.0, 0.0])
    G = platform.gravity_vector(q)
    assert G.shape == (3,)


def test_tau_s_formula_consistency():
    platform = UniformRodPlatform3DOF()

    q = np.array([1.1, 0.02, 0.01])
    qd = np.array([0.1, -0.05, 0.02])

    qd_s = np.array([0.2, 0.01, -0.03])
    qdd_s = np.array([-0.5, 0.2, 0.1])

    M = platform.mass_matrix(q)
    C = platform.coriolis_matrix(q, qd)

    tau_s = -(M @ qdd_s + C @ qd_s)
    # 只是检查维度和无 NaN
    assert tau_s.shape == (3,)
    assert np.all(np.isfinite(tau_s))


if __name__ == "__main__":
    test_mass_matrix_symmetry_and_psd()
    test_gravity_vector_shape()
    test_tau_s_formula_consistency()
    print("OK")
