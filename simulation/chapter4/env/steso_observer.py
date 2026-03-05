"""STESO (Super-Twisting Extended State Observer) for Chapter 4.

This implementation is aligned to the *structure* of the reference `steso.py` you provided,
but adapted to the Chapter 4 environment signals.

Key design decision (IMPORTANT):

We estimate the lumped disturbance in the **acceleration domain**:

    qdd = qdd_nom + w

where `qdd_nom` is computed from the known model using the applied input `u_legs`.
Then we map it back to the generalized-force domain to match `d_actual`:

    d_hat = M(q) @ w_hat

This avoids ambiguity around whether the observer's state lives in force-domain or acceleration-domain,
and makes the comparison against `d_actual = M*qdd + C*qd + G - u` directly meaningful.

We use a boundary-layer saturation instead of a hard sign() to reduce numerical chattering.
"""

from __future__ import annotations

import numpy as np


class STESO:
    def __init__(
        self,
        dim: int = 3,
        lambda1: float = 5.0,
        beta1: float = 20.0,
        beta2: float = 200.0,
        dt: float = 0.01,
        eps: float = 1e-6,
        phi: float = 1e-2,
        w_max: float | None = 2.0,
    ):
        self.dim = dim
        self.lambda1 = float(lambda1)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.dt = float(dt)
        self.eps = float(eps)
        self.phi = float(phi)
        self.w_max = None if w_max is None else float(w_max)

        self.S_hat = np.zeros(self.dim, dtype=np.float64)
        # w_hat: acceleration-domain lumped disturbance (same unit as qdd)
        self.w_hat = np.zeros(self.dim, dtype=np.float64)
        self._initialized = False

    def reset(self):
        self.S_hat[:] = 0.0
        self.w_hat[:] = 0.0
        self._initialized = False

    def init_state(self, q: np.ndarray, qd: np.ndarray, q_des: np.ndarray, qd_des: np.ndarray):
        e = q - q_des
        ed = qd - qd_des
        S = self.lambda1 * e + ed
        self.S_hat = S.copy()
        self.w_hat[:] = 0.0
        self._initialized = True

    @staticmethod
    def _sig_half(x: np.ndarray, eps: float) -> np.ndarray:
        # |x|^{1/2} * sign(x) with an epsilon to avoid sqrt(0) issues
        return np.sqrt(np.abs(x) + eps) * np.sign(x)

    @staticmethod
    def _sat(x: np.ndarray, phi: float) -> np.ndarray:
        """Saturation (boundary layer) approximation of sign(x)."""
        if phi <= 0:
            return np.sign(x)
        return np.clip(x / phi, -1.0, 1.0)

    def update(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        q_des: np.ndarray,
        qd_des: np.ndarray,
        M: np.ndarray,
        qdd_nom: np.ndarray,
    ) -> np.ndarray:
        """One-step STESO update.

        Args:
            q, qd: measured states
            q_des, qd_des: desired trajectory
            M: mass matrix at current q
            qdd_nom: nominal acceleration from the known model (without external force disturbance)

        Returns:
            d_hat: estimated lumped disturbance in generalized-force domain, shape (3,)
        """
        q = np.asarray(q, dtype=np.float64)
        qd = np.asarray(qd, dtype=np.float64)
        q_des = np.asarray(q_des, dtype=np.float64)
        qd_des = np.asarray(qd_des, dtype=np.float64)
        M = np.asarray(M, dtype=np.float64)
        qdd_nom = np.asarray(qdd_nom, dtype=np.float64)

        if not self._initialized:
            self.init_state(q, qd, q_des, qd_des)

        # Sliding variable
        e = q - q_des
        ed = qd - qd_des
        S = self.lambda1 * e + ed

        # Known part in S_dot: lambda1*e_dot + qdd_nom + w
        V = self.lambda1 * ed + qdd_nom

        # Super-twisting on err = S - S_hat
        err = S - self.S_hat
        sig_half = self._sig_half(err, self.eps)
        sig_one = self._sat(err, self.phi)

        # Discrete Euler update (structure matches reference steso.py)
        # S_hat_dot = V + w_hat + beta1*Sig^{1/2}(err)
        # w_hat_dot = beta2*Sat(err)
        dS_hat = V + self.w_hat + self.beta1 * sig_half
        dw_hat = self.beta2 * sig_one

        self.S_hat = self.S_hat + self.dt * dS_hat
        self.w_hat = self.w_hat + self.dt * dw_hat

        if self.w_max is not None:
            self.w_hat = np.clip(self.w_hat, -self.w_max, self.w_max)

        # Map acceleration-domain disturbance to force-domain estimate
        d_hat = M @ self.w_hat
        return d_hat.copy()
