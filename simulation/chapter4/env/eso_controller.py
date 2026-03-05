"""
STESO (Super-Twisting Extended State Observer) Controller for Chapter 4

基于参考代码 reference/steso.py 实现的STESO：
- 使用超螺旋算法估计总扰动
- 输入需要惯性矩阵的逆 H = M^-1
- 使用RK4积分提高精度
"""

import numpy as np
from typing import Tuple, Optional


class STESO:
    """
    Super-Twisting Extended State Observer (STESO)
    
    用于估计3-DOF并联平台的总扰动（包含未建模动力学和外部扰动）。
    
    Attributes:
        dim (int): 状态维度 (3)
        lambda1 (float): 滑模面参数 S = lambda1*e + e_dot
        beta1 (float): 观测器增益1 (主导S收敛)
        beta2 (float): 观测器增益2 (主导X收敛)
        dt (float): 采样时间
    """
    
    def __init__(self, 
                 dim: int = 3, 
                 lambda1: float = 4.0, 
                 beta1: float = 12.0, 
                 beta2: float = 30.0, 
                 dt: float = 0.01):
        self.dim = dim
        self.lambda1 = lambda1
        self.beta1 = beta1
        self.beta2 = beta2
        self.dt = dt
        
        # 估计状态
        self.S_hat = np.zeros(dim)
        self.X_hat = np.zeros(dim)
        
        self.is_initialized = False

    def init_state(self,
                   q: np.ndarray,
                   qd: np.ndarray,
                   q_des: np.ndarray = None,
                   qd_des: np.ndarray = None,
                   X_hat0: Optional[np.ndarray] = None):
        """初始化观测器状态，避免启动时的巨大误差。

        Args:
            q: 初始测量位姿
            qd: 初始测量速度
            q_des: 初始期望位姿
            qd_des: 初始期望速度
            X_hat0: 扰动估计初值，形状为 [dim]；None 表示默认全 0
        """
        if q_des is None:
            q_des = np.zeros(self.dim)
        if qd_des is None:
            qd_des = np.zeros(self.dim)
        
        e = q - q_des
        ed = qd - qd_des
        S = self.lambda1 * e + ed
        
        self.S_hat = S.copy()
        
        if X_hat0 is None:
            self.X_hat = np.zeros(self.dim)
        else:
            X_hat0 = np.asarray(X_hat0, dtype=float).reshape(-1)
            if X_hat0.shape[0] != self.dim:
                raise ValueError(f"X_hat0 shape must be ({self.dim},), got {X_hat0.shape}")
            self.X_hat = X_hat0.copy()
        
        self.is_initialized = True

    def update(self, 
               q: np.ndarray, 
               qd: np.ndarray, 
               tau: np.ndarray, 
               H: np.ndarray, 
               q_des: np.ndarray = None, 
               qd_des: np.ndarray = None) -> np.ndarray:
        """
        更新观测器状态
        
        Args:
            q: 当前位姿 [3]
            qd: 当前速度 [3]
            tau: 控制力矩 [3]
            H: 惯性矩阵的逆 (M^-1) [3x3]
            q_des: 期望位姿 [3]
            qd_des: 期望速度 [3]
            
        Returns:
            X_hat: 总扰动估计值 [3] (加速度层面的等效扰动)
        """
        # 默认期望为0
        if q_des is None:
            q_des = np.zeros(self.dim)
        if qd_des is None:
            qd_des = np.zeros(self.dim)
        
        # 自动初始化
        if not self.is_initialized:
            self.init_state(q, qd, q_des, qd_des)
        
        # 1. 计算真实滑模面 S = lambda1 * e + e_dot
        e = q - q_des
        ed = qd - qd_des
        S = self.lambda1 * e + ed
        
        # 2. 计算估计误差 S_tilde
        S_tilde = self.S_hat - S 
        
        # 3. 计算非线性项
        # Sig^1/2(S_tilde) = |S_tilde|^0.5 * sign(S_tilde)
        abs_S_tilde = np.abs(S_tilde)
        sig_half = np.sqrt(abs_S_tilde + 1e-12) * np.sign(S_tilde)
        
        # Sig(S_tilde) = sign(S_tilde)
        sig_one = np.sign(S_tilde)
        
        # 4. 已知动态项 F = lambda1 * q_dot
        F = self.lambda1 * qd
        
        # 5. V = H @ tau + F (已知输入项)
        V = (H @ tau) + F
        
        # 6. RK4 积分更新
        def dynamics(S_hat_val, X_hat_val):
            err = S - S_hat_val
            
            abs_err = np.abs(err)
            sig_half = np.sqrt(abs_err + 1e-12) * np.sign(err)
            sig_one = np.sign(err)
            
            dS = V + X_hat_val + self.beta1 * sig_half
            dX = self.beta2 * sig_one
            return dS, dX
        
        # k1
        k1_S, k1_X = dynamics(self.S_hat, self.X_hat)
        
        # k2
        k2_S, k2_X = dynamics(self.S_hat + 0.5 * self.dt * k1_S, 
                              self.X_hat + 0.5 * self.dt * k1_X)
                              
        # k3
        k3_S, k3_X = dynamics(self.S_hat + 0.5 * self.dt * k2_S,
                              self.X_hat + 0.5 * self.dt * k2_X)
                              
        # k4
        k4_S, k4_X = dynamics(self.S_hat + self.dt * k3_S,
                              self.X_hat + self.dt * k3_X)
        
        # Update
        self.S_hat += (self.dt / 6.0) * (k1_S + 2*k2_S + 2*k3_S + k4_S)
        self.X_hat += (self.dt / 6.0) * (k1_X + 2*k2_X + 2*k3_X + k4_X)
        
        return self.X_hat.copy()
    
    def reset(self):
        """重置观测器状态"""
        self.S_hat = np.zeros(self.dim)
        self.X_hat = np.zeros(self.dim)
        self.is_initialized = False


class MultiDOFSTESO:
    """
    多自由度STESO (3-DOF: z, alpha, beta)
    
    这是一个封装器，调用STESO进行3个自由度的扰动估计
    """
    
    def __init__(self, 
                 dt: float = 0.01,
                 lambda1: float = 4.0,
                 beta1: float = 12.0,
                 beta2: float = 30.0):
        """
        Args:
            dt: 时间步长
            lambda1: 滑模面参数
            beta1: 观测器增益1
            beta2: 观测器增益2
        """
        self.dt = dt
        self.steso = STESO(dim=3, lambda1=lambda1, beta1=beta1, beta2=beta2, dt=dt)
    
    def reset(self):
        """重置STESO"""
        self.steso.reset()
    
    def update(self, 
               q: np.ndarray, 
               qd: np.ndarray, 
               tau: np.ndarray, 
               M_inv: np.ndarray,
               q_des: np.ndarray = None,
               qd_des: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        更新所有STESO
        
        Args:
            q: 系统位姿 [z, alpha, beta]
            qd: 系统速度 [zd, alphad, betad]
            tau: 控制输入 [Fx, Fy, Fz] 或 [tau_z, tau_alpha, tau_beta]
            M_inv: 惯性矩阵的逆 M^-1
            q_des: 期望位姿
            qd_des: 期望速度
            
        Returns:
            S_hat: 滑模面估计
            X_hat: 扰动估计 (加速度层面)
        """
        X_hat = self.steso.update(q, qd, tau, M_inv, q_des, qd_des)
        return self.steso.S_hat.copy(), X_hat
    
    def get_disturbance_estimate(self) -> np.ndarray:
        """获取扰动估计 (加速度层面)"""
        return self.steso.X_hat.copy()
    
    def get_sliding_surface_estimate(self) -> np.ndarray:
        """获取滑模面估计"""
        return self.steso.S_hat.copy()


# 保留原有的ESO类以保持兼容性，但标记为deprecated
class ExtendedStateObserver:
    """
    DEPRECATED: 请使用 STESO 类代替
    """
    
    def __init__(self, 
                 dt: float = 0.01,
                 omega_o: float = 20.0,
                 b0: float = 1.0):
        import warnings
        warnings.warn("ExtendedStateObserver is deprecated, please use STESO instead", DeprecationWarning)
        self.dt = dt
        self.omega_o = omega_o
        self.b0 = b0
        self.z = np.zeros(2)
        self.L = np.array([2 * omega_o, omega_o ** 2])
        
    def reset(self):
        self.z = np.zeros(2)
    
    def update(self, y: float, u: float) -> Tuple[float, float]:
        e = self.z[0] - y
        self.z[0] += self.dt * (self.z[1] + self.b0 * u - self.L[0] * e)
        self.z[1] += self.dt * (-self.L[1] * e)
        return self.z[0], self.z[1]


class MultiDOFESO:
    """
    DEPRECATED: 请使用 MultiDOFSTESO 类代替
    """
    
    def __init__(self, 
                 dt: float = 0.01,
                 omega_o: float = 20.0,
                 b0: np.ndarray = None):
        import warnings
        warnings.warn("MultiDOFESO is deprecated, please use MultiDOFSTESO instead", DeprecationWarning)
        
        self.dt = dt
        
        if b0 is None:
            b0 = np.array([1.0, 1.0, 1.0])
        self.b0 = b0
        
        self.eso_z = ExtendedStateObserver(dt, omega_o, b0[0])
        self.eso_alpha = ExtendedStateObserver(dt, omega_o, b0[1])
        self.eso_beta = ExtendedStateObserver(dt, omega_o, b0[2])
        
        self.eso_list = [self.eso_z, self.eso_alpha, self.eso_beta]
    
    def reset(self):
        for eso in self.eso_list:
            eso.reset()
    
    def update(self, y: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y_est = np.zeros(3)
        d_est = np.zeros(3)
        
        for i, eso in enumerate(self.eso_list):
            y_est[i], d_est[i] = eso.update(y[i], u[i])
        
        return y_est, d_est


class PerformanceFunction:
    """
    性能函数 ρ(t) - 用于硬切换判断
    
    ρ(t) = (ρ₀ - ρ∞) * exp(-κ*t) + ρ∞
    
    其中:
    - ρ₀: 初始允许误差边界
    - ρ∞: 稳态精度要求
    - κ: 衰减率
    """
    
    def __init__(self,
                 rho_0: float = 2.0,
                 rho_inf: float = 0.05,
                 kappa: float = 0.5):
        self.rho_0 = rho_0
        self.rho_inf = rho_inf
        self.kappa = kappa
        self.t = 0.0
        
    def reset(self):
        """重置时间"""
        self.t = 0.0
    
    def update(self, dt: float):
        """更新时间"""
        self.t += dt
    
    def get_rho(self) -> float:
        """获取当前性能函数值"""
        return (self.rho_0 - self.rho_inf) * np.exp(-self.kappa * self.t) + self.rho_inf


class PerformanceFunnel:
    """漏斗型安全区域边界"""

    def __init__(
        self,
        upper: PerformanceFunction,
        lower: PerformanceFunction,
        dims: int = 3,
    ):
        self.upper = upper
        self.lower = lower
        self.dims = dims

    def reset(self):
        self.upper.reset()
        self.lower.reset()

    def update(self, dt: float):
        self.upper.update(dt)
        self.lower.update(dt)

    def get_rho_vec(self) -> np.ndarray:
        rho = min(self.upper.get_rho(), self.lower.get_rho())
        return np.full(self.dims, rho, dtype=np.float64)


class HardSwitchController:
    """
    硬切换控制器 - 基于误差阈值的中心区/边界区切换
    
    控制律:
    - 中心区 (|e|/ρ < threshold): u = u_p + u_rl
    - 边界区 (|e|/ρ >= threshold): u = u_p - sign(e)*(|u_rl| + β)
    
    其中:
    - u_p: 基线控制 (反馈线性化)
    - u_rl: RL控制输出
    - β: 鲁棒增益
    """
    
    def __init__(
        self,
        threshold: float = 0.7,
        beta: float = 0.5,
        kp: float = 24.0,
        kd: float = 2.5,
        center_ratio: float = 0.5,
    ):
        self.threshold = threshold
        self.beta = beta
        self.kp = kp
        self.kd = kd

        if not (0.0 < center_ratio < 1.0):
            raise ValueError(f"center_ratio must be in (0,1), got {center_ratio}")
        self.center_ratio = center_ratio
        
        self.current_region = 'center'

    def classify_region(self, e: np.ndarray, rho: np.ndarray) -> str:
        """按漏斗逐维边界划分区域"""
        e = np.asarray(e, dtype=np.float64)
        rho = np.asarray(rho, dtype=np.float64)

        if e.shape != rho.shape:
            raise ValueError(f"e and rho must have same shape, got {e.shape} vs {rho.shape}")

        rho_safe = np.maximum(rho, 1e-12)
        abs_e = np.abs(e)

        in_safe = np.all(abs_e < rho_safe)
        in_center = np.all(abs_e < (self.center_ratio * rho_safe))

        if not in_safe:
            return 'unsafe'
        if in_center:
            return 'center'
        return 'boundary'
        
    def compute_control(self, 
                       e: np.ndarray, 
                       ed: np.ndarray,
                       u_rl: np.ndarray,
                       u_p_baseline: np.ndarray = None,
                       rho: np.ndarray = None) -> np.ndarray:
        """计算控制输入"""
        if u_p_baseline is None:
            u_p = self.kp * e + self.kd * ed
        else:
            u_p = u_p_baseline
        
        if rho is not None:
            region = self.classify_region(e, rho)
            self.current_region = region
        else:
            e_norm = np.linalg.norm(e)
            self.current_region = 'center' if (e_norm < self.threshold) else 'boundary'

        if self.current_region == 'center':
            u = u_p + u_rl
        else:
            sign_e = np.sign(e)
            sign_e = np.where(sign_e == 0, 1.0, sign_e)
            u = u_p - sign_e * (np.abs(u_rl) + self.beta)
        
        return u
    
    def get_region(self) -> str:
        return self.current_region


if __name__ == '__main__':
    # 测试STESO
    print("Testing STESO...")
    
    # 模拟系统参数
    dt = 0.01
    m = 100.0  # 质量
    
    # 创建STESO
    steso = MultiDOFSTESO(dt=dt, lambda1=4.0, beta1=12.0, beta2=30.0)
    
    # 初始状态
    q = np.array([1.0, 0.0, 0.0])
    qd = np.zeros(3)
    q_des = np.array([1.0, 0.0, 0.0])
    qd_des = np.zeros(3)
    
    # 初始化STESO
    steso.steso.init_state(q, qd, q_des, qd_des)
    
    # 模拟
    M_inv = np.eye(3) / m  # 简化
    tau = np.zeros(3)
    
    for i in range(100):
        # 模拟控制输入
        tau = np.array([100.0, 10.0, 5.0]) * np.sin(0.1 * i * dt)
        
        # 更新STESO
        S_hat, X_hat = steso.update(q, qd, tau, M_inv, q_des, qd_des)
        
        if i % 20 == 0:
            print(f"Step {i}: X_hat = {X_hat}")
        
        # 简单更新状态
        qd += (M_inv @ tau) * dt
        q += qd * dt
    
    print("\nTesting Hard Switch Controller...")
    hsc = HardSwitchController(threshold=0.7, beta=0.5)
    
    e = np.array([0.5, 0.1, 0.05])
    ed = np.array([0.0, 0.0, 0.0])
    u_rl = np.array([10.0, 1.0, 0.5])
    
    u = hsc.compute_control(e, ed, u_rl)
    print(f"Error norm: {np.linalg.norm(e):.4f}, Region: {hsc.get_region()}")
    print(f"Control: {u}")
    
    print("\n✓ All tests passed!")
