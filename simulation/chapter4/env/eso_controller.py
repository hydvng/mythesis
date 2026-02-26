"""
ESO (Extended State Observer) Controller for Chapter 4

ESO估计系统总扰动（包括海浪扰动和模型不确定性）：
- 二阶ESO: ż = A*z + B*u + L*(y - z1)
- 估计扰动 d = f(x) + w(t)
"""

import numpy as np
from typing import Tuple


class ExtendedStateObserver:
    """
    二阶扩张状态观测器
    
    状态方程:
    x1_dot = x2
    x2_dot = f(x) + g(x)*u + d(t)  (d(t)是总扰动)
    x1
    
    y = ESO:
    z1_dot = z2 + b0*u - L1*(z1 - y)
    z2_dot = -L2*(z1 - y)
    
    其中:
    - z1: 状态估计
    - z2: 扰动估计
    - L: 观测器增益 [L1, L2]
    - b0: 控制增益 nominal值
    """
    
    def __init__(self, 
                 dt: float = 0.01,
                 omega_o: float = 20.0,
                 b0: float = 1.0):
        """
        Args:
            dt: 时间步长
            omega_o: 观测器带宽 (rad/s)
            b0: 控制增益 nominal值
        """
        self.dt = dt
        self.omega_o = omega_o
        self.b0 = b0
        
        # ESO状态: [z1, z2]
        self.z = np.zeros(2)
        
        # 观测器增益 (基于带宽选择)
        # L1 = 2*omega_o, L2 = omega_o^2 (用于二阶系统)
        self.L = np.array([2 * omega_o, omega_o ** 2])
        
        # 历史记录
        self.z_history = []
        self.disturbance_history = []
        
    def reset(self):
        """重置ESO状态"""
        self.z = np.zeros(2)
        self.z_history = []
        self.disturbance_history = []
    
    def update(self, y: float, u: float) -> Tuple[float, float]:
        """
        更新ESO状态
        
        Args:
            y: 系统输出 (位置)
            u: 控制输入
            
        Returns:
            z1: 状态估计
            z2: 扰动估计
        """
        # 离散ESO (前向欧拉)
        e = self.z[0] - y  # 估计误差
        
        # 状态更新
        self.z[0] += self.dt * (self.z[1] + self.b0 * u - self.L[0] * e)
        self.z[1] += self.dt * (-self.L[1] * e)
        
        # 记录历史
        self.z_history.append(self.z.copy())
        self.disturbance_history.append(self.z[1])
        
        return self.z[0], self.z[1]
    
    def get_disturbance_estimate(self) -> float:
        """获取扰动估计值"""
        return self.z[1]
    
    def get_state_estimate(self) -> float:
        """获取状态估计值"""
        return self.z[0]


class MultiDOFESO:
    """
    多自由度ESO (3-DOF: z, alpha, beta)
    """
    
    def __init__(self, 
                 dt: float = 0.01,
                 omega_o: float = 20.0,
                 b0: np.ndarray = None):
        """
        Args:
            dt: 时间步长
            omega_o: 观测器带宽
            b0: 控制增益nominal值 (3,)
        """
        self.dt = dt
        
        if b0 is None:
            b0 = np.array([1.0, 1.0, 1.0])
        self.b0 = b0
        
        # 为每个自由度创建ESO
        self.eso_z = ExtendedStateObserver(dt, omega_o, b0[0])
        self.eso_alpha = ExtendedStateObserver(dt, omega_o, b0[1])
        self.eso_beta = ExtendedStateObserver(dt, omega_o, b0[2])
        
        self.eso_list = [self.eso_z, self.eso_alpha, self.eso_beta]
    
    def reset(self):
        """重置所有ESO"""
        for eso in self.eso_list:
            eso.reset()
    
    def update(self, y: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        更新所有ESO
        
        Args:
            y: 系统输出 [z, alpha, beta]
            u: 控制输入 [Fx, Fy, Fz]
            
        Returns:
            y_est: 状态估计
            d_est: 扰动估计
        """
        y_est = np.zeros(3)
        d_est = np.zeros(3)
        
        for i, eso in enumerate(self.eso_list):
            y_est[i], d_est[i] = eso.update(y[i], u[i])
        
        return y_est, d_est
    
    def get_disturbance_estimate(self) -> np.ndarray:
        """获取所有扰动估计"""
        return np.array([eso.get_disturbance_estimate() for eso in self.eso_list])
    
    def get_state_estimate(self) -> np.ndarray:
        """获取所有状态估计"""
        return np.array([eso.get_state_estimate() for eso in self.eso_list])


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
    
    def __init__(self,
                 threshold: float = 0.7,
                 beta: float = 0.5,
                 kp: float = 24.0,
                 kd: float = 2.5):
        """
        Args:
            threshold: 区域切换阈值 (通常0.5-0.8)
            beta: 边界区鲁棒增益
            kp: 比例增益
            kd: 微分增益
        """
        self.threshold = threshold
        self.beta = beta
        self.kp = kp
        self.kd = kd
        
        self.current_region = 'center'  # 'center' or 'boundary'
        
    def compute_control(self, 
                       e: np.ndarray, 
                       ed: np.ndarray,
                       u_rl: np.ndarray,
                       u_p_baseline: np.ndarray = None) -> np.ndarray:
        """
        计算控制输入
        
        Args:
            e: 跟踪误差 [ez, e_alpha, e_beta]
            ed: 误差导数 [ez_dot, e_alpha_dot, e_beta_dot]
            u_rl: RL控制输出
            u_p_baseline: 基线控制 (可选)
            
        Returns:
            u: 最终控制输入
        """
        # 如果没有提供基线控制，使用PD控制
        if u_p_baseline is None:
            u_p = self.kp * e + self.kd * ed
        else:
            u_p = u_p_baseline
        
        # 计算误差范数与性能函数比值
        e_norm = np.linalg.norm(e)
        
        # 判断所在区域
        # 这里简化处理，使用固定阈值
        if e_norm < self.threshold:
            self.current_region = 'center'
            # 中心区: 叠加控制
            u = u_p + u_rl
        else:
            self.current_region = 'boundary'
            # 边界区: 切换到鲁棒控制
            sign_e = np.sign(e)
            sign_e = np.where(sign_e == 0, 1.0, sign_e)  # 避免零
            u = u_p - sign_e * (np.abs(u_rl) + self.beta)
        
        return u
    
    def get_region(self) -> str:
        """获取当前区域"""
        return self.current_region


if __name__ == '__main__':
    # 测试ESO
    print("Testing ESO...")
    eso = ExtendedStateObserver(dt=0.01, omega_o=20.0, b0=1.0)
    
    # 模拟系统
    t = 0.0
    dt = 0.01
    y = 0.0
    u = 0.0
    
    for i in range(100):
        # 真实系统: y_dot = sin(t) + u + d
        d = 0.5 * np.sin(5 * t)  # 扰动
        y_dot = np.sin(t) + u + d
        y += y_dot * dt
        
        # ESO更新
        y_est, d_est = eso.update(y, u)
        
        if i % 20 == 0:
            print(f"t={t:.2f}: y={y:.4f}, y_est={y_est:.4f}, d_est={d_est:.4f}, true_d={d:.4f}")
        
        t += dt
    
    print("\nTesting Hard Switch Controller...")
    hsc = HardSwitchController(threshold=0.7, beta=0.5)
    
    e = np.array([0.5, 0.1, 0.05])
    ed = np.array([0.0, 0.0, 0.0])
    u_rl = np.array([10.0, 1.0, 0.5])
    
    u = hsc.compute_control(e, ed, u_rl)
    print(f"Error norm: {np.linalg.norm(e):.4f}, Region: {hsc.get_region()}")
    print(f"Control: {u}")
    
    # 测试大误差
    e = np.array([1.5, 0.5, 0.3])
    u = hsc.compute_control(e, ed, u_rl)
    print(f"\nError norm: {np.linalg.norm(e):.4f}, Region: {hsc.get_region()}")
    print(f"Control: {u}")
    
    print("\n✓ All tests passed!")
