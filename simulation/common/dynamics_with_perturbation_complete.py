"""
船载Stewart平台完整动力学模型（含扰动）
基于文档: dynamics_with_perturbation_complete.md

核心公式: M(q_u)*qdd_c + C(q_u, qd_u)*qd_c + G = tau_u - M*qdd_s - C*qd_s

作者: 二狗 🐕
日期: 2026-03-08
"""

import numpy as np
from typing import Tuple, Dict
import matplotlib.pyplot as plt


class PlatformDynamicsWithPerturbation:
    """
    带扰动的船载稳定平台动力学模型
    
    状态变量:
    - q_c: 平台控制量 [z_c, alpha_c, beta_c]
    - qd_c: 平台控制速度
    - q_s: 船体运动 [z_s, alpha_s, beta_s] (扰动输入)
    """
    
    def __init__(self,
                 m_platform: float = 347.54,
                 Ixx: float = 60.64,
                 Iyy: float = 115.4,
                 Izz: float = 80.0,
                 m_rods: float = 25.0,  # 杆件质量 (上9kg + 下16kg)
                 g: float = 9.81,
                 dt: float = 0.01):
        """初始化模型参数"""
        
        self.m_platform = m_platform
        self.m_rods = m_rods
        self.m_total = m_platform + m_rods  # 总质量
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz
        self.g = g
        self.dt = dt
        
    def compute_M(self, q_u: np.ndarray) -> np.ndarray:
        """
        计算质量矩阵 M(q_u)
        
        M(q_u) = [[m_total, 0, 0],
                  [0, Ixx, 0],
                  [0, 0, I_eq]]
        其中 I_eq = Iyy*cos²(α) + Izz*sin²(α)
        """
        alpha = q_u[1]
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        I_eq = self.Iyy * ca**2 + self.Izz * sa**2
        
        M = np.array([
            [self.m_total, 0, 0],
            [0, self.Ixx, 0],
            [0, 0, I_eq]
        ])
        
        return M
    
    def compute_C(self, q_u: np.ndarray, qd_u: np.ndarray) -> np.ndarray:
        """
        计算科里奥利矩阵 C(q_u, qd_u)
        
        C = [[0, 0, 0],
             [0, 0, -K*dq_beta],
             [0, K*dq_beta, K*dq_alpha]]
        
        其中 K = (Izz - Iyy) * sin(α) * cos(α)
        """
        alpha = q_u[1]
        dalpha = qd_u[1]
        dbeta = qd_u[2]
        
        K = (self.Izz - self.Iyy) * np.sin(alpha) * np.cos(alpha)
        
        C = np.array([
            [0, 0, 0],
            [0, 0, -K * dbeta],
            [0, K * dbeta, K * dalpha]
        ])
        
        return C
    
    def compute_G(self, q_u: np.ndarray) -> np.ndarray:
        """
        计算重力向量 G(q_u)
        
        G = [m_total*g, 0, 0]^T
        """
        return np.array([self.m_total * self.g, 0.0, 0.0])
    
    def compute_perturbation(self, q_s: np.ndarray, qd_s: np.ndarray, 
                            q_u: np.ndarray) -> np.ndarray:
        """
        计算扰动项 τ_dist = -M*qdd_s - C*qd_s
        
        扰动来自船体运动，通过惯性力和科里奥利力影响平台
        """
        # 这里简化处理：假设 qd_s 是已知的
        # 实际实现中需要从船体运动生成器获取
        M = self.compute_M(q_u)
        
        # 简化：假设加速度已知（或从速度差分得到）
        # τ_dist ≈ -M * qdd_s（忽略科里奥利项的简化）
        qdd_s = np.zeros(3)  # 需要从外部输入
        
        # 完整扰动项（需要外部提供 qdd_s）
        return -M @ qdd_s  # 简化版本
    
    def compute_qdd_c(self, q_c: np.ndarray, qd_c: np.ndarray,
                      tau_u: np.ndarray,
                      q_s: np.ndarray, qd_s: np.ndarray, 
                      qdd_s: np.ndarray) -> np.ndarray:
        """
        核心：求解平台控制加速度 qdd_c
        
        公式: qdd_c = M⁻¹(τ_u - C·qd_u - G) - qdd_s
        
        其中:
        - q_u = q_c + q_s (上平台总位移)
        - qd_u = qd_c + qd_s (上平台总速度)
        """
        # 上平台状态 = 控制量 + 船体运动
        q_u = q_c + q_s
        qd_u = qd_c + qd_s
        
        # 计算 M, C, G
        M = self.compute_M(q_u)
        C = self.compute_C(q_u, qd_u)
        G = self.compute_G(q_u)
        
        # 求解: M*qdd_c = τ_u - C*qd_u - G - M*qdd_s
        # 即: qdd_c = M⁻¹(τ_u - C*qd_u - G - M*qdd_s)
        rhs = tau_u - C @ qd_u - G - M @ qdd_s
        
        try:
            qdd_c = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            qdd_c = np.linalg.pinv(M) @ rhs
        
        return qdd_c


class ShipMotionGenerator:
    """
    船体运动生成器
    
    基于 RAO（Response Amplitude Operator）和 JONSWAP 谱生成船体运动
    """
    
    def __init__(self, 
                 Hs: float = 2.0,      # 有义波高 (m)
                 Tp: float = 8.0,      # 谱峰周期 (s)
                 gamma: float = 3.3,    # JONSWAP谱峰值因子
                 omega_min: float = 0.3,
                 omega_max: float = 2.0,
                 n_components: int = 50):
        """初始化船体运动参数"""
        
        self.Hs = Hs
        self.Tp = Tp
        self.gamma = gamma
        self.omega_min = omega_min
        self.omega_max = omega_max
        self.n_components = n_components
        
        # 预计算频率和相位
        self._generate_frequencies()
        
    def _generate_frequencies(self):
        """生成频率分量"""
        # 频率范围
        self.omegas = np.linspace(self.omega_min, self.omega_max, 
                                  self.n_components)
        
        # JONSWAP 谱
        omega_p = 2 * np.pi / self.Tp  # 谱峰频率
        
        sigma = np.where(self.omegas <= omega_p, 0.07, 0.09)
        gamma_exp = np.exp(-((self.omegas - omega_p)**2) / 
                          (2 * sigma**2 * omega_p**2))
        S_omega = (self.gamma**gamma_exp * 
                   5.0 * self.Hs**2 / (16 * np.pi * omega_p**4) *
                   self.omegas**(-5) * np.exp(-5.0/4 * (omega_p/self.omegas)**4))
        
        # 波幅 A_k = sqrt(2 * S(ω) * dω)
        domega = self.omegas[1] - self.omegas[0]
        self.amplitudes = np.sqrt(2 * S_omega * domega)
        
        # 随机相位
        np.random.seed(42)  # 固定种子以复现
        self.phases = np.random.uniform(0, 2*np.pi, self.n_components)
        
    def jonswap_spectrum(self, omega: float) -> float:
        """计算 JONSWAP 谱密度"""
        omega_p = 2 * np.pi / self.Tp
        
        if omega <= omega_p:
            sigma = 0.07
        else:
            sigma = 0.09
            
        gamma_exp = self.gamma ** np.exp(-((omega - omega_p)**2) / 
                                          (2 * sigma**2 * omega_p**2))
        
        return (gamma_exp * 5.0 * self.Hs**2 / (16 * np.pi * omega_p**4) *
                omega**(-5) * np.exp(-5.0/4 * (omega_p/omega)**4))
    
    def compute_rao(self, omega: float, dof: str = 'heave') -> float:
        """
        简化 RAO 模型
        
        实际应该从船舶运动数据库读取，这里用简化模型
        dof: 'heave'(z), 'roll'(alpha), 'pitch'(beta)
        """
        # 简化的 RAO 传递函数
        if dof == 'heave':
            # 升沉 RAO: 接近 1
            return 0.8
        elif dof == 'roll':
            # 横摇 RAO: 与频率相关
            return 30.0 * omega / (omega**2 + 0.5)
        elif dof == 'pitch':
            # 俯仰 RAO
            return 25.0 * omega / (omega**2 + 0.3)
        else:
            return 1.0
    
    def generate_ship_motion(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        生成船体运动 (位置、速度、加速度)
        
        返回: q_s, qd_s, qdd_s
        """
        # 多频叠加
        q_s = np.zeros(3)
        qd_s = np.zeros(3)
        qdd_s = np.zeros(3)
        
        for i, omega in enumerate(self.omegas):
            A = self.amplitudes[i]
            phi = self.phases[i]
            phase = omega * t + phi
            
            # RAO 传递函数
            rao_z = self.compute_rao(omega, 'heave')
            rao_alpha = self.compute_rao(omega, 'roll')
            rao_beta = self.compute_rao(omega, 'pitch')
            
            # 位置
            q_s[0] += rao_z * A * np.cos(phase)
            q_s[1] += rao_alpha * np.deg2rad(A * 0.5) * np.cos(phase)
            q_s[2] += rao_beta * np.deg2rad(A * 0.5) * np.cos(phase)
            
            # 速度 (导数)
            qd_s[0] += rao_z * A * (-omega) * np.sin(phase)
            qd_s[1] += rao_alpha * np.deg2rad(A * 0.5) * (-omega) * np.sin(phase)
            qd_s[2] += rao_beta * np.deg2rad(A * 0.5) * (-omega) * np.sin(phase)
            
            # 加速度 (二阶导数)
            qdd_s[0] += rao_z * A * (-omega**2) * np.cos(phase)
            qdd_s[1] += rao_alpha * np.deg2rad(A * 0.5) * (-omega**2) * np.cos(phase)
            qdd_s[2] += rao_beta * np.deg2rad(A * 0.5) * (-omega**2) * np.cos(phase)
        
        return q_s, qd_s, qdd_s


def test_dynamics_with_perturbation():
    """测试：带扰动的动力学仿真"""
    
    print("=" * 60)
    print("测试：船载稳定平台动力学（含扰动）")
    print("=" * 60)
    
    # 初始化
    dynamics = PlatformDynamicsWithPerturbation(
        m_platform=347.54,
        Ixx=60.64,
        Iyy=115.4,
        Izz=80.0,
        m_rods=25.0,
        g=9.81,
        dt=0.01
    )
    
    ship = ShipMotionGenerator(Hs=2.0, Tp=8.0)
    
    # 仿真参数
    t_end = 30.0
    dt = 0.01
    n_steps = int(t_end / dt)
    
    # 状态
    t_array = np.zeros(n_steps)
    q_c = np.zeros((n_steps, 3))      # 平台控制量
    qd_c = np.zeros((n_steps, 3))     # 平台控制速度
    q_s = np.zeros((n_steps, 3))      # 船体运动
    qd_s = np.zeros((n_steps, 3))     # 船体速度
    qdd_s = np.zeros((n_steps, 3))   # 船体加速度
    tau_u = np.zeros((n_steps, 3))    # 控制力
    
    # 初始条件
    q_c[0] = np.array([0.0, 0.0, 0.0])
    qd_c[0] = np.array([0.0, 0.0, 0.0])
    
    print(f"\n仿真参数:")
    print(f"  - 时长: {t_end}s")
    print(f"  - 步长: {dt}s")
    print(f"  - 步数: {n_steps}")
    print(f"  - 海况: Hs={ship.Hs}m, Tp={ship.Tp}s")
    
    # 仿真循环
    print("\n正在仿真...")
    for i in range(n_steps - 1):
        t = i * dt
        
        # 获取船体运动
        q_s[i], qd_s[i], qdd_s[i] = ship.generate_ship_motion(t)
        
        # 简单控制器：使用 PD 控制试图跟踪零位（抵消扰动）
        # tau_u = -K_p * q_c - K_d * qd_c (这里简化为零输入测试扰动响应)
        Kp = np.diag([10000, 5000, 5000])
        Kd = np.diag([5000, 2000, 2000])
        tau_u[i] = -Kp @ q_c[i] - Kd @ qd_c[i]
        
        # 计算加速度
        qdd_c = dynamics.compute_qdd_c(
            q_c[i], qd_c[i], tau_u[i],
            q_s[i], qd_s[i], qdd_s[i]
        )
        
        # 积分更新 (Euler方法)
        qd_c[i+1] = qd_c[i] + qdd_c * dt
        q_c[i+1] = q_c[i] + qd_c[i] * dt
        t_array[i+1] = t + dt
        
        if i % 1000 == 0:
            print(f"  进度: {100*i/n_steps:.1f}%")
    
    # 最终状态
    print(f"\n最终状态:")
    print(f"  平台位移 z: {q_c[-1, 0]*1000:.2f} mm")
    print(f"  平台转角 α: {np.rad2deg(q_c[-1, 1]):.3f} deg")
    print(f"  平台转角 β: {np.rad2deg(q_c[-1, 2]):.3f} deg")
    
    # 绘制结果
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # 船体运动
    axes[0].plot(t_array, q_s[:, 0] * 1000, 'b-', label='船体 z', alpha=0.7)
    axes[0].set_ylabel('z (mm)')
    axes[0].set_title('船体运动 (扰动输入)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(t_array, np.rad2deg(q_s[:, 1]), 'b-', label='船体 α (roll)', alpha=0.7)
    axes[1].plot(t_array, np.rad2deg(q_s[:, 2]), 'r-', label='船体 β (pitch)', alpha=0.7)
    axes[1].set_ylabel('角度 (deg)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 平台响应
    axes[2].plot(t_array, q_c[:, 0] * 1000, 'g-', label='平台 z', alpha=0.8)
    axes[2].set_xlabel('时间 (s)')
    axes[2].set_ylabel('z (mm)')
    axes[2].set_title('平台控制响应 (PD控制)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/Documents/mythesis/simulation/common/test_perturbation.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n图片已保存到: simulation/common/test_perturbation.png")
    
    return t_array, q_c, q_s


if __name__ == "__main__":
    test_dynamics_with_perturbation()
