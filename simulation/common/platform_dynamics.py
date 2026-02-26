"""
3-UPS/PU并联船载稳定平台完整动力学模型
第2章：系统建模与约束分析

坐标系定义：
- 基座坐标系 {B}: 原点在下平台中心，Z轴垂直向上
- 平台坐标系 {P}: 原点在动平台中心，随动平台运动

姿态定义（Z-Y-X欧拉角）：
- α (roll): 绕X轴旋转（横摇）
- β (pitch): 绕Y轴旋转（纵摇）  
- z: 沿Z轴平移（升沉）

自由度：3-DOF [z, α, β]
控制输入：3个电动缸力 [F1, F2, F3]
"""

import numpy as np
from typing import Tuple, Optional, Dict
from scipy.integrate import solve_ivp


class ParallelPlatform3DOF:
    """
    3-UPS/PU并联船载稳定平台完整动力学模型（三自由度）
    
    状态变量 q = [z, α, β]^T
    - z: 升沉位移 (m)
    - α: 横摇角 (rad)
    - β: 纵摇角 (rad)
    
    控制输入 u = [F1, F2, F3]^T
    """
    
    def __init__(self,
                 m_platform: float = 347.54,
                 Ixx: float = 60.64,
                 Iyy: float = 115.4,
                 Izz: float = 80.0,
                 r_base: float = 0.65,
                 r_platform: float = 0.58,
                 l_leg_nominal: float = 1.058,
                 l_leg_min: float = 0.808,
                 l_leg_max: float = 1.308,
                 m_cylinder: float = 25.0,
                 F_coulomb: float = 100.0,
                 F_viscous: float = 500.0,
                 u_max: float = 30000.0,
                 u_min: float = -30000.0,
                 v_max: float = 0.2,
                 alpha_max: float = np.pi/6,
                 beta_max: float = np.pi/6,
                 z_min: float = 0.808,
                 z_max: float = 1.308,
                 g: float = 9.81,
                 dt: float = 0.001):
        """初始化模型参数"""
        
        self.params = {
            'm_platform': m_platform,
            'Ixx': Ixx, 'Iyy': Iyy, 'Izz': Izz,
            'r_base': r_base, 'r_platform': r_platform,
            'l_leg_nominal': l_leg_nominal,
            'l_leg_min': l_leg_min, 'l_leg_max': l_leg_max,
            'm_cylinder': m_cylinder,
            'F_coulomb': F_coulomb, 'F_viscous': F_viscous,
            'u_max': u_max, 'u_min': u_min, 'v_max': v_max,
            'alpha_max': alpha_max, 'beta_max': beta_max,
            'z_min': z_min, 'z_max': z_max,
            'g': g, 'dt': dt
        }
        
        self.m_platform = m_platform
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz
        self.r_base = r_base
        self.r_platform = r_platform
        self.l_leg_min = l_leg_min
        self.l_leg_max = l_leg_max
        self.g = g
        self.dt = dt
        self.u_max = u_max
        self.u_min = u_min
        self.v_max = v_max
        self.alpha_max = alpha_max
        self.beta_max = beta_max
        self.z_min = z_min
        self.z_max = z_max
        
        self.base_joints = self._compute_base_joints()
        self.platform_joints_local = self._compute_platform_joints()
        
    def _compute_base_joints(self) -> np.ndarray:
        """计算基座铰链位置"""
        angles = np.array([0, 2*np.pi/3, 4*np.pi/3])
        joints = np.zeros((3, 3))
        for i in range(3):
            joints[i, 0] = self.r_base * np.cos(angles[i])
            joints[i, 1] = self.r_base * np.sin(angles[i])
            joints[i, 2] = 0.0
        return joints
    
    def _compute_platform_joints(self) -> np.ndarray:
        """计算动平台铰链位置"""
        angles = np.array([0, 2*np.pi/3, 4*np.pi/3])
        joints = np.zeros((3, 3))
        for i in range(3):
            joints[i, 0] = self.r_platform * np.cos(angles[i])
            joints[i, 1] = self.r_platform * np.sin(angles[i])
            joints[i, 2] = 0.0
        return joints
    
    def rotation_matrix(self, alpha: float, beta: float) -> np.ndarray:
        """计算旋转矩阵 R = Ry(β) * Rx(α)"""
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        
        Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
        Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
        
        return Ry @ Rx
    
    def platform_joints_global(self, q: np.ndarray) -> np.ndarray:
        """计算动平台铰链在全局坐标系中的位置"""
        z, alpha, beta = q
        R = self.rotation_matrix(alpha, beta)
        platform_center = np.array([0, 0, z])
        
        joints_global = np.zeros((3, 3))
        for i in range(3):
            joints_global[i] = platform_center + R @ self.platform_joints_local[i]
        
        return joints_global
    
    def compute_leg_lengths(self, q: np.ndarray) -> np.ndarray:
        """计算支链长度"""
        platform_joints = self.platform_joints_global(q)
        lengths = np.zeros(3)
        
        for i in range(3):
            leg_vec = platform_joints[i] - self.base_joints[i]
            lengths[i] = np.linalg.norm(leg_vec)
        
        return lengths
    
    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """计算雅可比矩阵"""
        z, alpha, beta = q
        platform_joints = self.platform_joints_global(q)
        
        J = np.zeros((3, 3))
        for i in range(3):
            leg_vec = platform_joints[i] - self.base_joints[i]
            leg_length = np.linalg.norm(leg_vec)
            
            if leg_length < 1e-6:
                leg_length = 1e-6
            
            leg_unit = leg_vec / leg_length
            
            J[i, 0] = leg_unit[2]
            
            r_local = self.platform_joints_local[i]
            v_rot_alpha = np.array([0, -r_local[2], r_local[1]])
            v_rot_alpha_global = self.rotation_matrix(alpha, beta) @ v_rot_alpha
            J[i, 1] = np.dot(leg_unit, v_rot_alpha_global)
            
            v_rot_beta = np.array([r_local[2], 0, -r_local[0]])
            v_rot_beta_global = self.rotation_matrix(alpha, beta) @ v_rot_beta
            J[i, 2] = np.dot(leg_unit, v_rot_beta_global)
        
        return J
    
    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """计算质量矩阵 M(q)"""
        alpha = q[1]
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        M = np.zeros((3, 3))
        M[0, 0] = self.m_platform
        M[1, 1] = self.Ixx
        M[2, 2] = self.Iyy * ca**2 + self.Izz * sa**2
        
        return M
    
    def coriolis_matrix(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """计算科里奥利矩阵 C(q, qd)"""
        alpha = q[1]
        dalpha = qd[1]
        dbeta = qd[2]
        
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        common_term = (self.Izz - self.Iyy) * sa * ca
        
        C = np.zeros((3, 3))
        C[1, 2] = -common_term * dbeta
        C[2, 1] = common_term * dbeta
        C[2, 2] = common_term * dalpha
        
        return C
    
    def gravity_vector(self, q: np.ndarray) -> np.ndarray:
        """计算重力向量 G(q)"""
        return np.array([self.m_platform * self.g, 0.0, 0.0])
    
    def friction_vector(self, qd: np.ndarray) -> np.ndarray:
        """计算摩擦力向量 F(qd) - 使用光滑近似避免不连续"""
        F_coulomb = self.params['F_coulomb']
        F_viscous = self.params['F_viscous']
        
        # 使用光滑tanh函数替代sign，避免零速度处的不连续
        # tanh(k*x) 当 k 足够大时，近似 sign(x)，但在 x=0 处光滑
        smoothness_factor = 50.0  # 光滑度因子，越大越接近sign
        F_c = F_coulomb * np.tanh(smoothness_factor * qd)
        
        # 粘性摩擦
        F_v = F_viscous * qd
        
        return F_c + F_v
    
    def forward_dynamics(self, q: np.ndarray, qd: np.ndarray, 
                        tau_legs: np.ndarray) -> np.ndarray:
        """正向动力学"""
        q = np.array(q).flatten()[:3]
        qd = np.array(qd).flatten()[:3]
        tau_legs = np.array(tau_legs).flatten()[:3]
        
        J = self.jacobian(q)
        M = self.mass_matrix(q)
        C = self.coriolis_matrix(q, qd)
        G = self.gravity_vector(q)
        F = self.friction_vector(qd)
        
        tau_task = J.T @ tau_legs
        qdd = np.linalg.solve(M, tau_task - C @ qd - G - F)
        
        return qdd
    
    def inverse_dynamics(self, q: np.ndarray, qd: np.ndarray, 
                        qdd: np.ndarray) -> np.ndarray:
        """逆向动力学"""
        q = np.array(q).flatten()[:3]
        qd = np.array(qd).flatten()[:3]
        qdd = np.array(qdd).flatten()[:3]
        
        J = self.jacobian(q)
        M = self.mass_matrix(q)
        C = self.coriolis_matrix(q, qd)
        G = self.gravity_vector(q)
        F = self.friction_vector(qd)
        
        tau_task = M @ qdd + C @ qd + G + F
        
        try:
            tau_legs = np.linalg.solve(J.T, tau_task)
        except np.linalg.LinAlgError:
            tau_legs = np.linalg.pinv(J.T) @ tau_task
        
        return tau_legs
    
    def energy(self, q: np.ndarray, qd: np.ndarray) -> Tuple[float, float, float]:
        """计算系统能量"""
        M = self.mass_matrix(q)
        kinetic = 0.5 * qd @ M @ qd
        potential = self.m_platform * self.g * q[0]
        total = kinetic + potential
        
        return float(kinetic), float(potential), float(total)
    
    def check_constraints(self, q: np.ndarray, tau: np.ndarray) -> Dict:
        """检查约束满足情况"""
        z, alpha, beta = q
        
        constraints = {
            'z_min_satisfied': z >= self.z_min,
            'z_max_satisfied': z <= self.z_max,
            'alpha_satisfied': abs(alpha) <= self.alpha_max,
            'beta_satisfied': abs(beta) <= self.beta_max,
            'input_satisfied': np.all((tau >= self.u_min) & (tau <= self.u_max)),
            'all_satisfied': True
        }
        
        constraints['all_satisfied'] = (
            constraints['z_min_satisfied'] and
            constraints['z_max_satisfied'] and
            constraints['alpha_satisfied'] and
            constraints['beta_satisfied'] and
            constraints['input_satisfied']
        )
        
        return constraints
    
    def step(self, q: np.ndarray, qd: np.ndarray, tau: np.ndarray, 
             dt: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """仿真单步推进"""
        if dt is None:
            dt = self.dt
        
        state = np.concatenate([q, qd])
        
        def dynamics(t, state):
            q = state[:3]
            qd = state[3:]
            qdd = self.forward_dynamics(q, qd, tau)
            return np.concatenate([qd, qdd])
        
        sol = solve_ivp(dynamics, [0, dt], state, method='RK45', 
                       t_eval=[dt], rtol=1e-6, atol=1e-9)
        
        state_new = sol.y[:, -1]
        return state_new[:3], state_new[3:]


def test_platform():
    """测试平台模型"""
    print("=" * 70)
    print("测试 3-UPS/PU并联船载稳定平台动力学模型")
    print("=" * 70)
    
    platform = ParallelPlatform3DOF()
    
    print(f"\n平台参数：")
    print(f"  质量: {platform.m_platform} kg")
    print(f"  惯量: Ixx={platform.Ixx}, Iyy={platform.Iyy}, Izz={platform.Izz}")
    
    q = np.array([0.529, 0.05, 0.03])
    qd = np.array([0.0, 0.0, 0.0])
    
    leg_lengths = platform.compute_leg_lengths(q)
    print(f"\n支链长度: {leg_lengths}")
    
    tau_hover = platform.inverse_dynamics(q, qd, np.zeros(3))
    print(f"悬停所需力: {tau_hover}")
    
    qdd = platform.forward_dynamics(q, qd, tau_hover)
    print(f"加速度: {qdd}")
    
    print("\n模型测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    test_platform()
