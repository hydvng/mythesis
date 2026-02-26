"""
Chapter 4: V5 + ESO + 硬切换 DRARL

在V5 Simplified基础上增加：
1. ESO: 估计总扰动，作为控制补偿
2. 硬切换: 基于误差阈值的中心区/边界区切换

核心改进：
- V5保持不变：纯ISE奖励 + SAC算法
- ESO: 扰动估计用于控制补偿
- 硬切换: 增强鲁棒性
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'common'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

import numpy as np
import gym
from gym import spaces
from typing import Tuple, Dict

from platform_dynamics import ParallelPlatform3DOF
from wave_disturbance import WaveDisturbance
from eso_controller import MultiDOFESO, HardSwitchController, PerformanceFunction


class PlatformRLEnvChapter4(gym.Env):
    """
    Chapter 4: V5 + ESO + 硬切换 DRARL
    
    在V5 Simplified基础上：
    - 增加ESO扰动估计
    - 增加硬切换机制
    - 保持纯ISE奖励函数
    """
    
    def __init__(self,
                 use_model_compensation: bool = True,
                 use_eso: bool = True,
                 use_hard_switch: bool = True,
                 dt: float = 0.01,
                 max_episode_steps: int = 5000,
                 Hs: float = 2.0,
                 T1: float = 8.0,
                 q_des_type: str = 'sinusoidal',
                 diverge_threshold: float = 0.9,
                 eso_omega: float = 20.0,
                 switch_threshold: float = 0.6,
                 switch_beta: float = 0.5):
        super().__init__()
        
        # 控制选项
        self.use_model_compensation = use_model_compensation
        self.use_eso = use_eso
        self.use_hard_switch = use_hard_switch
        
        # 仿真参数
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.q_des_type = q_des_type
        self.diverge_threshold = diverge_threshold
        
        # 平台和扰动
        self.platform = ParallelPlatform3DOF(dt=dt)
        self.wave = WaveDisturbance(Hs=Hs, T1=T1, random_seed=None)
        
        # 维度
        self.state_dim = 9
        self.action_dim = 3
        
        # 动作空间
        self.action_scale = 5000.0
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )
        
        # 观测空间
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.state_dim,), dtype=np.float32
        )
        
        # 状态归一化
        z_range = self.platform.z_max - self.platform.z_min
        alpha_range = 2 * self.platform.alpha_max
        beta_range = 2 * self.platform.beta_max
        
        self.state_scale = np.array([
            z_range / 2, alpha_range / 2, beta_range / 2,
            1.0, 1.0, 1.0,
            z_range / 2, alpha_range / 2, beta_range / 2
        ])
        self.state_offset = np.array([
            (self.platform.z_max + self.platform.z_min) / 2, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        ])
        
        # ====== 新增: ESO ======
        if self.use_eso:
            self.eso = MultiDOFESO(dt=dt, omega_o=eso_omega)
        
        # ====== 新增: 硬切换 ======
        if self.use_hard_switch:
            self.hsc = HardSwitchController(
                threshold=switch_threshold,
                beta=switch_beta,
                kp=24.0,
                kd=2.5
            )
            self.perf_func = PerformanceFunction(rho_0=2.0, rho_inf=0.05, kappa=0.5)
        
        # 状态变量
        self.q = None
        self.qd = None
        self.q_des = None
        self.qd_des = None
        self.step_count = 0
        self.episode_time = 0.0
        self.prev_u = None
        self.cumulative_error = None
        
        # 历史记录
        self.history = {
            'time': [], 'q': [], 'qd': [], 'q_des': [], 'u': [],
            'v_RL': [], 'tau_model': [], 'reward': [], 'ise': [],
            'eso_disturbance': [], 'region': []  # 新增
        }
    
    def _get_desired_trajectory(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        z_center = 1.0
        
        if self.q_des_type == 'sinusoidal':
            z_des = z_center + 0.1 * np.sin(2 * np.pi * 0.2 * t)
            alpha_des = 0.1 * np.sin(2 * np.pi * 0.15 * t)
            beta_des = 0.08 * np.sin(2 * np.pi * 0.25 * t + np.pi/6)
            
            zd_des = 0.1 * 2 * np.pi * 0.2 * np.cos(2 * np.pi * 0.2 * t)
            alphad_des = 0.1 * 2 * np.pi * 0.15 * np.cos(2 * np.pi * 0.15 * t)
            betad_des = 0.08 * 2 * np.pi * 0.25 * np.cos(2 * np.pi * 0.25 * t + np.pi/6)
            
        elif self.q_des_type == 'sinusoidal_small':
            z_des = z_center + 0.05 * np.sin(2 * np.pi * 0.15 * t)
            alpha_des = 0.05 * np.sin(2 * np.pi * 0.12 * t)
            beta_des = 0.04 * np.sin(2 * np.pi * 0.18 * t + np.pi/6)
            
            zd_des = 0.05 * 2 * np.pi * 0.15 * np.cos(2 * np.pi * 0.15 * t)
            alphad_des = 0.05 * 2 * np.pi * 0.12 * np.cos(2 * np.pi * 0.12 * t)
            betad_des = 0.04 * 2 * np.pi * 0.18 * np.cos(2 * np.pi * 0.18 * t + np.pi/6)
            
        elif self.q_des_type == 'constant':
            z_des = z_center
            alpha_des = 0.0
            beta_des = 0.0
            zd_des = 0.0
            alphad_des = 0.0
            betad_des = 0.0
        else:
            raise ValueError(f"Unknown q_des_type: {self.q_des_type}")
        
        return np.array([z_des, alpha_des, beta_des]), np.array([zd_des, alphad_des, betad_des])
    
    def _compute_model_compensation(self, q, qd, q_des, qd_des):
        if not self.use_model_compensation:
            return np.zeros(3)
        
        Kp = np.array([1000.0, 100.0, 100.0])
        Kd = np.array([100.0, 10.0, 10.0])
        
        e = q_des - q
        ed = qd_des - qd
        qdd_des = Kp * e + Kd * ed
        
        return self.platform.inverse_dynamics(q, qd, qdd_des)
    
    def _normalize_state(self, q, qd, q_des):
        e = q - q_des
        state = np.concatenate([q, qd, e])
        state_normalized = (state - self.state_offset) / self.state_scale
        return np.clip(state_normalized, -1.0, 1.0)
    
    def _denormalize_action(self, action):
        return action * self.action_scale
    
    def _compute_reward_simple(self, e, cumulative_error):
        """纯ISE奖励函数 (与V5相同)"""
        ise = np.sum(e**2)
        reward = -ise
        return reward, ise
    
    def reset(self):
        # 初始状态
        z0 = np.random.uniform(0.9, 1.1)
        alpha0 = np.random.uniform(-0.1, 0.1)
        beta0 = np.random.uniform(-0.1, 0.1)
        
        self.q = np.array([z0, alpha0, beta0])
        self.qd = np.zeros(3)
        self.step_count = 0
        self.episode_time = 0.0
        self.prev_u = None
        self.cumulative_error = np.zeros(3)
        
        self.q_des, self.qd_des = self._get_desired_trajectory(0)
        
        self.wave = WaveDisturbance(Hs=self.wave.Hs, T1=self.wave.T1, random_seed=None)
        
        # ====== 重置ESO ======
        if self.use_eso:
            self.eso.reset()
        
        # ====== 重置硬切换 ======
        if self.use_hard_switch:
            self.hsc.current_region = 'center'
            self.perf_func.reset()
        
        self.history = {
            'time': [], 'q': [], 'qd': [], 'q_des': [], 'u': [],
            'v_RL': [], 'tau_model': [], 'reward': [], 'ise': [],
            'eso_disturbance': [], 'region': [],
            'actual_disturbance': []  # 实际总扰动
        }
        
        return self._normalize_state(self.q, self.qd, self.q_des)
    
    def step(self, action):
        v_RL = self._denormalize_action(action)
        
        self.q_des, self.qd_des = self._get_desired_trajectory(self.episode_time)
        
        # 计算误差
        e = self.q - self.q_des
        ed = self.qd - self.qd_des
        
        # ====== ESO: 更新扰动估计 ======
        if self.use_eso:
            # 使用ESO估计扰动
            _, d_est = self.eso.update(self.q, v_RL)
        else:
            d_est = np.zeros(3)
        
        # ====== 硬切换控制 ======
        if self.use_hard_switch:
            # 更新性能函数
            self.perf_func.update(self.dt)
            rho = self.perf_func.get_rho()
            
            # 计算硬切换控制
            u_comp = self._compute_model_compensation(
                self.q, self.qd, self.q_des, self.qd_des
            )
            
            # 硬切换: 使用误差和ESO估计
            u_legs = self.hsc.compute_control(e, ed, v_RL, u_comp)
            
            region = self.hsc.get_region()
        else:
            # 使用V5的控制方式
            tau_model = self._compute_model_compensation(
                self.q, self.qd, self.q_des, self.qd_des
            )
            u_legs = tau_model + v_RL
            region = 'center'
        
        # 生成扰动
        disturbance = self.wave.generate_disturbance(
            np.array([self.episode_time])
        )[0]
        
        # 保存更新前的速度，用于计算实际扰动
        qd_before = self.qd.copy()
        qdd = self.platform.forward_dynamics(self.q, self.qd, u_legs)
        M = self.platform.mass_matrix(self.q)
        M_inv = np.linalg.inv(M)
        qdd_disturbed = qdd + M_inv @ disturbance
        self.qd = self.qd + qdd_disturbed * self.dt
        self.q = self.q + self.qd * self.dt
        # d_actual = M * qdd_actual + C * qd + G - u
        qdd_actual = (self.qd - qd_before) / self.dt  # 实际加速度
        C = self.platform.coriolis_matrix(self.q, self.qd)
        G = self.platform.gravity_vector(self.q)
        d_actual = M @ qdd_actual + C @ qd_before + G - u_legs  # 实际总扰动
        
        # 更新累积误差
        e = self.q - self.q_des
        self.cumulative_error += e * self.dt
        
        # 奖励计算 (纯ISE，与V5相同)
        reward, ise = self._compute_reward_simple(e, self.cumulative_error)
        
        self.prev_u = u_legs.copy()
        
        self.step_count += 1
        self.episode_time += self.dt
        
        # 终止条件
        done = False
        
        if self.step_count >= self.max_episode_steps:
            done = True
        
        error_norm = np.linalg.norm(e)
        if error_norm > self.diverge_threshold:
            done = True
            reward -= 100.0
        
        # 历史记录
        self.history['time'].append(self.episode_time)
        self.history['q'].append(self.q.copy())
        self.history['qd'].append(self.qd.copy())
        self.history['q_des'].append(self.q_des.copy())
        self.history['u'].append(u_legs.copy())
        self.history['v_RL'].append(v_RL.copy())
        self.history['reward'].append(reward)
        self.history['ise'].append(ise)
        self.history['eso_disturbance'].append(d_est.copy())
        self.history['actual_disturbance'].append(d_actual.copy())
        self.history['region'].append(region)
        self.history['region'].append(region)
        
        info = {
            'error_norm': error_norm,
            'ise': ise,
            'cumulative_ise': np.sum(self.cumulative_error**2),
            'region': region,
            'disturbance_est': np.linalg.norm(d_est) if self.use_eso else 0.0
        }
        
        return self._normalize_state(self.q, self.qd, self.q_des), reward, done, info
    
    def get_history(self):
        return self.history
    
    def compute_ise(self):
        """计算整个episode的ISE"""
        if not self.history['ise']:
            return 0.0
        return sum(self.history['ise']) * self.dt


if __name__ == '__main__':
    # 测试Chapter4环境
    print("Testing Chapter 4 Environment...")
    env = PlatformRLEnvChapter4(
        use_model_compensation=True,
        use_eso=True,
        use_hard_switch=True,
        max_episode_steps=100,
        Hs=2.0,
        T1=8.0,
        q_des_type='sinusoidal',
        diverge_threshold=0.5,
        eso_omega=20.0,
        switch_threshold=0.7,
        switch_beta=0.5
    )
    
    print("Environment Info:")
    print(f"  State dim: {env.state_dim}")
    print(f"  Action dim: {env.action_dim}")
    print(f"  Use ESO: {env.use_eso}")
    print(f"  Use Hard Switch: {env.use_hard_switch}")
    
    state = env.reset()
    
    total_reward = 0
    for i in range(100):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        total_reward += reward
        
        if i % 20 == 0:
            print(f"Step {i}: reward={reward:.4f}, ise={info['ise']:.6f}, "
                  f"region={info['region']}, d_est={info['disturbance_est']:.4f}")
        
        if done:
            break
    
    print(f"\nTotal reward: {total_reward:.2f}")
    print(f"Final ISE: {env.compute_ise():.6f}")
    print(f"\n✓ Chapter 4 Environment test passed!")
