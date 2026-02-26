"""
V5 Simplified: 简化奖励函数的版本
改进点：
1. 移除收敛奖励（会导致"假装"收敛）
2. 移除平滑度惩罚
3. 移除时间权重
4. 简化为纯ISE目标
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'common'))

import numpy as np
import gym
from gym import spaces
from typing import Tuple, Dict

from platform_dynamics import ParallelPlatform3DOF
from wave_disturbance import WaveDisturbance


class PlatformRLEnvV5Simplified(gym.Env):
    """V5简化版 - 简化的奖励函数"""
    
    def __init__(self,
                 use_model_compensation: bool = True,
                 dt: float = 0.01,
                 max_episode_steps: int = 5000,
                 Hs: float = 2.0,
                 T1: float = 8.0,
                 q_des_type: str = 'sinusoidal',
                 diverge_threshold: float = 0.5,
                 warning_penalty: float = 0.0):  # 默认0.0
        super().__init__()
        
        self.use_model_compensation = use_model_compensation
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.q_des_type = q_des_type
        self.diverge_threshold = diverge_threshold
        self.warning_penalty = warning_penalty
        
        self.platform = ParallelPlatform3DOF(dt=dt)
        self.wave = WaveDisturbance(Hs=Hs, T1=T1, random_seed=None)
        
        self.state_dim = 9
        self.action_dim = 3
        
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.state_dim,), dtype=np.float32
        )
        
        self.action_scale = 5000.0
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )
        
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
        
        self.q = None
        self.qd = None
        self.q_des = None
        self.qd_des = None
        self.step_count = 0
        self.episode_time = 0.0
        self.prev_u = None
        self.cumulative_error = None
        
        self.history = {
            'time': [], 'q': [], 'qd': [], 'q_des': [], 'u': [],
            'v_RL': [], 'tau_model': [], 'reward': [], 'ise': []
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
        """
        简化的奖励函数 - 只有ISE目标
        reward = -ISE (积分平方误差)
        
        ISE = integral of (error^2) dt
        离散形式: sum(error^2 * dt)
        
        目标: 最小化ISE
        """
        # ISE作为负奖励（最大化奖励 = 最小化ISE）
        # 使用原始误差，不做任何缩放
        ise = np.sum(e**2)
        
        # 负ISE作为奖励（越大越好 -> 误差越小）
        reward = -ise
        
        return reward, ise
    
    def reset(self):
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
        
        self.history = {
            'time': [], 'q': [], 'qd': [], 'q_des': [], 'u': [],
            'v_RL': [], 'tau_model': [], 'reward': [], 'ise': []
        }
        
        return self._normalize_state(self.q, self.qd, self.q_des)
    
    def step(self, action):
        v_RL = self._denormalize_action(action)
        
        self.q_des, self.qd_des = self._get_desired_trajectory(self.episode_time)
        
        tau_model = self._compute_model_compensation(
            self.q, self.qd, self.q_des, self.qd_des
        )
        
        u_legs = tau_model + v_RL
        
        # 不限制输入，信任模型补偿
        disturbance = self.wave.generate_disturbance(
            np.array([self.episode_time])
        )[0]
        
        qdd = self.platform.forward_dynamics(self.q, self.qd, u_legs)
        
        M = self.platform.mass_matrix(self.q)
        M_inv = np.linalg.inv(M)
        qdd_disturbed = qdd + M_inv @ disturbance
        
        self.qd = self.qd + qdd_disturbed * self.dt
        self.q = self.q + self.qd * self.dt
        
        # 更新累积误差
        e = self.q - self.q_des
        self.cumulative_error += e * self.dt
        
        # 简化的奖励计算
        reward, ise = self._compute_reward_simple(e, self.cumulative_error)
        
        self.prev_u = u_legs.copy()
        
        self.step_count += 1
        self.episode_time += self.dt
        
        # 终止条件
        done = False
        
        # 1. 达到最大步数
        if self.step_count >= self.max_episode_steps:
            done = True
        
        # 2. 严重偏离才终止
        error_norm = np.linalg.norm(e)
        if error_norm > self.diverge_threshold:
            done = True
            reward -= 100.0  # 发散惩罚
        
        # 3. 预警机制（可选）
        if error_norm > 0.4 and self.warning_penalty != 0.0:
            reward -= self.warning_penalty
        
        self.history['time'].append(self.episode_time)
        self.history['q'].append(self.q.copy())
        self.history['qd'].append(self.qd.copy())
        self.history['q_des'].append(self.q_des.copy())
        self.history['u'].append(u_legs.copy())
        self.history['v_RL'].append(v_RL.copy())
        self.history['tau_model'].append(tau_model.copy())
        self.history['reward'].append(reward)
        self.history['ise'].append(ise)
        
        info = {
            'error_norm': error_norm,
            'ise': ise,
            'cumulative_ise': np.sum(self.cumulative_error**2)
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
    env = PlatformRLEnvV5Simplified(
        use_model_compensation=True,
        max_episode_steps=2000,
        Hs=2.0,
        T1=8.0,
        q_des_type='sinusoidal',
        diverge_threshold=0.5,
        warning_penalty=0.0
    )
    
    print("V5 Simplified 环境测试")
    state = env.reset()
    
    total_reward = 0
    for i in range(100):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        total_reward += reward
        
        if i % 20 == 0:
            print(f"Step {i}: reward={reward:.4f}, ise={info['ise']:.6f}, error={info['error_norm']:.4f}")
        
        if done:
            break
    
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final ISE: {env.compute_ise():.6f}")
