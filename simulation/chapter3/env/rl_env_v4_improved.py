"""
V4 Improved: 改进终止条件和收敛判定
改进点：
1. 放宽终止阈值：从0.5改为1.0
2. 软约束处理：用奖励惩罚代替强制终止
3. 增加预警机制：检测发散趋势
4. 支持可配置的预警惩罚
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'common'))

import numpy as np
import gym
from gym import spaces
from typing import Tuple, Dict, Optional

from platform_dynamics import ParallelPlatform3DOF
from wave_disturbance import WaveDisturbance


class PlatformRLEnvV4Improved(gym.Env):
    """V4改进版环境 - 更宽松的终止条件"""
    
    def __init__(self,
                 use_model_compensation: bool = True,
                 dt: float = 0.01,
                 max_episode_steps: int = 5000,
                 Hs: float = 2.0,
                 T1: float = 8.0,
                 q_des_type: str = 'sinusoidal',
                 constraint_penalty: float = 10.0,
                 soft_constraint: bool = True,
                 diverge_threshold: float = 0.5,
                 warning_threshold: float = 0.4,
                 warning_penalty: float = 5.0):  # 新增：可配置预警惩罚
        super().__init__()
        
        self.use_model_compensation = use_model_compensation
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.constraint_penalty = constraint_penalty
        self.soft_constraint = soft_constraint
        self.q_des_type = q_des_type
        self.diverge_threshold = diverge_threshold
        self.warning_threshold = warning_threshold
        self.warning_penalty = warning_penalty  # 保存预警惩罚值
        
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
        self.max_error = None
        
        self.history = {
            'time': [], 'q': [], 'qd': [], 'q_des': [], 'u': [],
            'v_RL': [], 'tau_model': [], 'reward': [], 'constraint_violation': []
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
    
    def _compute_temporal_weight_v4(self, step):
        progress = step / self.max_episode_steps if self.max_episode_steps > 0 else 0
        return 1.0 + 1.0 * progress
    
    def _compute_reward_v4(self, q, qd, q_des, qd_des, u, constraint_info, time_weight, cumulative_error):
        e = q - q_des
        ed = qd - qd_des
        
        pos_weights = np.array([2000.0, 800.0, 800.0])
        r_pos = -time_weight * np.sum(pos_weights * e**2) / 10000.0
        
        vel_weights = np.array([100.0, 50.0, 50.0])
        r_vel = -time_weight * np.sum(vel_weights * ed**2) / 10000.0
        
        r_control = -0.0001 * np.sum((u / 10000.0)**2)
        
        reward = r_pos + r_vel + r_control
        
        # ISE惩罚
        if cumulative_error is not None:
            ise_weights = np.array([5000.0, 2000.0, 2000.0])
            ise_penalty = -np.sum(ise_weights * cumulative_error**2) / 100000.0
            reward += ise_penalty
        
        # 软约束处理
        if self.soft_constraint:
            constraint_penalty = 0.0
            
            if q[0] < self.platform.z_min:
                constraint_penalty += 0.5 * (self.platform.z_min - q[0])**2
            if q[0] > self.platform.z_max:
                constraint_penalty += 0.5 * (q[0] - self.platform.z_max)**2
            if abs(q[1]) > self.platform.alpha_max:
                constraint_penalty += 0.5 * (abs(q[1]) - self.platform.alpha_max)**2
            if abs(q[2]) > self.platform.beta_max:
                constraint_penalty += 0.5 * (abs(q[2]) - self.platform.beta_max)**2
            
            reward -= constraint_penalty
        
        return float(np.clip(reward, -10.0, 0.0))
    
    def _compute_smoothness_reward(self, u, prev_u):
        if prev_u is None:
            return 0.0
        
        delta_u = u - prev_u
        smoothness_penalty = 0.001 * np.sum(delta_u**2) / 10000.0
        return -smoothness_penalty

    def _compute_multi_stage_convergence_reward(self, e, step):
        """V4多阶段收敛奖励"""
        progress = step / self.max_episode_steps if self.max_episode_steps > 0 else 0
        reward = 0.0
        
        # 第一阶段阈值 (较宽松)
        z_threshold_1 = 0.005  # 5mm
        angle_threshold_1 = 0.0175  # 1度
        
        # 第二阶段阈值 (更严格)
        z_threshold_2 = 0.0025  # 2.5mm
        angle_threshold_2 = 0.0087  # 0.5度
        
        # 最后20%回合 - 第一阶段收敛奖励
        if progress >= 0.8:
            converged_1 = (abs(e[0]) < z_threshold_1 and 
                          abs(e[1]) < angle_threshold_1 and 
                          abs(e[2]) < angle_threshold_1)
            if converged_1:
                reward += 5.0
        
        # 最后10%回合 - 第二阶段更严格奖励
        if progress >= 0.9:
            converged_2 = (abs(e[0]) < z_threshold_2 and 
                          abs(e[1]) < angle_threshold_2 and 
                          abs(e[2]) < angle_threshold_2)
            if converged_2:
                reward += 10.0
        
        return reward
    
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
        self.max_error = 0.0
        
        self.q_des, self.qd_des = self._get_desired_trajectory(0)
        
        self.wave = WaveDisturbance(Hs=self.wave.Hs, T1=self.wave.T1, random_seed=None)
        
        self.history = {
            'time': [], 'q': [], 'qd': [], 'q_des': [], 'u': [],
            'v_RL': [], 'tau_model': [], 'reward': [], 'constraint_violation': []
        }
        
        return self._normalize_state(self.q, self.qd, self.q_des)
    
    def step(self, action):
        v_RL = self._denormalize_action(action)
        
        self.q_des, self.qd_des = self._get_desired_trajectory(self.episode_time)
        
        tau_model = self._compute_model_compensation(
            self.q, self.qd, self.q_des, self.qd_des
        )
        
        u_legs = tau_model + v_RL
        
        if not self.soft_constraint:
            u_legs = np.clip(u_legs, self.platform.u_min, self.platform.u_max)
        
        disturbance = self.wave.generate_disturbance(
            np.array([self.episode_time])
        )[0]
        
        qdd = self.platform.forward_dynamics(self.q, self.qd, u_legs)
        
        M = self.platform.mass_matrix(self.q)
        M_inv = np.linalg.inv(M)
        qdd_disturbed = qdd + M_inv @ disturbance
        
        self.qd = self.qd + qdd_disturbed * self.dt
        self.q = self.q + self.qd * self.dt
        
        constraint_info = self.platform.check_constraints(self.q, u_legs)
        
        time_weight = self._compute_temporal_weight_v4(self.step_count)
        
        # 更新累积误差和最大误差
        e = self.q - self.q_des
        self.cumulative_error += e * self.dt
        current_error = np.linalg.norm(e)
        self.max_error = max(self.max_error, current_error)
        
        reward = self._compute_reward_v4(
            self.q, self.qd, self.q_des, self.qd_des,
            u_legs, constraint_info, time_weight, self.cumulative_error
        )
        
        smoothness_reward = self._compute_smoothness_reward(u_legs, self.prev_u)
        reward += smoothness_reward
        # 收敛奖励
        convergence_reward = self._compute_multi_stage_convergence_reward(e, self.step_count)
        reward += convergence_reward
        self.prev_u = u_legs.copy()
        
        self.step_count += 1
        self.episode_time += self.dt
        
        # 改进的终止条件
        done = False
        
        # 1. 达到最大步数
        if self.step_count >= self.max_episode_steps:
            done = True
        
        # 2. 严重偏离才终止
        error_norm = np.linalg.norm(self.q - self.q_des)
        if error_norm > self.diverge_threshold:
            done = True
            reward -= 100.0
        
        # 3. 预警机制：使用可配置的惩罚值
        if error_norm > self.warning_threshold and self.max_error > self.warning_threshold:
            reward -= self.warning_penalty  # 使用配置的预警惩罚
        
        self.history['time'].append(self.episode_time)
        self.history['q'].append(self.q.copy())
        self.history['qd'].append(self.qd.copy())
        self.history['q_des'].append(self.q_des.copy())
        self.history['u'].append(u_legs.copy())
        self.history['v_RL'].append(v_RL.copy())
        self.history['tau_model'].append(tau_model.copy())
        self.history['reward'].append(reward)
        self.history['constraint_violation'].append(not constraint_info['all_satisfied'])
        
        info = {
            'constraint_info': constraint_info,
            'tau_model': tau_model,
            'v_RL': v_RL,
            'error_norm': error_norm,
            'max_error': self.max_error
        }
        
        return self._normalize_state(self.q, self.qd, self.q_des), reward, done, info
    
    def get_history(self):
        return self.history


if __name__ == '__main__':
    env = PlatformRLEnvV4Improved(
        use_model_compensation=True,
        max_episode_steps=2000,
        Hs=2.0,
        T1=8.0,
        q_des_type='sinusoidal',
        diverge_threshold=0.5,
        warning_threshold=0.4,
        warning_penalty=5.0  # 可以设置为0.0来禁用预警惩罚
    )
    
    print("V4 Improved 环境测试（支持可配置预警惩罚）")
    state = env.reset()
    
    for i in range(100):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        
        if i % 20 == 0:
            print(f"Step {i}: reward={reward:.4f}, error={info['error_norm']:.4f}")
        
        if done:
            print(f"Episode done at step {i}, max_error={info['max_error']:.4f}")
            break
    
    print("测试完成!")
