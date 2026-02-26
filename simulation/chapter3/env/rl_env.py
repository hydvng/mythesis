"""
第3章：强化学习环境封装
用于SAC算法的训练环境
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


class PlatformRLEnv(gym.Env):
    """
    3-UPS/PU平台强化学习环境
    
    状态空间 (9维):
    - q = [z, alpha, beta] (3维): 当前姿态
    - qd = [z_dot, alpha_dot, beta_dot] (3维): 当前速度  
    - e = q - q_des (3维): 跟踪误差
    
    动作空间 (3维):
    - v_RL = [v_z, v_alpha, v_beta]: RL补偿量
    
    控制律:
    - u = tau_model + v_RL
    - tau_model: 模型补偿项 (可选)
    """
    
    def __init__(self,
                 use_model_compensation: bool = True,
                 dt: float = 0.01,
                 max_episode_steps: int = 1000,
                 Hs: float = 2.0,
                 T1: float = 8.0,
                 q_des_type: str = 'sinusoidal',
                 constraint_penalty: float = 10.0,
                 soft_constraint: bool = True):
        """
        初始化环境
        
        Args:
            use_model_compensation: 是否使用模型补偿
            dt: 仿真时间步长
            max_episode_steps: 每回合最大步数
            Hs: 海浪有效波高
            T1: 海浪平均周期
            q_des_type: 期望轨迹类型 ('sinusoidal', 'constant', 'random')
            constraint_penalty: 约束违反惩罚系数
            soft_constraint: 是否使用软约束（奖励惩罚）
        """
        super().__init__()
        
        self.use_model_compensation = use_model_compensation
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.constraint_penalty = constraint_penalty
        self.soft_constraint = soft_constraint
        self.q_des_type = q_des_type
        
        # 平台模型
        self.platform = ParallelPlatform3DOF(dt=dt)
        
        # 海浪扰动
        self.wave = WaveDisturbance(Hs=Hs, T1=T1, random_seed=None)
        
        # 状态空间: [z, alpha, beta, z_dot, alpha_dot, beta_dot, e_z, e_alpha, e_beta]
        # 归一化到[-1, 1]
        z_range = self.platform.z_max - self.platform.z_min
        alpha_range = 2 * self.platform.alpha_max
        beta_range = 2 * self.platform.beta_max
        v_max = self.platform.v_max
        
        self.state_dim = 9
        self.action_dim = 3
        
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.state_dim,), dtype=np.float32
        )
        
        # 动作空间: v_RL (RL补偿量)
        # SAC输出经过tanh后在[-1, 1]，需要缩放到实际控制量范围
        # RL补偿量范围设为±5000N，因为模型补偿已承担大部分工作
        # 这是一个合理的补偿量范围，足够处理残余动态和扰动
        self.action_scale = 5000.0  # RL补偿量最大幅度 (N)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )
        
        # 状态归一化因子
        # 使用更大的范围确保状态在扰动下也能归一化到[-1, 1]
        # 位置范围
        self.state_scale = np.array([
            z_range / 2, alpha_range / 2, beta_range / 2,  # 位置
            1.0, 1.0, 1.0,  # 速度 (rad/s 或 m/s，使用1.0作为基准，实际速度会更小)
            z_range / 2, alpha_range / 2, beta_range / 2   # 误差
        ])
        self.state_offset = np.array([
            (self.platform.z_max + self.platform.z_min) / 2, 0.0, 0.0,  # 位置
            0.0, 0.0, 0.0,  # 速度
            0.0, 0.0, 0.0   # 误差
        ])
        
        # 内部状态
        self.q = None
        self.qd = None
        self.q_des = None
        self.qd_des = None
        self.step_count = 0
        self.episode_time = 0.0
        
        # 历史记录（用于绘图）
        self.history = {
            'time': [],
            'q': [],
            'qd': [],
            'q_des': [],
            'u': [],
            'v_RL': [],
            'tau_model': [],
            'reward': [],
            'constraint_violation': []
        }
        
    def _get_desired_trajectory(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """生成期望轨迹"""
        # 期望z值修正：使用1.0作为基准，在约束范围内[0.808, 1.308]
        z_center = 1.0  # 约束范围中心附近
        
        if self.q_des_type == 'sinusoidal':
            # 正弦轨迹（振幅控制在约束范围内）
            z_des = z_center + 0.1 * np.sin(2 * np.pi * 0.2 * t)
            alpha_des = 0.1 * np.sin(2 * np.pi * 0.15 * t)
            beta_des = 0.08 * np.sin(2 * np.pi * 0.25 * t + np.pi/6)
            
            zd_des = 0.1 * 2 * np.pi * 0.2 * np.cos(2 * np.pi * 0.2 * t)
            alphad_des = 0.1 * 2 * np.pi * 0.15 * np.cos(2 * np.pi * 0.15 * t)
            betad_des = 0.08 * 2 * np.pi * 0.25 * np.cos(2 * np.pi * 0.25 * t + np.pi/6)
            
        elif self.q_des_type == 'constant':
            # 恒定姿态
            z_des = z_center
            alpha_des = 0.0
            beta_des = 0.0
            zd_des = 0.0
            alphad_des = 0.0
            betad_des = 0.0
            
        elif self.q_des_type == 'random':
            # 随机轨迹（每回合生成）
            if not hasattr(self, '_random_trajectory'):
                self._generate_random_trajectory()
            z_des, alpha_des, beta_des = self._random_trajectory['q'](t)
            zd_des, alphad_des, betad_des = self._random_trajectory['qd'](t)
        else:
            raise ValueError(f"Unknown q_des_type: {self.q_des_type}")
        
        q_des = np.array([z_des, alpha_des, beta_des])
        qd_des = np.array([zd_des, alphad_des, betad_des])
        
        return q_des, qd_des
    
    def _generate_random_trajectory(self):
        """生成随机轨迹（多个正弦叠加）"""
        np.random.seed()
        n_freq = 3
        freqs = np.random.uniform(0.1, 0.5, n_freq)
        amps_z = np.random.uniform(0.02, 0.08, n_freq)
        amps_alpha = np.random.uniform(0.02, 0.1, n_freq)
        amps_beta = np.random.uniform(0.02, 0.08, n_freq)
        phases = np.random.uniform(0, 2*np.pi, n_freq)
        
        def q_func(t):
            z = 1.0 + np.sum(amps_z * np.sin(2 * np.pi * freqs * t + phases))  # 修正z基准
            alpha = np.sum(amps_alpha * np.sin(2 * np.pi * freqs * t + phases + np.pi/3))
            beta = np.sum(amps_beta * np.sin(2 * np.pi * freqs * t + phases + 2*np.pi/3))
            return z, alpha, beta
        
        def qd_func(t):
            zd = np.sum(amps_z * 2 * np.pi * freqs * np.cos(2 * np.pi * freqs * t + phases))
            alphad = np.sum(amps_alpha * 2 * np.pi * freqs * np.cos(2 * np.pi * freqs * t + phases + np.pi/3))
            betad = np.sum(amps_beta * 2 * np.pi * freqs * np.cos(2 * np.pi * freqs * t + phases + 2*np.pi/3))
            return zd, alphad, betad
        
        self._random_trajectory = {'q': q_func, 'qd': qd_func}
    
    def _compute_model_compensation(self, q: np.ndarray, qd: np.ndarray,
                                    q_des: np.ndarray, qd_des: np.ndarray) -> np.ndarray:
        """计算模型补偿项 tau_model"""
        if not self.use_model_compensation:
            return np.zeros(3)
        
        # 计算期望加速度（简单PD控制）
        Kp = np.array([1000.0, 100.0, 100.0])
        Kd = np.array([100.0, 10.0, 10.0])
        
        e = q_des - q
        ed = qd_des - qd
        
        qdd_des = Kp * e + Kd * ed
        
        # 逆动力学计算所需力
        tau_model = self.platform.inverse_dynamics(q, qd, qdd_des)
        
        return tau_model
    
    def _normalize_state(self, q: np.ndarray, qd: np.ndarray,
                        q_des: np.ndarray) -> np.ndarray:
        """归一化状态到[-1, 1]"""
        e = q - q_des
        state = np.concatenate([q, qd, e])
        state_normalized = (state - self.state_offset) / self.state_scale
        return np.clip(state_normalized, -1.0, 1.0)
    
    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """将归一化动作转换回实际控制量"""
        # SAC输出在[-1, 1]，缩放到实际补偿量范围
        return action * self.action_scale
    
    def _compute_reward(self, q: np.ndarray, qd: np.ndarray,
                        q_des: np.ndarray, qd_des: np.ndarray,
                        u: np.ndarray, constraint_info: Dict,
                        time_weight: float = 1.0) -> float:
        """计算奖励 - 支持时序加权
        
        Args:
            time_weight: 时间权重，越往后权重越大，鼓励收敛
        """
        e = q - q_des
        ed = qd - qd_des
        
        # 归一化位置误差（时序加权）
        # 误差单位: z~0.01m, α/β~0.01rad
        pos_weights = np.array([1000.0, 500.0, 500.0])
        r_pos = -time_weight * np.sum(pos_weights * e**2) / 10000.0
        
        # 归一化速度误差（时序加权）
        vel_weights = np.array([100.0, 50.0, 50.0])
        r_vel = -time_weight * np.sum(vel_weights * ed**2) / 10000.0
        
        # 控制量惩罚
        r_control = -0.0001 * np.sum((u / 10000.0)**2)
        
        reward = r_pos + r_vel + r_control
        
        # 软约束处理
        if self.soft_constraint:
            constraint_penalty = 0.0
            
            # 位置约束违反惩罚
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
    
    def _compute_temporal_weight(self, step: int) -> float:
        """计算时序权重 - 随时间增加"""
        # 线性增加: 从1.0开始，到2.0结束
        progress = step / self.max_episode_steps if self.max_episode_steps > 0 else 0
        return 1.0 + progress  # 范围: [1.0, 2.0]
    
    def _compute_convergence_reward(self, e: np.ndarray, step: int) -> float:
        """计算收敛奖励 - 如果最后阶段误差很小，给予奖励"""
        # 只有在最后20%回合才给予收敛奖励
        if step < 0.8 * self.max_episode_steps:
            return 0.0
        
        # 误差阈值
        z_threshold = 0.01  # 1cm
        alpha_threshold = 0.02  # ~1.15度
        beta_threshold = 0.02
        
        # 检查是否收敛
        converged = (abs(e[0]) < z_threshold and 
                    abs(e[1]) < alpha_threshold and 
                    abs(e[2]) < beta_threshold)
        
        if converged:
            return 2.0  # 收敛奖励
        return 0.0
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        # 随机初始状态（在安全区域内）
        # z范围修正：z_min=0.808, z_max=1.308，使用0.9-1.1作为安全初始范围
        z0 = np.random.uniform(0.9, 1.1)
        alpha0 = np.random.uniform(-0.1, 0.1)
        beta0 = np.random.uniform(-0.1, 0.1)
        
        self.q = np.array([z0, alpha0, beta0])
        self.qd = np.zeros(3)
        self.step_count = 0
        self.episode_time = 0.0
        
        # 重置随机轨迹
        if self.q_des_type == 'random':
            if hasattr(self, '_random_trajectory'):
                delattr(self, '_random_trajectory')
        
        # 重置海浪扰动
        self.wave = WaveDisturbance(Hs=self.wave.Hs, T1=self.wave.T1, random_seed=None)
        
        # 清空历史
        self.history = {
            'time': [],
            'q': [],
            'qd': [],
            'q_des': [],
            'u': [],
            'v_RL': [],
            'tau_model': [],
            'reward': [],
            'constraint_violation': []
        }
        
        # 获取期望轨迹
        self.q_des, self.qd_des = self._get_desired_trajectory(0.0)
        
        return self._normalize_state(self.q, self.qd, self.q_des)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """环境单步推进"""
        action = np.array(action).flatten()
        
        # 反归一化动作（RL补偿量）
        v_RL = self._denormalize_action(action)
        
        # 获取当前期望轨迹
        self.q_des, self.qd_des = self._get_desired_trajectory(self.episode_time)
        
        # 计算模型补偿
        tau_model = self._compute_model_compensation(
            self.q, self.qd, self.q_des, self.qd_des
        )
        
        # 总控制量
        u_legs = tau_model + v_RL
        
        # 如果不用软约束，进行硬裁剪
        if not self.soft_constraint:
            u_legs = np.clip(u_legs, self.platform.u_min, self.platform.u_max)
        
        # 施加扰动
        disturbance = self.wave.generate_disturbance(
            np.array([self.episode_time])
        )[0]  # [F_z, M_alpha, M_beta]
        
        # 动力学推进
        qdd = self.platform.forward_dynamics(self.q, self.qd, u_legs)
        
        # 添加扰动（简化处理：直接加在加速度层）
        M = self.platform.mass_matrix(self.q)
        M_inv = np.linalg.inv(M)
        qdd_disturbed = qdd + M_inv @ disturbance
        
        # 数值积分
        self.qd = self.qd + qdd_disturbed * self.dt
        self.q = self.q + self.qd * self.dt
        
        # 检查约束
        constraint_info = self.platform.check_constraints(self.q, u_legs)
        
        # 计算时序权重（越往后权重越大）
        time_weight = self._compute_temporal_weight(self.step_count)
        
        # 计算基础奖励（时序加权）
        base_reward = self._compute_reward(
            self.q, self.qd, self.q_des, self.qd_des, u_legs, constraint_info,
            time_weight=time_weight
        )
        
        # 计算收敛奖励
        e = self.q - self.q_des
        convergence_reward = self._compute_convergence_reward(e, self.step_count)
        
        # 总奖励
        reward = base_reward + convergence_reward
        
        # 更新时间
        self.step_count += 1
        self.episode_time += self.dt
        
        # 记录历史
        self.history['time'].append(self.episode_time)
        self.history['q'].append(self.q.copy())
        self.history['qd'].append(self.qd.copy())
        self.history['q_des'].append(self.q_des.copy())
        self.history['u'].append(u_legs.copy())
        self.history['v_RL'].append(v_RL.copy())
        self.history['tau_model'].append(tau_model.copy())
        self.history['reward'].append(reward)
        self.history['constraint_violation'].append(not constraint_info['all_satisfied'])
        
        # 判断是否结束
        done = False
        if self.step_count >= self.max_episode_steps:
            done = True
        
        # 终止条件：严重偏离或约束违反
        if np.linalg.norm(self.q - self.q_des) > 0.5:
            done = True
            reward -= 100.0  # 严重惩罚
        
        info = {
            'constraint_info': constraint_info,
            'tau_model': tau_model,
            'v_RL': v_RL,
            'u_total': u_legs
        }
        
        next_state = self._normalize_state(self.q, self.qd, self.q_des)
        
        return next_state, reward, done, info
    
    def render(self, mode='human'):
        """渲染（可选）"""
        pass
    
    def get_history(self) -> Dict:
        """获取历史数据（用于绘图分析）"""
        return self.history
    
    def close(self):
        """关闭环境"""
        pass


if __name__ == "__main__":
    # 测试环境
    print("Testing RL Environment...")
    
    # 测试1：不使用模型补偿
    env_no_model = PlatformRLEnv(use_model_compensation=False, q_des_type='sinusoidal')
    state = env_no_model.reset()
    print(f"State shape: {state.shape}")
    print(f"State range: [{state.min():.3f}, {state.max():.3f}]")
    
    total_reward = 0
    for _ in range(100):
        action = env_no_model.action_space.sample()
        state, reward, done, info = env_no_model.step(action)
        total_reward += reward
        if done:
            break
    
    print(f"Test episode reward: {total_reward:.2f}")
    
    # 测试2：使用模型补偿
    env_with_model = PlatformRLEnv(use_model_compensation=True, q_des_type='sinusoidal')
    state = env_with_model.reset()
    total_reward = 0
    for _ in range(100):
        action = env_with_model.action_space.sample()
        state, reward, done, info = env_with_model.step(action)
        total_reward += reward
        if done:
            break
    
    print(f"Test episode reward (with model): {total_reward:.2f}")
    print("Environment test passed!")
