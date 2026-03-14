"""
第3章：基于 UniformRodPlatform3DOF 的强化学习环境封装

说明：
- 本文件是 `rl_env.py` 的并行实现，用于尝试将第2章中更完整的
  `UniformRodPlatform3DOF` 动力学模型接入第3章训练。
- 原始 `rl_env.py` 保留不动，作为 baseline / 参考实现。
"""

import sys
import os

THIS_DIR = os.path.dirname(__file__)
SIM_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
COMMON_DIR = os.path.join(SIM_DIR, 'common')
DISTURBANCE_DIR = os.path.join(SIM_DIR, 'disturbance')

for p in [COMMON_DIR, DISTURBANCE_DIR, SIM_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import gym
from gym import spaces
from typing import Tuple, Dict

try:
    from uniform_rod_platform_dynamics import UniformRodPlatform3DOF
    from wave_disturbance import WaveDisturbance
except ImportError:  # pragma: no cover
    from uniform_rod_platform_dynamics import UniformRodPlatform3DOF
    from wave_disturbance import WaveDisturbance


class PlatformRLEnvUniform(gym.Env):
    """
    使用 UniformRodPlatform3DOF 的 3-UPS/PU 平台强化学习环境。

    状态空间 (9维):
    - q = [z, alpha, beta]
    - qd = [z_dot, alpha_dot, beta_dot]
    - e = q - q_des

    动作空间 (3维):
    - v_RL = [v_z, v_alpha, v_beta]

    控制律:
    - u = tau_model + v_RL
    - tau_model: 基于 uniform-rod 动力学的模型补偿项
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
        super().__init__()

        self.use_model_compensation = use_model_compensation
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.constraint_penalty = constraint_penalty
        self.soft_constraint = soft_constraint
        self.q_des_type = q_des_type

        # 动力学模型替换为 uniform rod 版本
        self.platform = UniformRodPlatform3DOF(dt=dt)

        # 海浪扰动保持与旧版一致
        self.wave = WaveDisturbance(Hs=Hs, T1=T1, random_seed=None)

        z_range = self.platform.z_max - self.platform.z_min
        alpha_range = 2 * self.platform.alpha_max
        beta_range = 2 * self.platform.beta_max

        self.state_dim = 9
        self.action_dim = 3

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.state_dim,), dtype=np.float32
        )

        self.action_scale = 5000.0
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

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
        z_center = 1.0

        if self.q_des_type == 'sinusoidal':
            z_des = z_center + 0.1 * np.sin(2 * np.pi * 0.2 * t)
            alpha_des = 0.1 * np.sin(2 * np.pi * 0.15 * t)
            beta_des = 0.08 * np.sin(2 * np.pi * 0.25 * t + np.pi / 6)

            zd_des = 0.1 * 2 * np.pi * 0.2 * np.cos(2 * np.pi * 0.2 * t)
            alphad_des = 0.1 * 2 * np.pi * 0.15 * np.cos(2 * np.pi * 0.15 * t)
            betad_des = 0.08 * 2 * np.pi * 0.25 * np.cos(2 * np.pi * 0.25 * t + np.pi / 6)
        elif self.q_des_type == 'constant':
            z_des = z_center
            alpha_des = 0.0
            beta_des = 0.0
            zd_des = 0.0
            alphad_des = 0.0
            betad_des = 0.0
        elif self.q_des_type == 'random':
            if not hasattr(self, '_random_trajectory'):
                self._generate_random_trajectory()
            z_des, alpha_des, beta_des = self._random_trajectory['q'](t)
            zd_des, alphad_des, betad_des = self._random_trajectory['qd'](t)
        else:
            raise ValueError(f"Unknown q_des_type: {self.q_des_type}")

        return np.array([z_des, alpha_des, beta_des]), np.array([zd_des, alphad_des, betad_des])

    def _generate_random_trajectory(self):
        np.random.seed()
        n_freq = 3
        freqs = np.random.uniform(0.1, 0.5, n_freq)
        amps_z = np.random.uniform(0.02, 0.08, n_freq)
        amps_alpha = np.random.uniform(0.02, 0.1, n_freq)
        amps_beta = np.random.uniform(0.02, 0.08, n_freq)
        phases = np.random.uniform(0, 2 * np.pi, n_freq)

        def q_func(t):
            z = 1.0 + np.sum(amps_z * np.sin(2 * np.pi * freqs * t + phases))
            alpha = np.sum(amps_alpha * np.sin(2 * np.pi * freqs * t + phases + np.pi / 3))
            beta = np.sum(amps_beta * np.sin(2 * np.pi * freqs * t + phases + 2 * np.pi / 3))
            return z, alpha, beta

        def qd_func(t):
            zd = np.sum(amps_z * 2 * np.pi * freqs * np.cos(2 * np.pi * freqs * t + phases))
            alphad = np.sum(amps_alpha * 2 * np.pi * freqs * np.cos(2 * np.pi * freqs * t + phases + np.pi / 3))
            betad = np.sum(amps_beta * 2 * np.pi * freqs * np.cos(2 * np.pi * freqs * t + phases + 2 * np.pi / 3))
            return zd, alphad, betad

        self._random_trajectory = {'q': q_func, 'qd': qd_func}

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

    def _compute_reward(self, q, qd, q_des, qd_des, u, constraint_info, time_weight=1.0):
        e = q - q_des
        ed = qd - qd_des

        pos_weights = np.array([1000.0, 500.0, 500.0])
        r_pos = -time_weight * np.sum(pos_weights * e ** 2) / 10000.0

        vel_weights = np.array([100.0, 50.0, 50.0])
        r_vel = -time_weight * np.sum(vel_weights * ed ** 2) / 10000.0

        r_control = -0.0001 * np.sum((u / 10000.0) ** 2)
        reward = r_pos + r_vel + r_control

        if self.soft_constraint:
            constraint_penalty = 0.0
            if q[0] < self.platform.z_min:
                constraint_penalty += 0.5 * (self.platform.z_min - q[0]) ** 2
            if q[0] > self.platform.z_max:
                constraint_penalty += 0.5 * (q[0] - self.platform.z_max) ** 2
            if abs(q[1]) > self.platform.alpha_max:
                constraint_penalty += 0.5 * (abs(q[1]) - self.platform.alpha_max) ** 2
            if abs(q[2]) > self.platform.beta_max:
                constraint_penalty += 0.5 * (abs(q[2]) - self.platform.beta_max) ** 2

            reward -= constraint_penalty

        return float(np.clip(reward, -10.0, 0.0))

    def _compute_temporal_weight(self, step):
        progress = step / self.max_episode_steps if self.max_episode_steps > 0 else 0
        return 1.0 + progress

    def _compute_convergence_reward(self, e, step):
        if step < 0.8 * self.max_episode_steps:
            return 0.0

        z_threshold = 0.01
        alpha_threshold = 0.02
        beta_threshold = 0.02
        converged = (abs(e[0]) < z_threshold and abs(e[1]) < alpha_threshold and abs(e[2]) < beta_threshold)
        return 2.0 if converged else 0.0

    def reset(self):
        z0 = np.random.uniform(0.9, 1.1)
        alpha0 = np.random.uniform(-0.1, 0.1)
        beta0 = np.random.uniform(-0.1, 0.1)

        self.q = np.array([z0, alpha0, beta0])
        self.qd = np.zeros(3)
        self.step_count = 0
        self.episode_time = 0.0

        if self.q_des_type == 'random' and hasattr(self, '_random_trajectory'):
            delattr(self, '_random_trajectory')

        # Reuse the existing wave model to avoid repeated RAO/object initialization cost.
        # If you later need per-episode random wave phases, add an explicit reseed method
        # instead of reconstructing the full object here.

        self.history = {
            'time': [], 'q': [], 'qd': [], 'q_des': [], 'u': [], 'v_RL': [],
            'tau_model': [], 'reward': [], 'constraint_violation': []
        }

        self.q_des, self.qd_des = self._get_desired_trajectory(0.0)
        return self._normalize_state(self.q, self.qd, self.q_des)

    def step(self, action):
        action = np.array(action).flatten()
        v_RL = self._denormalize_action(action)

        self.q_des, self.qd_des = self._get_desired_trajectory(self.episode_time)
        tau_model = self._compute_model_compensation(self.q, self.qd, self.q_des, self.qd_des)
        u_legs = tau_model + v_RL

        if not self.soft_constraint:
            u_legs = np.clip(u_legs, self.platform.u_min, self.platform.u_max)

        disturbance = self.wave.generate_disturbance(np.array([self.episode_time]))[0]

        qdd = self.platform.forward_dynamics(self.q, self.qd, u_legs)
        M = self.platform.mass_matrix(self.q)
        qdd_disturbed = qdd + np.linalg.solve(M, disturbance)

        self.qd = self.qd + qdd_disturbed * self.dt
        self.q = self.q + self.qd * self.dt

        constraint_info = self.platform.check_constraints(self.q, u_legs)
        time_weight = self._compute_temporal_weight(self.step_count)
        base_reward = self._compute_reward(
            self.q, self.qd, self.q_des, self.qd_des, u_legs, constraint_info,
            time_weight=time_weight
        )

        e = self.q - self.q_des
        convergence_reward = self._compute_convergence_reward(e, self.step_count)
        reward = base_reward + convergence_reward

        self.step_count += 1
        self.episode_time += self.dt

        self.history['time'].append(self.episode_time)
        self.history['q'].append(self.q.copy())
        self.history['qd'].append(self.qd.copy())
        self.history['q_des'].append(self.q_des.copy())
        self.history['u'].append(u_legs.copy())
        self.history['v_RL'].append(v_RL.copy())
        self.history['tau_model'].append(tau_model.copy())
        self.history['reward'].append(reward)
        self.history['constraint_violation'].append(not constraint_info['all_satisfied'])

        done = False
        if self.step_count >= self.max_episode_steps:
            done = True

        if np.linalg.norm(self.q - self.q_des) > 0.5:
            done = True
            reward -= 100.0

        info = {
            'constraint_info': constraint_info,
            'tau_model': tau_model,
            'v_RL': v_RL,
            'u_total': u_legs,
            'model_type': 'UniformRodPlatform3DOF'
        }

        next_state = self._normalize_state(self.q, self.qd, self.q_des)
        return next_state, reward, done, info

    def render(self, mode='human'):
        pass

    def get_history(self):
        return self.history

    def close(self):
        pass


if __name__ == "__main__":
    print("Testing RL Environment with UniformRodPlatform3DOF...")
    env = PlatformRLEnvUniform(use_model_compensation=True, q_des_type='sinusoidal')
    state = env.reset()
    print(f"State shape: {state.shape}")
    print(f"State range: [{state.min():.3f}, {state.max():.3f}]")

    total_reward = 0.0
    for _ in range(100):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break

    print(f"Test episode reward: {total_reward:.2f}")
    print(f"Model type: {info['model_type']}")
    print("Environment test passed!")