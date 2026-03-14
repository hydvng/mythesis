"""
第3章：V5 风格的 UniformRod 强化学习环境

核心思路：
1. 延续 UniformRodPlatform3DOF 动力学
2. 奖励函数简化为纯 ISE / step：reward = -||e||^2
3. 保留模型补偿 + RL 残差控制结构
4. 支持可选 burst-step 突变扰动，用于模拟突加载荷/冲击
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
from typing import Tuple

try:
    from uniform_rod_platform_dynamics import UniformRodPlatform3DOF
    from wave_disturbance import WaveDisturbance
except ImportError:  # pragma: no cover
    from uniform_rod_platform_dynamics import UniformRodPlatform3DOF
    from wave_disturbance import WaveDisturbance


class PlatformRLEnvV5Uniform(gym.Env):
    """V5 风格的 UniformRod 环境：纯 ISE 奖励 + 可选突变扰动。"""

    def __init__(self,
                 use_model_compensation: bool = True,
                 dt: float = 0.01,
                 max_episode_steps: int = 3000,
                 Hs: float = 2.0,
                 T1: float = 8.0,
                 q_des_type: str = 'sinusoidal',
                 diverge_threshold: float = 0.5,
                 enable_burst_step: bool = False,
                 step_t0: float = 15.0,
                 step_duration: float = 0.3,
                 step_amplitude=(3000.0, 150.0, 150.0),
                 step_ramp_time: float = 0.03):
        super().__init__()

        self.use_model_compensation = use_model_compensation
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.q_des_type = q_des_type
        self.diverge_threshold = diverge_threshold

        self.enable_burst_step = enable_burst_step
        self.step_t0 = step_t0
        self.step_duration = step_duration
        self.step_amplitude = np.asarray(step_amplitude, dtype=float)
        self.step_ramp_time = step_ramp_time

        self.platform = UniformRodPlatform3DOF(dt=dt)
        self.wave = self._build_wave(Hs, T1)

        z_range = self.platform.z_max - self.platform.z_min
        alpha_range = 2 * self.platform.alpha_max
        beta_range = 2 * self.platform.beta_max

        self.state_dim = 9
        self.action_dim = 3
        self.action_scale = 5000.0

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.state_dim,), dtype=np.float32
        )
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
        self.cumulative_ise = 0.0

        self.history = self._make_history()

    def _build_wave(self, Hs: float, T1: float):
        return WaveDisturbance(
            Hs=Hs,
            T1=T1,
            random_seed=None,
            enable_burst_step=self.enable_burst_step,
            step_t0=self.step_t0,
            step_duration=self.step_duration,
            step_amplitude=self.step_amplitude,
            step_ramp_time=self.step_ramp_time,
        )

    def _make_history(self):
        return {
            'time': [],
            'q': [],
            'qd': [],
            'q_des': [],
            'u': [],
            'v_RL': [],
            'tau_model': [],
            'reward': [],
            'ise': [],
            'error_norm': []
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
        elif self.q_des_type == 'sinusoidal_small':
            z_des = z_center + 0.05 * np.sin(2 * np.pi * 0.15 * t)
            alpha_des = 0.05 * np.sin(2 * np.pi * 0.12 * t)
            beta_des = 0.04 * np.sin(2 * np.pi * 0.18 * t + np.pi / 6)

            zd_des = 0.05 * 2 * np.pi * 0.15 * np.cos(2 * np.pi * 0.15 * t)
            alphad_des = 0.05 * 2 * np.pi * 0.12 * np.cos(2 * np.pi * 0.12 * t)
            betad_des = 0.04 * 2 * np.pi * 0.18 * np.cos(2 * np.pi * 0.18 * t + np.pi / 6)
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
        return np.array(action).flatten() * self.action_scale

    def _compute_reward(self, e):
        ise = float(np.sum(e ** 2))
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
        self.cumulative_ise = 0.0

        self.wave = self._build_wave(self.wave.Hs, self.wave.T1)
        self.history = self._make_history()

        self.q_des, self.qd_des = self._get_desired_trajectory(0.0)
        return self._normalize_state(self.q, self.qd, self.q_des)

    def step(self, action):
        v_RL = self._denormalize_action(action)
        self.q_des, self.qd_des = self._get_desired_trajectory(self.episode_time)

        tau_model = self._compute_model_compensation(self.q, self.qd, self.q_des, self.qd_des)
        u_legs = tau_model + v_RL

        disturbance = self.wave.generate_disturbance(np.array([self.episode_time]))[0]
        qdd = self.platform.forward_dynamics(self.q, self.qd, u_legs)
        M = self.platform.mass_matrix(self.q)
        qdd_disturbed = qdd + np.linalg.solve(M, disturbance)

        self.qd = self.qd + qdd_disturbed * self.dt
        self.q = self.q + self.qd * self.dt

        e = self.q - self.q_des
        reward, ise = self._compute_reward(e)
        self.cumulative_ise += ise * self.dt

        self.step_count += 1
        self.episode_time += self.dt

        error_norm = float(np.linalg.norm(e))
        done = False
        if self.step_count >= self.max_episode_steps:
            done = True
        if error_norm > self.diverge_threshold:
            done = True
            reward -= 100.0

        self.history['time'].append(self.episode_time)
        self.history['q'].append(self.q.copy())
        self.history['qd'].append(self.qd.copy())
        self.history['q_des'].append(self.q_des.copy())
        self.history['u'].append(u_legs.copy())
        self.history['v_RL'].append(v_RL.copy())
        self.history['tau_model'].append(tau_model.copy())
        self.history['reward'].append(reward)
        self.history['ise'].append(ise)
        self.history['error_norm'].append(error_norm)

        info = {
            'error_norm': error_norm,
            'ise': ise,
            'cumulative_ise': self.cumulative_ise,
            'tau_model': tau_model,
            'v_RL': v_RL,
            'u_total': u_legs,
            'model_type': 'UniformRodPlatform3DOF',
            'burst_enabled': self.enable_burst_step,
        }

        next_state = self._normalize_state(self.q, self.qd, self.q_des)
        return next_state, reward, done, info

    def get_history(self):
        return self.history

    def compute_episode_ise(self):
        if not self.history['ise']:
            return 0.0
        return float(np.sum(self.history['ise']) * self.dt)

    def render(self, mode='human'):
        pass

    def close(self):
        pass


if __name__ == '__main__':
    print('Testing V5 Uniform RL Environment...')
    env = PlatformRLEnvV5Uniform(
        use_model_compensation=True,
        max_episode_steps=200,
        Hs=2.0,
        T1=8.0,
        q_des_type='sinusoidal',
        enable_burst_step=True,
        step_t0=0.8,
        step_duration=0.2,
    )

    state = env.reset()
    total_reward = 0.0
    for i in range(100):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        total_reward += reward
        if i % 20 == 0:
            print(f'Step {i}: reward={reward:.4f}, ise={info["ise"]:.6f}, error={info["error_norm"]:.4f}')
        if done:
            break

    print(f'Total reward: {total_reward:.4f}')
    print(f'Episode ISE: {env.compute_episode_ise():.6f}')
    print('Environment test passed.')