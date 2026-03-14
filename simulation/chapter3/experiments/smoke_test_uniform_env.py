"""
第3章 uniform-rod 环境 smoke test。

用途：
- 独立验证 `PlatformRLEnvUniform` 能否完成导入、reset、step；
- 不启动完整训练，便于排查环境/依赖/路径问题；
- 保留原 chapter3 训练实现不变。
"""

import sys
import os

THIS_DIR = os.path.dirname(__file__)
CH3_DIR = os.path.abspath(os.path.join(THIS_DIR, '..'))
SIM_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
COMMON_DIR = os.path.join(SIM_DIR, 'common')
DISTURBANCE_DIR = os.path.join(SIM_DIR, 'disturbance')
ENV_DIR = os.path.join(CH3_DIR, 'env')

for p in [ENV_DIR, COMMON_DIR, DISTURBANCE_DIR, SIM_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

try:
    from rl_env_uniform import PlatformRLEnvUniform
except ImportError:  # pragma: no cover
    from rl_env_uniform import PlatformRLEnvUniform


def run_smoke_test(n_steps: int = 20):
    print("=" * 70)
    print("Smoke test: PlatformRLEnvUniform")
    print("=" * 70)

    env = PlatformRLEnvUniform(
        use_model_compensation=True,
        dt=0.01,
        max_episode_steps=200,
        Hs=1.0,
        T1=8.0,
        q_des_type='constant'
    )

    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    print(f"Initial normalized state range: [{state.min():.3f}, {state.max():.3f}]")

    total_reward = 0.0
    info = {}
    for k in range(n_steps):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        print(
            f"step={k+1:02d} | reward={reward:8.3f} | done={done} | "
            f"q={env.q.round(4)}"
        )
        state = next_state
        if done:
            break

    print("-" * 70)
    print(f"Total reward: {total_reward:.3f}")
    print(f"Model type: {info.get('model_type', 'N/A')}")
    print(f"Constraint satisfied: {info.get('constraint_info', {}).get('all_satisfied', 'N/A')}")
    print("Smoke test finished.")


if __name__ == '__main__':
    run_smoke_test()
