"""
第3章V3课程学习训练脚本（UniformRodPlatform3DOF 版本）

说明：
- 本脚本保留原 `train_v3_curriculum.py` 不变；
- 仅将环境替换为 `PlatformRLEnvUniform`，用于尝试第2章更完整动力学模型；
- 训练配置、agent 架构、课程设计尽量保持一致，便于公平对比。
"""

import sys
import os

THIS_DIR = os.path.dirname(__file__)
CH3_DIR = os.path.abspath(os.path.join(THIS_DIR, '..'))
SIM_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
COMMON_DIR = os.path.join(SIM_DIR, 'common')
DISTURBANCE_DIR = os.path.join(SIM_DIR, 'disturbance')
ENV_DIR = os.path.join(CH3_DIR, 'env')
AGENTS_DIR = os.path.join(CH3_DIR, 'agents')

for p in [ENV_DIR, AGENTS_DIR, COMMON_DIR, DISTURBANCE_DIR, SIM_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import torch

try:
    from rl_env_uniform import PlatformRLEnvUniform
    from advanced_sac import AdvancedSAC
except ImportError:  # pragma: no cover
    from rl_env_uniform import PlatformRLEnvUniform
    from advanced_sac import AdvancedSAC


class CurriculumConfigUniform:
    """uniform-rod 版本的课程学习配置。"""

    def __init__(self):
        self.stage1 = {
            'name': 'Stage1_BasicStabilization_Uniform',
            'episode_length': 10,
            'dt': 0.01,
            'max_episodes': 200,
            'Hs': 1.0,
            'T1': 8.0,
            'q_des_type': 'constant',
            'save_dir': 'models_v3_uniform/stage1'
        }

        self.stage2 = {
            'name': 'Stage2_Tracking_Uniform',
            'episode_length': 20,
            'dt': 0.01,
            'max_episodes': 300,
            'Hs': 2.0,
            'T1': 8.0,
            'q_des_type': 'sinusoidal',
            'save_dir': 'models_v3_uniform/stage2'
        }

        self.stage3 = {
            'name': 'Stage3_FullTask_Uniform',
            'episode_length': 30,
            'dt': 0.01,
            'max_episodes': 500,
            'Hs': 2.0,
            'T1': 8.0,
            'q_des_type': 'sinusoidal',
            'save_dir': 'models_v3_uniform/stage3'
        }

    def make_quick_test(self):
        """生成一个用于快速验证的轻量配置。"""
        quick = CurriculumConfigUniform()
        quick.stage1.update({
            'name': 'Stage1_QuickTest_Uniform',
            'episode_length': 2,
            'max_episodes': 2,
            'save_dir': 'models_v3_uniform_quick/stage1'
        })
        quick.stage2.update({
            'name': 'Stage2_QuickTest_Uniform',
            'episode_length': 2,
            'max_episodes': 2,
            'save_dir': 'models_v3_uniform_quick/stage2'
        })
        quick.stage3.update({
            'name': 'Stage3_QuickTest_Uniform',
            'episode_length': 2,
            'max_episodes': 2,
            'save_dir': 'models_v3_uniform_quick/stage3'
        })
        return quick


def train_stage(agent, env_config, stage_name, load_model=None):
    print(f"\n{'='*70}")
    print(f"开始训练 (uniform-rod): {stage_name}")
    print(f"回合长度: {env_config['episode_length']}秒")
    print(f"最大回合: {env_config['max_episodes']}")
    print(f"{'='*70}\n")

    env = PlatformRLEnvUniform(
        use_model_compensation=True,
        dt=env_config['dt'],
        max_episode_steps=int(env_config['episode_length'] / env_config['dt']),
        Hs=env_config['Hs'],
        T1=env_config['T1'],
        q_des_type=env_config['q_des_type']
    )

    if load_model and Path(load_model).exists():
        print(f"加载模型: {load_model}")
        agent.load(load_model)

    save_dir = Path(__file__).parent / env_config['save_dir']
    save_dir.mkdir(parents=True, exist_ok=True)

    history = {
        'episode_rewards': [],
        'best_reward': -float('inf'),
        'best_episode': 0
    }

    step_count = 0
    start_time = time.time()
    max_steps = int(env_config['episode_length'] / env_config['dt'])
    warmup_steps = 10000

    for episode in range(env_config['max_episodes']):
        state = env.reset()
        episode_reward = 0.0

        for _ in range(max_steps):
            if step_count < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, deterministic=False)

            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.add(state, action, reward, next_state, float(done))

            episode_reward += reward
            state = next_state
            step_count += 1

            if step_count >= warmup_steps and len(agent.replay_buffer) >= 512:
                agent.update(batch_size=512)

            if done:
                break

        history['episode_rewards'].append(episode_reward)

        if episode_reward > history['best_reward']:
            history['best_reward'] = episode_reward
            history['best_episode'] = episode
            best_path = save_dir / f'{stage_name}_best.pt'
            agent.save(str(best_path))

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(history['episode_rewards'][-10:])
            elapsed = time.time() - start_time
            print(f"Episode {episode+1:4d} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Avg(10): {avg_reward:7.2f} | "
                  f"Best: {history['best_reward']:.2f} | "
                  f"Time: {elapsed:.0f}s")

        if (episode + 1) % 50 == 0:
            checkpoint_path = save_dir / f'{stage_name}_ep{episode+1}.pt'
            agent.save(str(checkpoint_path))

    final_path = save_dir / f'{stage_name}_final.pt'
    agent.save(str(final_path))

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"{stage_name} 训练完成!")
    print(f"总时间: {total_time/60:.1f}分钟")
    print(f"最佳奖励: {history['best_reward']:.2f} (回合 {history['best_episode']})")
    print(f"最终平均奖励: {np.mean(history['episode_rewards'][-50:]):.2f}")
    print(f"{'='*70}\n")

    return agent, history


def train_curriculum_uniform():
    print("\n" + "="*70)
    print("第3章V3: 课程学习训练 (UniformRodPlatform3DOF + Advanced SAC)")
    print("="*70)

    config = CurriculumConfigUniform()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    agent = AdvancedSAC(
        state_dim=9,
        action_dim=3,
        hidden_dims=[512, 512, 256, 256],
        lr_actor=3e-4,
        lr_critic=1e-3,
        lr_alpha=3e-4,
        gamma=0.99,
        tau=0.005,
        device=device
    )

    print(f"训练设备: {device}")
    print(f"网络架构: {agent.actor.layers}")
    print(f"Actor参数量: {sum(p.numel() for p in agent.actor.parameters())}")
    print(f"Critic参数量: {sum(p.numel() for p in agent.critic1.parameters())}")

    agent, history1 = train_stage(agent, config.stage1, 'Stage1_Uniform', load_model=None)
    agent, history2 = train_stage(
        agent, config.stage2, 'Stage2_Uniform',
        load_model='models_v3_uniform/stage1/Stage1_Uniform_best.pt'
    )
    agent, history3 = train_stage(
        agent, config.stage3, 'Stage3_Uniform',
        load_model='models_v3_uniform/stage2/Stage2_Uniform_best.pt'
    )

    print("\n" + "="*70)
    print("uniform-rod 课程学习训练全部完成!")
    print("="*70)
    print(f"阶段1最佳: {history1['best_reward']:.2f}")
    print(f"阶段2最佳: {history2['best_reward']:.2f}")
    print(f"阶段3最佳: {history3['best_reward']:.2f}")
    print("="*70 + "\n")


def train_curriculum_uniform_quick():
    """用于快速验证环境、训练循环和模型保存路径。"""
    print("\n" + "="*70)
    print("第3章V3 Quick Test: UniformRodPlatform3DOF + Advanced SAC")
    print("="*70)

    config = CurriculumConfigUniform().make_quick_test()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    agent = AdvancedSAC(
        state_dim=9,
        action_dim=3,
        hidden_dims=[128, 128],
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_alpha=3e-4,
        gamma=0.99,
        tau=0.005,
        device=device
    )

    print(f"训练设备: {device}")
    agent, _ = train_stage(agent, config.stage1, 'Stage1_Quick_Uniform', load_model=None)
    print("Quick test completed.")


if __name__ == '__main__':
    train_curriculum_uniform()