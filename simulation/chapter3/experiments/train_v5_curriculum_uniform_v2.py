"""
第3章：V5.1 风格课程学习训练脚本（UniformRodPlatform3DOF 版本）

核心修复：
1. Stage2 改回完整 sinusoidal，减少阶段间分布跳变
2. Stage3 默认关闭 burst-step，先学主任务再谈鲁棒性
3. warmup_steps 恢复到 10000
4. lr_critic 恢复到 1e-3，增强 critic 拟合速度
"""

import sys
import os
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

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
import torch

try:
    from rl_env_v5_uniform_v2 import PlatformRLEnvV5UniformV2
    from advanced_sac import AdvancedSAC
except ImportError:  # pragma: no cover
    from rl_env_v5_uniform_v2 import PlatformRLEnvV5UniformV2
    from advanced_sac import AdvancedSAC


class CurriculumConfigV5UniformV2:
    def __init__(self):
        self.stage1 = {
            'name': 'Stage1_V51Uniform_Basic',
            'episode_length': 10,
            'dt': 0.01,
            'max_episodes': 150,
            'Hs': 1.0,
            'T1': 8.0,
            'q_des_type': 'constant',
            'enable_burst_step': False,
            'save_dir': 'models_v5_uniform_v2/stage1'
        }
        self.stage2 = {
            'name': 'Stage2_V51Uniform_Tracking',
            'episode_length': 20,
            'dt': 0.01,
            'max_episodes': 250,
            'Hs': 2.0,
            'T1': 8.0,
            'q_des_type': 'sinusoidal',
            'enable_burst_step': False,
            'save_dir': 'models_v5_uniform_v2/stage2'
        }
        self.stage3 = {
            'name': 'Stage3_V51Uniform_FullTask',
            'episode_length': 30,
            'dt': 0.01,
            'max_episodes': 400,
            'Hs': 2.0,
            'T1': 8.0,
            'q_des_type': 'sinusoidal',
            'enable_burst_step': False,
            'step_t0': 15.0,
            'step_duration': 0.3,
            'step_amplitude': [3000.0, 150.0, 150.0],
            'step_ramp_time': 0.03,
            'save_dir': 'models_v5_uniform_v2/stage3'
        }

    def make_quick_test(self):
        quick = CurriculumConfigV5UniformV2()
        quick.stage1.update({
            'name': 'Stage1_V51Uniform_Quick',
            'episode_length': 2,
            'max_episodes': 2,
            'save_dir': 'models_v5_uniform_v2_quick/stage1'
        })
        quick.stage2.update({
            'name': 'Stage2_V51Uniform_Quick',
            'episode_length': 2,
            'max_episodes': 2,
            'save_dir': 'models_v5_uniform_v2_quick/stage2'
        })
        quick.stage3.update({
            'name': 'Stage3_V51Uniform_Quick',
            'episode_length': 2,
            'max_episodes': 2,
            'save_dir': 'models_v5_uniform_v2_quick/stage3'
        })
        return quick


def create_env(env_config):
    max_steps = int(env_config['episode_length'] / env_config['dt'])
    return PlatformRLEnvV5UniformV2(
        use_model_compensation=True,
        dt=env_config['dt'],
        max_episode_steps=max_steps,
        Hs=env_config['Hs'],
        T1=env_config['T1'],
        q_des_type=env_config['q_des_type'],
        diverge_threshold=0.5,
        vel_error_weight=0.10,
        constraint_penalty_weight=2.0,
        enable_burst_step=env_config.get('enable_burst_step', False),
        step_t0=env_config.get('step_t0', 15.0),
        step_duration=env_config.get('step_duration', 0.3),
        step_amplitude=env_config.get('step_amplitude', [3000.0, 150.0, 150.0]),
        step_ramp_time=env_config.get('step_ramp_time', 0.03),
    )


def train_stage(agent, env_config, stage_label, load_model=None, warmup_steps=10000, batch_size=512):
    print(f"\n{'=' * 72}")
    print(f"开始训练: {stage_label}")
    print(f"回合长度: {env_config['episode_length']} 秒")
    print(f"最大回合: {env_config['max_episodes']}")
    print(f"轨迹: {env_config['q_des_type']} | 波浪 Hs={env_config['Hs']} | burst={env_config.get('enable_burst_step', False)}")
    print(f"{'=' * 72}\n")

    env = create_env(env_config)

    if load_model and Path(load_model).exists():
        print(f"加载预训练模型: {load_model}")
        agent.load(load_model)

    save_dir = Path(__file__).parent / env_config['save_dir']
    save_dir.mkdir(parents=True, exist_ok=True)

    history = {
        'episode_rewards': [],
        'episode_ise': [],
        'best_reward': -float('inf'),
        'best_ise': float('inf'),
        'best_episode': -1,
    }

    step_count = 0
    start_time = time.time()
    max_steps = int(env_config['episode_length'] / env_config['dt'])

    for episode in range(env_config['max_episodes']):
        state = env.reset()
        episode_reward = 0.0
        episode_ise = 0.0

        for _ in range(max_steps):
            if step_count < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, deterministic=False)

            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.add(state, action, reward, next_state, float(done))

            episode_reward += reward
            episode_ise += info.get('ise', 0.0) * env.dt
            state = next_state
            step_count += 1

            if step_count >= warmup_steps and len(agent.replay_buffer) >= batch_size:
                agent.update(batch_size=batch_size)

            if done:
                break

        history['episode_rewards'].append(float(episode_reward))
        history['episode_ise'].append(float(episode_ise))

        if episode_reward > history['best_reward']:
            history['best_reward'] = float(episode_reward)
            history['best_ise'] = float(episode_ise)
            history['best_episode'] = episode
            best_path = save_dir / f'{stage_label}_best.pt'
            agent.save(str(best_path))
            print(f"Episode {episode:4d}: 新最佳 Reward={episode_reward:.4f}, ISE={episode_ise:.6f}")
        elif (episode + 1) % 10 == 0:
            avg_reward = float(np.mean(history['episode_rewards'][-10:]))
            avg_ise = float(np.mean(history['episode_ise'][-10:]))
            print(
                f"Episode {episode+1:4d} | Reward={episode_reward:9.4f} | Avg10={avg_reward:9.4f} | "
                f"ISE={episode_ise:.6f} | AvgISE10={avg_ise:.6f} | Best={history['best_reward']:.4f}"
            )

        if (episode + 1) % 50 == 0:
            checkpoint_path = save_dir / f'{stage_label}_ep{episode+1}.pt'
            agent.save(str(checkpoint_path))

    final_path = save_dir / f'{stage_label}_final.pt'
    agent.save(str(final_path))

    total_time = time.time() - start_time
    summary = {
        'stage_name': stage_label,
        'config': env_config,
        'best_reward': history['best_reward'],
        'best_ise': history['best_ise'],
        'best_episode': history['best_episode'],
        'final_avg_reward': float(np.mean(history['episode_rewards'][-min(20, len(history['episode_rewards'])):])),
        'final_avg_ise': float(np.mean(history['episode_ise'][-min(20, len(history['episode_ise'])):])),
        'training_time_sec': total_time,
    }

    with open(save_dir / 'training_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{stage_label} 训练完成")
    print(f"总时间: {total_time / 60:.1f} 分钟")
    print(f"最佳 Reward: {history['best_reward']:.4f} (Episode {history['best_episode']})")
    print(f"最佳 ISE: {history['best_ise']:.6f}")

    return agent, history, summary


def build_agent(device='cpu'):
    return AdvancedSAC(
        state_dim=9,
        action_dim=3,
        hidden_dims=[512, 512, 256, 256],
        lr_actor=3e-4,
        lr_critic=1e-3,
        lr_alpha=3e-4,
        gamma=0.99,
        tau=0.005,
        device=device,
    )


def train_curriculum_v5_uniform_v2():
    print("\n" + "=" * 72)
    print("第3章 V5.1 Curriculum Training (UniformRod + AdvancedSAC)")
    print("=" * 72)

    config = CurriculumConfigV5UniformV2()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = build_agent(device=device)

    print(f"训练设备: {device}")
    print(f"Actor 参数量: {sum(p.numel() for p in agent.actor.parameters())}")
    print(f"Critic 参数量: {sum(p.numel() for p in agent.critic1.parameters())}")

    run_dir = Path(__file__).parent / 'models_v5_uniform_v2'
    run_dir.mkdir(parents=True, exist_ok=True)

    overall = {
        'variant': 'V5_1_Uniform_Curriculum',
        'started_at': datetime.now().isoformat(),
        'device': device,
        'stages': []
    }

    agent, history1, summary1 = train_stage(agent, config.stage1, 'Stage1_V51Uniform', load_model=None)
    overall['stages'].append(summary1)

    stage1_best = str(Path(__file__).parent / config.stage1['save_dir'] / 'Stage1_V51Uniform_best.pt')
    agent, history2, summary2 = train_stage(agent, config.stage2, 'Stage2_V51Uniform', load_model=stage1_best)
    overall['stages'].append(summary2)

    stage2_best = str(Path(__file__).parent / config.stage2['save_dir'] / 'Stage2_V51Uniform_best.pt')
    agent, history3, summary3 = train_stage(agent, config.stage3, 'Stage3_V51Uniform', load_model=stage2_best)
    overall['stages'].append(summary3)

    overall['finished_at'] = datetime.now().isoformat()
    with open(run_dir / 'overall_summary.json', 'w', encoding='utf-8') as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print("V5.1 Uniform Curriculum 全部完成")
    print("=" * 72)
    print(f"Stage1 最佳 Reward: {history1['best_reward']:.4f}, ISE={history1['best_ise']:.6f}")
    print(f"Stage2 最佳 Reward: {history2['best_reward']:.4f}, ISE={history2['best_ise']:.6f}")
    print(f"Stage3 最佳 Reward: {history3['best_reward']:.4f}, ISE={history3['best_ise']:.6f}")
    print("=" * 72 + "\n")


def train_curriculum_v5_uniform_v2_quick():
    print("\n" + "=" * 72)
    print("第3章 V5.1 Uniform Quick Test")
    print("=" * 72)

    config = CurriculumConfigV5UniformV2().make_quick_test()
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
        device=device,
    )

    agent, _, _ = train_stage(agent, config.stage1, 'Stage1_V51Uniform_Quick', warmup_steps=10, batch_size=16)
    stage1_best = str(Path(__file__).parent / config.stage1['save_dir'] / 'Stage1_V51Uniform_Quick_best.pt')
    agent, _, _ = train_stage(agent, config.stage2, 'Stage2_V51Uniform_Quick', load_model=stage1_best, warmup_steps=10, batch_size=16)
    stage2_best = str(Path(__file__).parent / config.stage2['save_dir'] / 'Stage2_V51Uniform_Quick_best.pt')
    agent, _, _ = train_stage(agent, config.stage3, 'Stage3_V51Uniform_Quick', load_model=stage2_best, warmup_steps=10, batch_size=16)
    print('Quick test completed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train V5.1-style Uniform curriculum SAC')
    parser.add_argument('--quick', action='store_true', help='run a tiny smoke/quick test')
    args = parser.parse_args()

    if args.quick:
        train_curriculum_v5_uniform_v2_quick()
    else:
        train_curriculum_v5_uniform_v2()