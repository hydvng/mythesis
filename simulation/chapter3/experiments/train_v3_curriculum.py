"""
第3章V3课程学习训练脚本
三阶段训练：
- 阶段1: 10秒回合, Hs=1.0m, 恒定姿态
- 阶段2: 20秒回合, Hs=2.0m, 小幅正弦
- 阶段3: 30秒回合, Hs=2.0m, 全幅正弦
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'common'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'env'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'agents'))

import numpy as np
import torch
import time
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from rl_env import PlatformRLEnv
from advanced_sac import AdvancedSAC


class CurriculumConfig:
    """课程学习配置"""
    
    def __init__(self):
        # 阶段1: 基础稳定
        self.stage1 = {
            'name': 'Stage1_BasicStabilization',
            'episode_length': 10,  # 秒
            'dt': 0.01,
            'max_episodes': 200,
            'Hs': 1.0,
            'T1': 8.0,
            'q_des_type': 'constant',
            'save_dir': 'models_v3/stage1'
        }
        
        # 阶段2: 跟踪训练
        self.stage2 = {
            'name': 'Stage2_Tracking',
            'episode_length': 20,
            'dt': 0.01,
            'max_episodes': 300,
            'Hs': 2.0,
            'T1': 8.0,
            'q_des_type': 'sinusoidal',
            'save_dir': 'models_v3/stage2'
        }
        
        # 阶段3: 完整任务
        self.stage3 = {
            'name': 'Stage3_FullTask',
            'episode_length': 30,
            'dt': 0.01,
            'max_episodes': 500,
            'Hs': 2.0,
            'T1': 8.0,
            'q_des_type': 'sinusoidal',
            'save_dir': 'models_v3/stage3'
        }


def train_stage(agent, env_config, stage_name, load_model=None):
    """训练单个阶段"""
    
    print(f"\n{'='*70}")
    print(f"开始训练: {stage_name}")
    print(f"回合长度: {env_config['episode_length']}秒")
    print(f"最大回合: {env_config['max_episodes']}")
    print(f"{'='*70}\n")
    
    # 创建环境
    env = PlatformRLEnv(
        use_model_compensation=True,
        dt=env_config['dt'],
        max_episode_steps=int(env_config['episode_length'] / env_config['dt']),
        Hs=env_config['Hs'],
        T1=env_config['T1'],
        q_des_type=env_config['q_des_type']
    )
    
    # 加载模型
    if load_model and Path(load_model).exists():
        print(f"加载模型: {load_model}")
        agent.load(load_model)
    
    # 创建保存目录
    save_dir = Path(__file__).parent / env_config['save_dir']
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练统计
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
        episode_reward = 0
        
        for step in range(max_steps):
            # 选择动作
            if step_count < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, deterministic=False)
            
            # 执行
            next_state, reward, done, info = env.step(action)
            
            # 存储
            agent.replay_buffer.add(state, action, reward, next_state, float(done))
            
            episode_reward += reward
            state = next_state
            step_count += 1
            
            # 训练
            if step_count >= warmup_steps and len(agent.replay_buffer) >= 512:
                agent.update(batch_size=512)
            
            if done:
                break
        
        history['episode_rewards'].append(episode_reward)
        
        # 保存最佳模型
        if episode_reward > history['best_reward']:
            history['best_reward'] = episode_reward
            history['best_episode'] = episode
            best_path = save_dir / f'{stage_name}_best.pt'
            agent.save(str(best_path))
        
        # 打印进度
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(history['episode_rewards'][-10:])
            elapsed = time.time() - start_time
            print(f"Episode {episode+1:4d} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Avg(10): {avg_reward:7.2f} | "
                  f"Best: {history['best_reward']:.2f} | "
                  f"Time: {elapsed:.0f}s")
        
        # 定期保存
        if (episode + 1) % 50 == 0:
            checkpoint_path = save_dir / f'{stage_name}_ep{episode+1}.pt'
            agent.save(str(checkpoint_path))
    
    # 最终保存
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


def train_curriculum():
    """完整课程学习训练"""
    
    print("\n" + "="*70)
    print("第3章V3: 课程学习训练 (Advanced SAC)")
    print("="*70)
    
    config = CurriculumConfig()
    
    # 创建SAC agent (大网络)
    agent = AdvancedSAC(
        state_dim=9,
        action_dim=3,
        hidden_dims=[512, 512, 256, 256],
        lr_actor=3e-4,
        lr_critic=1e-3,
        lr_alpha=3e-4,
        gamma=0.99,
        tau=0.005,
        device='cpu'
    )
    
    print(f"网络架构: {agent.actor.layers}")
    print(f"Actor参数量: {sum(p.numel() for p in agent.actor.parameters())}")
    print(f"Critic参数量: {sum(p.numel() for p in agent.critic1.parameters())}")
    
    # 阶段1: 基础稳定
    agent, history1 = train_stage(
        agent, 
        config.stage1, 
        'Stage1',
        load_model=None
    )
    
    # 阶段2: 跟踪训练
    agent, history2 = train_stage(
        agent,
        config.stage2,
        'Stage2',
        load_model=f"models_v3/stage1/Stage1_best.pt"
    )
    
    # 阶段3: 完整任务
    agent, history3 = train_stage(
        agent,
        config.stage3,
        'Stage3',
        load_model=f"models_v3/stage2/Stage2_best.pt"
    )
    
    print("\n" + "="*70)
    print("课程学习训练全部完成!")
    print("="*70)
    print(f"阶段1最佳: {history1['best_reward']:.2f}")
    print(f"阶段2最佳: {history2['best_reward']:.2f}")
    print(f"阶段3最佳: {history3['best_reward']:.2f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    train_curriculum()
