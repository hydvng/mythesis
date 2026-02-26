"""
V4优化版训练脚本
改进:
1. 40秒回合 (原为50秒)
2. 三阶段课程: 20s -> 30s -> 40s
3. 更大的梯度裁剪: 1.0 (原为0.5)
4. 降低学习率: critic 3e-4, actor 1e-4
5. 更大的回放缓冲区: 200万
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

from rl_env_v4 import PlatformRLEnvV4
from v4_sac import V4SAC


class V4OptimizedConfig:
    """V4优化版配置"""
    
    def __init__(self):
        # 阶段1: 基础 (20秒)
        self.stage1 = {
            'name': 'Stage1_Basic',
            'episode_length': 20,
            'dt': 0.01,
            'max_episodes': 200,
            'Hs': 1.0,
            'T1': 8.0,
            'q_des_type': 'constant',
            'save_dir': 'models_v4_optimized/stage1',
            'lr_critic': 3e-4,
            'lr_actor': 1e-4,
            'patience': 30
        }
        
        # 阶段2: 过渡 (30秒)
        self.stage2 = {
            'name': 'Stage2_Transition',
            'episode_length': 30,
            'dt': 0.01,
            'max_episodes': 250,
            'Hs': 2.0,
            'T1': 8.0,
            'q_des_type': 'sinusoidal',
            'save_dir': 'models_v4_optimized/stage2',
            'lr_critic': 3e-4,
            'lr_actor': 1e-4,
            'patience': 40
        }
        
        # 阶段3: 完整 (40秒)
        self.stage3 = {
            'name': 'Stage3_Full',
            'episode_length': 40,
            'dt': 0.01,
            'max_episodes': 300,
            'Hs': 2.0,
            'T1': 8.0,
            'q_des_type': 'sinusoidal',
            'save_dir': 'models_v4_optimized/stage3',
            'lr_critic': 2e-4,
            'lr_actor': 5e-5,
            'patience': 50
        }


def train_stage(agent, env_config, stage_name, load_model=None):
    """训练单个阶段"""
    
    print(f"\n{'='*70}")
    print(f"开始训练: {stage_name}")
    print(f"回合长度: {env_config['episode_length']}秒")
    print(f"最大回合: {env_config['max_episodes']}")
    print(f"学习率: critic={env_config['lr_critic']:.0e}, actor={env_config['lr_actor']:.0e}")
    print(f"{'='*70}\n")
    
    # 调整agent学习率
    agent.lr_critic_init = env_config['lr_critic']
    agent.lr_actor_init = env_config['lr_actor']
    for param_group in agent.critic1_optimizer.param_groups:
        param_group['lr'] = env_config['lr_critic']
    for param_group in agent.actor_optimizer.param_groups:
        param_group['lr'] = env_config['lr_actor']
    
    # 创建环境
    env = PlatformRLEnvV4(
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
        'best_episode': 0,
        'no_improvement_count': 0
    }
    
    step_count = 0
    start_time = time.time()
    
    max_steps = int(env_config['episode_length'] / env_config['dt'])
    warmup_steps = 5000
    patience = env_config['patience']
    
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
        
        # 检查是否有提升
        improved = False
        if episode_reward > history['best_reward']:
            history['best_reward'] = episode_reward
            history['best_episode'] = episode
            history['no_improvement_count'] = 0
            improved = True
            
            # 保存最佳模型
            best_path = save_dir / f'{stage_name}_best.pt'
            agent.save(str(best_path))
        else:
            history['no_improvement_count'] += 1
        
        # 检查发散（更严格的阈值）
        if episode_reward < -100 and episode > 20:
            print(f"\n⚠️ 检测到发散 (reward={episode_reward:.2f})，加载最佳模型")
            best_path = save_dir / f'{stage_name}_best.pt'
            if best_path.exists():
                agent.load(str(best_path))
                history['no_improvement_count'] = 0
        
        # 早停检查
        if history['no_improvement_count'] >= patience:
            print(f"\n⏹️ 早停: {patience}回合无提升")
            break
        
        # 学习率衰减（每50回合）
        if (episode + 1) % 50 == 0:
            agent.adjust_lr(factor=0.95)
        
        # 打印进度
        if (episode + 1) % 10 == 0 or improved:
            avg_reward = np.mean(history['episode_rewards'][-10:])
            elapsed = time.time() - start_time
            print(f"Episode {episode+1:4d} | "
                  f"Reward: {episode_reward:8.2f} | "
                  f"Avg(10): {avg_reward:8.2f} | "
                  f"Best: {history['best_reward']:.2f} (#{history['best_episode']}) | "
                  f"NoImp: {history['no_improvement_count']:2d} | "
                  f"Time: {elapsed/60:.1f}min")
        
        # 定期保存检查点
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


def train_v4_optimized():
    """V4优化版完整训练"""
    
    print("\n" + "="*70)
    print("V4优化版: 三阶段课程学习训练 (40秒回合)")
    print("改进: 更温和的时序权重, 更大的梯度裁剪, 40秒回合")
    print("="*70)
    
    config = V4OptimizedConfig()
    
    # 创建V4 SAC agent (优化版)
    agent = V4SAC(
        state_dim=9,
        action_dim=3,
        hidden_dims=[512, 512, 256, 256],
        lr_actor=1e-4,
        lr_critic=3e-4,
        lr_alpha=1e-4,
        gamma=0.995,
        tau=0.002,
        grad_clip=1.0,    # 优化: 更大的梯度裁剪
        dropout=0.1,
        device=None       # 自动检测GPU
    )
    
    print(f"网络架构: {agent.actor.layers}")
    print(f"梯度裁剪: {agent.grad_clip}")
    print(f"回放缓冲区: {agent.replay_buffer.capacity}")
    
    # 阶段1: 基础 (20秒)
    agent, history1 = train_stage(
        agent, 
        config.stage1, 
        'Stage1',
        load_model=None
    )
    
    # 阶段2: 过渡 (30秒)
    agent, history2 = train_stage(
        agent,
        config.stage2,
        'Stage2',
        load_model=f"models_v4_optimized/stage1/Stage1_best.pt"
    )
    
    # 阶段3: 完整 (40秒)
    agent, history3 = train_stage(
        agent,
        config.stage3,
        'Stage3',
        load_model=f"models_v4_optimized/stage2/Stage2_best.pt"
    )
    
    # 最终总结
    print("\n" + "="*70)
    print("V4优化版训练全部完成!")
    print("="*70)
    print(f"阶段1 (20s): {history1['best_reward']:8.2f}")
    print(f"阶段2 (30s): {history2['best_reward']:8.2f}")
    print(f"阶段3 (40s): {history3['best_reward']:8.2f}")
    print("="*70 + "\n")
    
    # 保存最终模型路径
    final_model_path = Path(__file__).parent / 'models_v4_optimized/stage3/Stage3_best.pt'
    print(f"最终最佳模型: {final_model_path}")


if __name__ == "__main__":
    train_v4_optimized()
