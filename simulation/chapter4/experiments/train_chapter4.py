"""
Chapter 4 训练脚本
V5 + ESO + 硬切换 DRARL
"""

import sys
import os

# Get absolute paths
PROJECT_ROOT = '/home/haydn/Documents/AERAOFMINE/mythesis/simulation'
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'common'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'chapter3', 'agents'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'chapter4', 'env'))

import numpy as np
import torch
import time
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from rl_env_chapter4 import PlatformRLEnvChapter4
from v4_sac import V4SAC


class Chapter4Trainer:
    """Chapter 4 训练器 - V5 + ESO + 硬切换"""
    
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Chapter 4训练配置 - 与V5相同，保持可比性
        self.training_config = {
            'episode_length': 30,  # 30秒
            'max_episodes': 400,   # 400回合
            'Hs': 2.0,
            'T1': 8.0,
            'q_des_type': 'sinusoidal',
            'lr_actor': 1e-4,
            'lr_critic': 3e-4,
            # Chapter 4 新增参数
            'use_eso': True,
            'use_hard_switch': True,
            'eso_omega': 20.0,
            'switch_threshold': 0.7,
            'switch_beta': 0.5,
        }
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.training_history = {
            'variant_name': 'Chapter4_ESO_HardSwitch',
            'config': self.training_config,
            'start_time': None,
            'end_time': None,
            'episodes': []
        }
    
    def create_env(self):
        """创建Chapter 4训练环境"""
        max_steps = int(self.training_config['episode_length'] / 0.01)
        
        env = PlatformRLEnvChapter4(
            use_model_compensation=True,
            use_eso=self.training_config['use_eso'],
            use_hard_switch=self.training_config['use_hard_switch'],
            max_episode_steps=max_steps,
            Hs=self.training_config['Hs'],
            T1=self.training_config['T1'],
            q_des_type=self.training_config['q_des_type'],
            diverge_threshold=0.5,
            eso_omega=self.training_config['eso_omega'],
            switch_threshold=self.training_config['switch_threshold'],
            switch_beta=self.training_config['switch_beta'],
        )
        
        return env
    
    def train(self):
        """开始训练"""
        print("\n" + "="*70)
        print("Chapter 4 训练开始 (V5 + ESO + 硬切换)")
        print("="*70)
        print(f"配置: {self.training_config}")
        
        # 创建环境
        env = self.create_env()
        
        # 创建SAC智能体
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        agent = V4SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[512, 512, 256, 256],
            lr_actor=self.training_config['lr_actor'],
            lr_critic=self.training_config['lr_critic'],
            gamma=0.995,
            tau=0.002,
            device=self.device
        )
        
        # 计算参数量
        actor_params = sum(p.numel() for p in agent.actor.parameters())
        critic_params = sum(p.numel() for p in agent.critic1.parameters()) + sum(p.numel() for p in agent.critic2.parameters())
        print(f"V4SAC Network Info:")
        print(f"  Actor parameters: {actor_params:,}")
        print(f"  Critic parameters: {critic_params:,}")
        print(f"  Total parameters: {actor_params + critic_params:,}")
        print(f"  Gamma: {agent.gamma}, Tau: {agent.tau}, GradClip: {agent.grad_clip}")
        print(f"设备: {self.device}")
        print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
        
        max_steps = int(self.training_config['episode_length'] / 0.01)
        
        # 训练统计
        episode_rewards = []
        best_reward = -float('inf')
        best_episode = 0
        
        step_count = 0
        start_time = time.time()
        warmup_steps = 5000
        batch_size = 512
        
        self.training_history['start_time'] = datetime.now().isoformat()
        
        for episode in range(self.training_config['max_episodes']):
            state = env.reset()
            episode_reward = 0
            episode_ise = 0
            
            for step in range(max_steps):
                # 选择动作
                if step_count < warmup_steps:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(state, deterministic=False)
                
                # 执行动作
                next_state, reward, done, info = env.step(action)
                
                # 存储经验
                agent.replay_buffer.add(state, action, reward, next_state, float(done))
                
                episode_reward += reward
                episode_ise += info.get('ise', 0)
                
                # 训练智能体
                if step_count >= warmup_steps and len(agent.replay_buffer) >= batch_size:
                    agent.update(batch_size)
                
                state = next_state
                step_count += 1
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            
            # 记录最佳
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_episode = episode
                # 保存最佳模型
                agent.save(self.save_dir / 'best_model.pt')
                print(f"Episode {episode}: 新最佳! Reward={episode_reward:.2f}, ISE={episode_ise*0.01:.6f}")
            elif episode % 50 == 0:
                print(f"Episode {episode}: Reward={episode_reward:.2f}, ISE={episode_ise*0.01:.6f}, Best={best_reward:.2f}")
            
            # 保存 checkpoint
            if episode > 0 and episode % 100 == 0:
                agent.save(self.save_dir / f'checkpoint_ep{episode}.pt')
            
            # 记录历史
            self.training_history['episodes'].append({
                'episode': episode,
                'reward': episode_reward,
                'ise': episode_ise * 0.01,
                'best': episode == best_episode
            })
        
        # 保存最终模型
        agent.save(self.save_dir / 'final_model.pt')
        
        # 保存训练历史
        self.training_history['end_time'] = datetime.now().isoformat()
        self.training_history['best_reward'] = best_reward
        self.training_history['best_episode'] = best_episode
        self.training_history['total_time'] = time.time() - start_time
        
        with open(self.save_dir / 'training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"\n训练完成!")
        print(f"最佳Reward: {best_reward:.2f} (Episode {best_episode})")
        print(f"总时间: {self.training_history['total_time']/60:.1f}分钟")
        
        return agent, self.training_history


if __name__ == '__main__':
    # 创建输出目录
    output_dir = Path(__file__).parent / 'chapter4_models'
    
    # 创建训练器并开始训练
    trainer = Chapter4Trainer(save_dir=str(output_dir))
    agent, history = trainer.train()
    
    print(f"\n模型保存在: {output_dir}")
    print(f"训练历史: {output_dir / 'training_history.json'}")
