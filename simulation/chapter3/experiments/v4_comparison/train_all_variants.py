"""
V4 Improved变体训练脚本
为每个改进方案独立训练完整模型
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'common'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'env'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'agents'))

import numpy as np
import torch
import time
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from rl_env_v4_improved import PlatformRLEnvV4Improved
from v4_sac import V4SAC


class V4ImprovedTrainer:
    """V4 Improved变体训练器"""
    
    def __init__(self, variant_name: str, variant_config: dict, base_dir: str):
        self.variant_name = variant_name
        self.variant_config = variant_config
        self.base_dir = Path(base_dir)
        self.save_dir = self.base_dir / variant_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练配置
        self.stages = [
            {
                'name': 'Stage1',
                'episode_length': 20,
                'max_episodes': 200,
                'Hs': 1.0,
                'T1': 8.0,
                'q_des_type': 'constant',
                'lr_actor': 1e-4,
                'lr_critic': 3e-4,
                'patience': 30
            },
            {
                'name': 'Stage2',
                'episode_length': 30,
                'max_episodes': 250,
                'Hs': 2.0,
                'T1': 8.0,
                'q_des_type': 'sinusoidal',
                'lr_actor': 1e-4,
                'lr_critic': 3e-4,
                'patience': 40
            },
            {
                'name': 'Stage3',
                'episode_length': 40,
                'max_episodes': 300,
                'Hs': 2.0,
                'T1': 8.0,
                'q_des_type': 'sinusoidal',
                'lr_actor': 5e-5,
                'lr_critic': 2e-4,
                'patience': 50
            }
        ]
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 训练历史
        self.training_history = {
            'variant_name': variant_name,
            'variant_config': variant_config,
            'stages': [],
            'start_time': None,
            'end_time': None
        }
    
    def create_env(self, stage_config):
        """创建训练环境"""
        max_steps = int(stage_config['episode_length'] / 0.01)
        
        env = PlatformRLEnvV4Improved(
            use_model_compensation=True,
            max_episode_steps=max_steps,
            Hs=stage_config['Hs'],
            T1=stage_config['T1'],
            q_des_type=stage_config['q_des_type'],
            **self.variant_config  # 应用变体特定配置
        )
        
        return env
    
    def train_stage(self, agent, stage_config, stage_name, load_model=None):
        """训练单个阶段"""
        print(f"\n{'='*70}")
        print(f"训练 {stage_name} - {self.variant_name}")
        print(f"{'='*70}")
        print(f"回合长度: {stage_config['episode_length']}秒")
        print(f"最大回合: {stage_config['max_episodes']}")
        print(f"海浪Hs: {stage_config['Hs']}m")
        print(f"轨迹类型: {stage_config['q_des_type']}")
        print(f"变体配置: {self.variant_config}")
        
        # 创建环境
        env = self.create_env(stage_config)
        
        # 调整学习率
        for param_group in agent.actor_optimizer.param_groups:
            param_group['lr'] = stage_config['lr_actor']
        for param_group in agent.critic1_optimizer.param_groups:
            param_group['lr'] = stage_config['lr_critic']
        for param_group in agent.critic2_optimizer.param_groups:
            param_group['lr'] = stage_config['lr_critic']
        
        # 加载模型
        if load_model and Path(load_model).exists():
            print(f"加载模型: {load_model}")
            agent.load(load_model)
        
        # 训练统计
        stage_history = {
            'stage_name': stage_name,
            'config': stage_config,
            'episode_rewards': [],
            'best_reward': -float('inf'),
            'best_episode': 0,
            'no_improvement_count': 0
        }
        
        step_count = 0
        start_time = time.time()
        warmup_steps = 5000
        batch_size = 512
        max_steps = int(stage_config['episode_length'] / 0.01)
        
        for episode in range(stage_config['max_episodes']):
            state = env.reset()
            episode_reward = 0
            
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
                state = next_state
                step_count += 1
                
                # 训练
                if step_count >= warmup_steps and len(agent.replay_buffer) >= batch_size:
                    agent.update(batch_size=batch_size)
                
                if done:
                    break
            
            stage_history['episode_rewards'].append(episode_reward)
            
            # 检查改进
            improved = False
            if episode_reward > stage_history['best_reward']:
                stage_history['best_reward'] = episode_reward
                stage_history['best_episode'] = episode
                stage_history['no_improvement_count'] = 0
                improved = True
                
                # 保存最佳模型
                best_path = self.save_dir / f'{stage_name}_best.pt'
                agent.save(str(best_path))
            else:
                stage_history['no_improvement_count'] += 1
            
            # 检查发散
            if episode_reward < -100 and episode > 20:
                print(f"\n⚠️ 检测到发散 (reward={episode_reward:.2f})，加载最佳模型")
                best_path = self.save_dir / f'{stage_name}_best.pt'
                if best_path.exists():
                    agent.load(str(best_path))
                    stage_history['no_improvement_count'] = 0
            
            # 早停
            if stage_history['no_improvement_count'] >= stage_config['patience']:
                print(f"\n⏹️ 早停: {stage_config['patience']}回合无提升")
                break
            
            # 打印进度
            if (episode + 1) % 10 == 0 or improved:
                avg_reward = np.mean(stage_history['episode_rewards'][-10:])
                elapsed = time.time() - start_time
                print(f"Episode {episode+1:4d} | "
                      f"Reward: {episode_reward:8.2f} | "
                      f"Avg(10): {avg_reward:8.2f} | "
                      f"Best: {stage_history['best_reward']:.2f} (#{stage_history['best_episode']}) | "
                      f"NoImp: {stage_history['no_improvement_count']:2d} | "
                      f"Time: {elapsed/60:.1f}min")
            
            # 定期保存
            if (episode + 1) % 50 == 0:
                checkpoint_path = self.save_dir / f'{stage_name}_ep{episode+1}.pt'
                agent.save(str(checkpoint_path))
        
        # 最终保存
        final_path = self.save_dir / f'{stage_name}_final.pt'
        agent.save(str(final_path))
        
        total_time = time.time() - start_time
        stage_history['total_time'] = total_time
        stage_history['total_episodes'] = len(stage_history['episode_rewards'])
        
        print(f"\n{stage_name} 完成!")
        print(f"  总回合: {stage_history['total_episodes']}")
        print(f"  最佳奖励: {stage_history['best_reward']:.2f} (回合 {stage_history['best_episode']})")
        print(f"  总时间: {total_time/60:.1f}分钟")
        
        return agent, stage_history
    
    def train(self):
        """完整训练流程"""
        print(f"\n{'='*70}")
        print(f"开始训练: {self.variant_name}")
        print(f"变体配置: {self.variant_config}")
        print(f"设备: {self.device}")
        print(f"{'='*70}")
        
        self.training_history['start_time'] = datetime.now().isoformat()
        
        # 创建agent
        agent = V4SAC(
            state_dim=9,
            action_dim=3,
            hidden_dims=[512, 512, 256, 256],
            lr_actor=1e-4,
            lr_critic=3e-4,
            lr_alpha=1e-4,
            gamma=0.995,
            tau=0.002,
            grad_clip=1.0,
            dropout=0.1,
            device=self.device
        )
        
        total_start = time.time()
        
        # Stage 1
        agent, history1 = self.train_stage(
            agent, self.stages[0], 'Stage1', load_model=None
        )
        self.training_history['stages'].append(history1)
        
        # Stage 2
        stage1_best = self.save_dir / 'Stage1_best.pt'
        agent, history2 = self.train_stage(
            agent, self.stages[1], 'Stage2', load_model=str(stage1_best)
        )
        self.training_history['stages'].append(history2)
        
        # Stage 3
        stage2_best = self.save_dir / 'Stage2_best.pt'
        agent, history3 = self.train_stage(
            agent, self.stages[2], 'Stage3', load_model=str(stage2_best)
        )
        self.training_history['stages'].append(history3)
        
        total_time = time.time() - total_start
        self.training_history['end_time'] = datetime.now().isoformat()
        self.training_history['total_time'] = total_time
        
        # 保存训练历史
        history_file = self.save_dir / 'training_history.json'
        with open(history_file, 'w') as f:
            # 转换为可序列化格式
            history_serializable = {
                'variant_name': self.training_history['variant_name'],
                'variant_config': self.training_history['variant_config'],
                'start_time': self.training_history['start_time'],
                'end_time': self.training_history['end_time'],
                'total_time': total_time,
                'stages': [
                    {
                        'stage_name': h['stage_name'],
                        'best_reward': float(h['best_reward']),
                        'best_episode': h['best_episode'],
                        'total_episodes': h['total_episodes'],
                        'total_time': h['total_time']
                    }
                    for h in self.training_history['stages']
                ]
            }
            json.dump(history_serializable, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"{self.variant_name} 训练完成!")
        print(f"{'='*70}")
        print(f"总时间: {total_time/60:.1f}分钟")
        print(f"各阶段最佳奖励:")
        print(f"  Stage1: {history1['best_reward']:.2f}")
        print(f"  Stage2: {history2['best_reward']:.2f}")
        print(f"  Stage3: {history3['best_reward']:.2f}")
        print(f"训练历史已保存: {history_file}")
        
        return agent


def main():
    """训练所有变体"""
    base_dir = Path(__file__).parent / 'trained_variants'
    
    # 定义所有变体
    variants = {
        'Original_V4_Improved': {
            'diverge_threshold': 0.5,
            'warning_threshold': 0.4,
            'warning_penalty': 5.0  # 默认预警惩罚
        },
        'Variant1_Relaxed_Threshold': {
            'diverge_threshold': 1.0,  # ✓ 放宽到1.0
            'warning_threshold': 0.4,
            'warning_penalty': 5.0
        },
        'Variant2_No_Warning_Penalty': {
            'diverge_threshold': 0.5,
            'warning_threshold': 0.4,
            'warning_penalty': 0.0  # ✓ 禁用预警惩罚
        },
    }
    
    print(f"\n{'='*70}")
    print("V4 Improved变体完整训练")
    print(f"{'='*70}")
    print(f"将训练 {len(variants)} 个变体:")
    for name, config in variants.items():
        print(f"  - {name}:")
        print(f"      diverge_threshold={config['diverge_threshold']}, "
              f"warning_penalty={config['warning_penalty']}")
    print(f"\n预计总时间: {len(variants) * 3} - {len(variants) * 4} 小时")
    print(f"{'='*70}\n")
    
    # 逐个训练
    for variant_name, variant_config in variants.items():
        print(f"\n\n{'#'*70}")
        print(f"# 开始训练: {variant_name}")
        print(f"{'#'*70}\n")
        
        trainer = V4ImprovedTrainer(variant_name, variant_config, str(base_dir))
        agent = trainer.train()
        
        print(f"\n✓ {variant_name} 训练完成！\n")
    
    print(f"\n\n{'='*70}")
    print("所有变体训练完成！")
    print(f"{'='*70}")
    print(f"训练结果保存在: {base_dir}")


if __name__ == "__main__":
    main()
