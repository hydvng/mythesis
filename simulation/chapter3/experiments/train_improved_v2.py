"""
第3章改进版训练脚本 - 简化版
借鉴DRARLC最佳实践，只改RL配置，不改算法
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'common'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'env'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'agents'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import time
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

from rl_env import PlatformRLEnv
from sac_agent import ReplayBuffer


class LayerNormActor(nn.Module):
    """带LayerNorm的Actor - 4层"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        
        return mean, log_std
    
    def sample(self, state, deterministic=False):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        if deterministic:
            action = torch.tanh(mean)
            log_prob = None
        else:
            normal = Normal(mean, std)
            z = normal.rsample()
            action = torch.tanh(z)
            log_prob = normal.log_prob(z)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob


class LayerNormCritic(nn.Module):
    """带LayerNorm的Critic - 4层"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.q = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        return self.q(x)


class ImprovedSAC:
    """改进版SAC"""
    
    def __init__(self, state_dim=9, action_dim=3, hidden_dim=256,
                 lr_actor=3e-4, lr_critic=3e-3, lr_alpha=3e-4,
                 gamma=0.99, tau=0.005, device='cpu'):
        
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        
        # Actor
        self.actor = LayerNormActor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Twin Critics
        self.critic1 = LayerNormCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic2 = LayerNormCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr_critic)
        
        # Targets
        self.critic1_target = LayerNormCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic2_target = LayerNormCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Auto entropy
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, capacity=200000)
    
    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _ = self.actor.sample(state_tensor, deterministic)
            return action.cpu().numpy()[0]
    
    def update(self, batch_size=256):
        if len(self.replay_buffer) < batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Target Q
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next = self.critic1_target(next_states, next_actions)
            q2_next = self.critic2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + (1 - dones) * self.gamma * q_next
        
        # Update Critics
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        
        critic1_loss = F.mse_loss(q1, q_target)
        critic2_loss = F.mse_loss(q2, q_target)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step()
        
        # Update Actor
        new_actions, log_probs = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Update Alpha
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()
        
        # Soft update
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)
        
        return {
            'critic_loss': critic1_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha.item()
        }
    
    def _soft_update(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def save(self, filepath):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'alpha': self.alpha.item(),
        }, filepath)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())


def train_improved_v2():
    """改进版训练"""
    
    save_dir = Path(__file__).parent / 'models_improved'
    save_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("改进版SAC训练 V2")
    print("="*70)
    
    # 环境
    env = PlatformRLEnv(
        use_model_compensation=True,
        dt=0.01,
        max_episode_steps=500,
        Hs=2.0,
        T1=8.0
    )
    
    # Agent
    agent = ImprovedSAC(
        state_dim=9,
        action_dim=3,
        hidden_dim=256,
        lr_actor=3e-4,
        lr_critic=3e-3,
        lr_alpha=3e-4,
        device='cpu'
    )
    
    # 学习率调度
    scheduler_actor = optim.lr_scheduler.CosineAnnealingLR(
        agent.actor_optimizer, T_max=500
    )
    
    # 配置
    max_episodes = 500
    warmup_steps = 5000
    updates_per_step = 2
    
    history = {
        'episode_rewards': [],
        'best_reward': -float('inf'),
        'best_episode': 0
    }
    
    step_count = 0
    start_time = time.time()
    
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(500):
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
            if step_count >= warmup_steps:
                for _ in range(updates_per_step):
                    agent.update(256)
            
            if done:
                break
        
        history['episode_rewards'].append(episode_reward)
        
        # 打印进度
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(history['episode_rewards'][-10:])
            elapsed = time.time() - start_time
            print(f"Episode {episode+1:4d} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Avg(10): {avg_reward:7.2f} | "
                  f"Time: {elapsed:.0f}s")
        
        # 保存最佳
        if episode_reward > history['best_reward']:
            history['best_reward'] = episode_reward
            history['best_episode'] = episode
            agent.save(str(save_dir / 'improved_v2_best.pt'))
        
        # 学习率调度
        scheduler_actor.step()
        
        # 定期保存
        if (episode + 1) % 100 == 0:
            agent.save(str(save_dir / f'improved_v2_ep{episode+1}.pt'))
    
    # 最终保存
    agent.save(str(save_dir / 'improved_v2_final.pt'))
    
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print("训练完成!")
    print(f"总时间: {total_time:.1f}s")
    print(f"最佳奖励: {history['best_reward']:.2f} (回合 {history['best_episode']})")
    print(f"最终平均奖励: {np.mean(history['episode_rewards'][-50:]):.2f}")
    print("="*70)
    
    return agent, history


if __name__ == "__main__":
    train_improved_v2()
