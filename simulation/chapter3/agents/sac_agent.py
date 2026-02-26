"""
第3章：SAC (Soft Actor-Critic) 强化学习算法实现
基于PyTorch的简化版SAC实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import Tuple, Dict, List
import copy


# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, state_dim: int, action_dim: int, capacity: int = 1000000):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
    
    def add(self, state, action, reward, next_state, done):
        """添加经验"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple:
        """采样批次"""
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.FloatTensor(self.states[idxs]),
            torch.FloatTensor(self.actions[idxs]),
            torch.FloatTensor(self.rewards[idxs]),
            torch.FloatTensor(self.next_states[idxs]),
            torch.FloatTensor(self.dones[idxs])
        )
    
    def __len__(self):
        return self.size


class ActorNetwork(nn.Module):
    """Actor网络（策略网络）"""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_dim: int = 256, log_std_min: float = -20, 
                 log_std_max: float = 2):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # 共享网络层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 均值输出
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        
        # 标准差对数输出
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        x = self.shared(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """采样动作"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        if deterministic:
            action = torch.tanh(mean)
            log_prob = None
        else:
            # 重参数化技巧
            normal = Normal(mean, std)
            x_t = normal.rsample()  # 重参数化采样
            action = torch.tanh(x_t)
            
            # 计算log概率（考虑tanh变换）
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob


class CriticNetwork(nn.Module):
    """Critic网络（Q函数）"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2（Double Q-learning）
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        x = torch.cat([state, action], dim=1)
        q1 = self.q1(x)
        q2 = self.q2(x)
        return q1, q2
    
    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """只返回Q1（用于策略更新）"""
        x = torch.cat([state, action], dim=1)
        return self.q1(x)


class SACAgent:
    """
    SAC (Soft Actor-Critic) Agent
    
    特点：
    - 最大熵强化学习（平衡探索和利用）
    - 重参数化技巧
    - Double Q-learning
    - 自动温度系数调整
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 lr_actor: float = 3e-4,
                 lr_critic: float = 3e-4,
                 lr_alpha: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 automatic_entropy_tuning: bool = True,
                 target_entropy: float = None,
                 device: str = 'cpu'):
        """
        初始化SAC智能体
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
            lr_actor: Actor学习率
            lr_critic: Critic学习率
            lr_alpha: 温度系数学习率
            gamma: 折扣因子
            tau: 软更新系数
            alpha: 初始温度系数（如果自动调整则忽略）
            automatic_entropy_tuning: 是否自动调整温度系数
            target_entropy: 目标熵（默认-action_dim）
            device: 计算设备
        """
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.automatic_entropy_tuning = automatic_entropy_tuning
        
        # Actor网络
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Critic网络
        self.critic = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 目标Critic网络（软更新）
        self.critic_target = copy.deepcopy(self.critic)
        
        # 温度系数alpha
        if automatic_entropy_tuning:
            self.target_entropy = target_entropy if target_entropy is not None else -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
        else:
            self.alpha = torch.tensor(alpha, device=self.device)
        
        # 训练统计
        self.train_info = {
            'actor_loss': [],
            'critic_loss': [],
            'alpha_loss': [],
            'alpha_values': [],
            'entropy': []
        }
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """选择动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _ = self.actor.sample(state_tensor, deterministic)
            action = action.cpu().numpy()[0]
        return action
    
    def update(self, batch: Tuple, update_alpha: bool = True) -> Dict:
        """
        更新网络参数
        
        Returns:
            dict: 训练信息
        """
        states, actions, rewards, next_states, dones = batch
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # =============== Critic Update ===============
        with torch.no_grad():
            # 采样下一动作
            next_actions, next_log_probs = self.actor.sample(next_states)
            
            # 目标Q值
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # 当前Q值
        current_q1, current_q2 = self.critic(states, actions)
        
        # Critic损失
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # =============== Actor Update ===============
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        # Actor损失（最小化 -Q + alpha * log_prob）
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        # 更新Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # =============== Alpha Update ===============
        alpha_loss = torch.tensor(0.0)
        if self.automatic_entropy_tuning and update_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
        
        # =============== Soft Update Target Networks ===============
        self._soft_update(self.critic, self.critic_target)
        
        # 记录训练信息
        info = {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'alpha_loss': alpha_loss.item() if isinstance(alpha_loss, torch.Tensor) else 0.0,
            'alpha_values': self.alpha.item() if isinstance(self.alpha, torch.Tensor) else self.alpha,
            'entropy': -log_probs.mean().item()
        }
        
        # 保存历史
        for key, value in info.items():
            self.train_info[key].append(value)
        
        return info
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """软更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'alpha': self.alpha if not self.automatic_entropy_tuning else self.log_alpha,
            'train_info': self.train_info
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        if self.automatic_entropy_tuning:
            self.log_alpha = checkpoint['alpha']
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = checkpoint['alpha']
        
        self.train_info = checkpoint['train_info']
        print(f"Model loaded from {filepath}")
    
    def get_train_info(self) -> Dict:
        """获取训练统计信息"""
        return self.train_info


def train_sac(env, agent: SACAgent, replay_buffer: ReplayBuffer,
              max_episodes: int = 1000, max_steps: int = 1000,
              batch_size: int = 256, update_interval: int = 1,
              warmup_steps: int = 1000, eval_interval: int = 100,
              save_interval: int = 500, save_path: str = None) -> Dict:
    """
    训练SAC智能体
    
    Args:
        env: 环境
        agent: SAC智能体
        replay_buffer: 经验回放缓冲区
        max_episodes: 最大训练回合数
        max_steps: 每回合最大步数
        batch_size: 批次大小
        update_interval: 更新间隔（每多少步更新一次）
        warmup_steps: 预热步数（随机探索）
        eval_interval: 评估间隔
        save_interval: 保存间隔
        save_path: 模型保存路径
    
    Returns:
        dict: 训练历史
    """
    print("="*70)
    print("SAC Training Started")
    print("="*70)
    
    episode_rewards = []
    eval_rewards = []
    step_count = 0
    
    for episode in range(max_episodes):
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
            replay_buffer.add(state, action, reward, next_state, float(done))
            
            episode_reward += reward
            state = next_state
            step_count += 1
            
            # 更新网络
            if step_count >= warmup_steps and step_count % update_interval == 0:
                if len(replay_buffer) >= batch_size:
                    batch = replay_buffer.sample(batch_size)
                    agent.update(batch)
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # 打印进度
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode+1}/{max_episodes}, "
                  f"Avg Reward (last 10): {avg_reward:.2f}, "
                  f"Steps: {step_count}")
        
        # 评估
        if (episode + 1) % eval_interval == 0:
            eval_reward = evaluate_agent(env, agent, n_episodes=3, max_steps=max_steps)
            eval_rewards.append(eval_reward)
            print(f"  [Eval] Episode {episode+1}, Eval Reward: {eval_reward:.2f}")
        
        # 保存模型
        if save_path and (episode + 1) % save_interval == 0:
            agent.save(f"{save_path}_episode_{episode+1}.pt")
    
    # 最终保存
    if save_path:
        agent.save(f"{save_path}_final.pt")
    
    print("="*70)
    print("Training Completed!")
    print("="*70)
    
    return {
        'episode_rewards': episode_rewards,
        'eval_rewards': eval_rewards,
        'train_info': agent.get_train_info()
    }


def evaluate_agent(env, agent: SACAgent, n_episodes: int = 5, 
                   max_steps: int = 1000) -> float:
    """评估智能体"""
    total_reward = 0
    
    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        
        for _ in range(max_steps):
            action = agent.select_action(state, deterministic=True)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            if done:
                break
        
        total_reward += episode_reward
    
    return total_reward / n_episodes


if __name__ == "__main__":
    # 测试SAC Agent
    print("Testing SAC Agent...")
    
    state_dim = 9
    action_dim = 3
    
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        automatic_entropy_tuning=True,
        device='cpu'
    )
    
    # 测试前向传播
    state = np.random.randn(state_dim)
    action = agent.select_action(state)
    print(f"State shape: {state.shape}")
    print(f"Action shape: {action.shape}")
    print(f"Action range: [{action.min():.3f}, {action.max():.3f}]")
    
    # 测试更新
    replay_buffer = ReplayBuffer(state_dim, action_dim, capacity=10000)
    
    # 填充一些随机经验
    for _ in range(1000):
        s = np.random.randn(state_dim)
        a = np.random.randn(action_dim)
        r = np.random.randn()
        s_next = np.random.randn(state_dim)
        d = 0.0
        replay_buffer.add(s, a, r, s_next, d)
    
    # 更新一次
    batch = replay_buffer.sample(256)
    info = agent.update(batch)
    
    print("\nUpdate Info:")
    for key, value in info.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nSAC Agent test passed!")
