"""
高级SAC实现 - V3版本
网络架构: 512x2 + 256x2 (深度前馈网络)
改进:
1. 更大的网络容量 (~500K参数)
2. LayerNorm稳定训练
3. Xavier初始化
4. 支持课程学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np


class AdvancedActor(nn.Module):
    """
    Actor网络 - 512x2 + 256x2
    输入: 状态 (state_dim)
    输出: 动作均值和对数标准差
    """
    
    def __init__(self, state_dim, action_dim, 
                 hidden_dims=[512, 512, 256, 256],
                 log_std_min=-20, log_std_max=2):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # 构建网络层
        layers = []
        prev_dim = state_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        
        self.layers = nn.ModuleList(layers)
        
        # 输出层
        self.mean = nn.Linear(prev_dim, action_dim)
        self.log_std = nn.Linear(prev_dim, action_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, state):
        x = state
        
        # 通过隐藏层
        for i in range(0, len(self.layers), 2):
            x = F.relu(self.layers[i+1](self.layers[i](x)))
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
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


class AdvancedCritic(nn.Module):
    """
    Critic网络 (Q网络) - 512x2 + 256x2
    输入: 状态 + 动作
    输出: Q值
    """
    
    def __init__(self, state_dim, action_dim,
                 hidden_dims=[512, 512, 256, 256]):
        super().__init__()
        
        # 构建网络层
        layers = []
        prev_dim = state_dim + action_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        
        self.layers = nn.ModuleList(layers)
        
        # 输出层
        self.q = nn.Linear(prev_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        
        # 通过隐藏层
        for i in range(0, len(self.layers), 2):
            x = F.relu(self.layers[i+1](self.layers[i](x)))
        
        return self.q(x)


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, state_dim, action_dim, capacity=1000000):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
    
    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.FloatTensor(self.states[idx]),
            torch.FloatTensor(self.actions[idx]),
            torch.FloatTensor(self.rewards[idx]),
            torch.FloatTensor(self.next_states[idx]),
            torch.FloatTensor(self.dones[idx])
        )
    
    def __len__(self):
        return self.size


class AdvancedSAC:
    """
    高级SAC实现
    支持课程学习和长回合训练
    """
    
    def __init__(self, state_dim=9, action_dim=3,
                 hidden_dims=[512, 512, 256, 256],
                 lr_actor=3e-4, lr_critic=1e-3, lr_alpha=3e-4,
                 gamma=0.99, tau=0.005,
                 auto_entropy_tuning=True,
                 target_entropy=None,
                 action_scale=5000.0,
                 device='cpu'):
        
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.action_scale = action_scale
        self.auto_entropy_tuning = auto_entropy_tuning
        
        # Actor网络
        self.actor = AdvancedActor(state_dim, action_dim, hidden_dims).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Twin Critic网络
        self.critic1 = AdvancedCritic(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic2 = AdvancedCritic(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr_critic)
        
        # Target网络
        self.critic1_target = AdvancedCritic(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic2_target = AdvancedCritic(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # 熵温度
        if auto_entropy_tuning:
            if target_entropy is None:
                self.target_entropy = -action_dim
            else:
                self.target_entropy = target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
        else:
            self.alpha = torch.tensor(0.2, device=self.device)
        
        # 回放缓冲区
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, capacity=1000000)
        
        # 训练统计
        self.training_step = 0
    
    def select_action(self, state, deterministic=False):
        """选择动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _ = self.actor.sample(state_tensor, deterministic)
            return action.cpu().numpy()[0]
    
    def update(self, batch_size=512):
        """更新网络"""
        if len(self.replay_buffer) < batch_size:
            return None
        
        # 采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # 计算Target Q值
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next = self.critic1_target(next_states, next_actions)
            q2_next = self.critic2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + (1 - dones) * self.gamma * q_next
        
        # 更新Critic
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
        
        # 更新Actor
        new_actions, log_probs = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # 更新Alpha
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        
        # 软更新Target网络
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)
        
        self.training_step += 1
        
        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha.item()
        }
    
    def _soft_update(self, source, target):
        """软更新"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def save(self, filepath):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'alpha': self.alpha.item() if isinstance(self.alpha, torch.Tensor) else self.alpha,
            'training_step': self.training_step
        }, filepath)
    
    def load(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        if 'training_step' in checkpoint:
            self.training_step = checkpoint['training_step']


if __name__ == "__main__":
    # 测试
    print("Testing Advanced SAC...")
    sac = AdvancedSAC(state_dim=9, action_dim=3)
    
    # 测试网络
    state = np.random.randn(9)
    action = sac.select_action(state)
    print(f"State: {state[:3]}")
    print(f"Action: {action}")
    
    # 测试更新
    for i in range(1000):
        s = np.random.randn(9)
        a = np.random.randn(3)
        r = np.random.randn()
        s_next = np.random.randn(9)
        done = 0.0
        sac.replay_buffer.add(s, a, r, s_next, done)
    
    result = sac.update(batch_size=512)
    print(f"Update result: {result}")
    print("Advanced SAC test passed!")
