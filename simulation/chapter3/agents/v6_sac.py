"""
V6 SAC - 大网络训练方案（7.5M参数）
目标：成功训练大网络，达到>6500奖励
关键：预训练 + 渐进课程 + 大量数据
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np


class V6Actor(nn.Module):
    """V6 Actor - 大网络，使用Pre-LayerNorm（更稳定）"""
    
    def __init__(self, state_dim, action_dim, 
                 hidden_dims=[768, 768, 512, 512],
                 log_std_min=-20, log_std_max=2,
                 dropout=0.1):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Pre-LayerNorm: 先归一化再线性变换（更稳定）
        self.input_ln = nn.LayerNorm(state_dim)
        self.input_layer = nn.Linear(state_dim, hidden_dims[0])
        nn.init.orthogonal_(self.input_layer.weight, gain=1.0)
        
        # 隐藏层
        self.ln1 = nn.LayerNorm(hidden_dims[0])
        self.fc1 = nn.Linear(hidden_dims[0], hidden_dims[1])
        nn.init.orthogonal_(self.fc1.weight, gain=1.0)
        
        self.ln2 = nn.LayerNorm(hidden_dims[1])
        self.fc2 = nn.Linear(hidden_dims[1], hidden_dims[2])
        nn.init.orthogonal_(self.fc2.weight, gain=1.0)
        
        self.ln3 = nn.LayerNorm(hidden_dims[2])
        self.fc3 = nn.Linear(hidden_dims[2], hidden_dims[3])
        nn.init.orthogonal_(self.fc3.weight, gain=1.0)
        
        # 残差连接（带缩放）
        self.residual1 = nn.Linear(hidden_dims[0], hidden_dims[1])
        nn.init.orthogonal_(self.residual1.weight, gain=0.5)
        self.residual2 = nn.Linear(hidden_dims[1], hidden_dims[2])
        nn.init.orthogonal_(self.residual2.weight, gain=0.5)
        self.residual3 = nn.Linear(hidden_dims[2], hidden_dims[3])
        nn.init.orthogonal_(self.residual3.weight, gain=0.5)
        
        self.dropout = nn.Dropout(dropout)
        
        # 输出层（小初始化）
        self.mean = nn.Linear(hidden_dims[3], action_dim)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        self.log_std = nn.Linear(hidden_dims[3], action_dim)
        nn.init.orthogonal_(self.log_std.weight, gain=0.01)
    
    def forward(self, state):
        # Pre-LayerNorm
        x = self.input_ln(state)
        x = self.input_layer(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 1
        residual = self.residual1(x)
        x = self.ln1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x + residual * 0.5  # 缩放残差
        
        # Layer 2
        residual = self.residual2(x)
        x = self.ln2(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x + residual * 0.5
        
        # Layer 3
        residual = self.residual3(x)
        x = self.ln3(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = x + residual * 0.5
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state, deterministic=False):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        
        log_prob = normal.log_prob(z)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        if deterministic:
            action = torch.tanh(mean)
        
        return action, log_prob


class V6Critic(nn.Module):
    """V6 Critic - Pre-LayerNorm"""
    
    def __init__(self, state_dim, action_dim,
                 hidden_dims=[768, 768, 512, 512],
                 dropout=0.1):
        super().__init__()
        
        input_dim = state_dim + action_dim
        
        # Pre-LayerNorm
        self.input_ln = nn.LayerNorm(input_dim)
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        nn.init.orthogonal_(self.input_layer.weight, gain=1.0)
        
        # 隐藏层
        self.ln1 = nn.LayerNorm(hidden_dims[0])
        self.fc1 = nn.Linear(hidden_dims[0], hidden_dims[1])
        nn.init.orthogonal_(self.fc1.weight, gain=1.0)
        
        self.ln2 = nn.LayerNorm(hidden_dims[1])
        self.fc2 = nn.Linear(hidden_dims[1], hidden_dims[2])
        nn.init.orthogonal_(self.fc2.weight, gain=1.0)
        
        self.ln3 = nn.LayerNorm(hidden_dims[2])
        self.fc3 = nn.Linear(hidden_dims[2], hidden_dims[3])
        nn.init.orthogonal_(self.fc3.weight, gain=1.0)
        
        # 残差连接
        self.residual1 = nn.Linear(hidden_dims[0], hidden_dims[1])
        nn.init.orthogonal_(self.residual1.weight, gain=0.5)
        self.residual2 = nn.Linear(hidden_dims[1], hidden_dims[2])
        nn.init.orthogonal_(self.residual2.weight, gain=0.5)
        self.residual3 = nn.Linear(hidden_dims[2], hidden_dims[3])
        nn.init.orthogonal_(self.residual3.weight, gain=0.5)
        
        self.dropout = nn.Dropout(dropout)
        
        self.q = nn.Linear(hidden_dims[3], 1)
        nn.init.orthogonal_(self.q.weight, gain=1.0)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        
        x = self.input_ln(x)
        x = self.input_layer(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 1
        residual = self.residual1(x)
        x = self.ln1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x + residual * 0.5
        
        # Layer 2
        residual = self.residual2(x)
        x = self.ln2(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x + residual * 0.5
        
        # Layer 3
        residual = self.residual3(x)
        x = self.ln3(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = x + residual * 0.5
        
        return self.q(x)


class ReplayBuffer:
    """大回放缓冲区 - 500万"""
    
    def __init__(self, state_dim, action_dim, capacity=5000000):
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


class V6SAC:
    """V6 SAC - 大网络版本"""
    
    def __init__(self, state_dim=9, action_dim=3,
                 hidden_dims=[768, 768, 512, 512],
                 lr_actor=5e-5,  # 更低学习率
                 lr_critic=1e-4,
                 lr_alpha=5e-5,
                 gamma=0.999,  # 更高gamma
                 tau=0.001,    # 更慢更新
                 grad_clip=0.5,  # 更严格梯度裁剪
                 dropout=0.1,
                 auto_entropy_tuning=True,
                 target_entropy=None,
                 action_scale=5000.0,
                 device=None):
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.grad_clip = grad_clip
        self.action_scale = action_scale
        self.auto_entropy_tuning = auto_entropy_tuning
        
        self.lr_actor_init = lr_actor
        self.lr_critic_init = lr_critic
        self.lr_alpha_init = lr_alpha
        
        # 网络
        self.actor = V6Actor(state_dim, action_dim, hidden_dims, dropout=dropout).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor, eps=1e-4)
        
        self.critic1 = V6Critic(state_dim, action_dim, hidden_dims, dropout=dropout).to(self.device)
        self.critic2 = V6Critic(state_dim, action_dim, hidden_dims, dropout=dropout).to(self.device)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr_critic, eps=1e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr_critic, eps=1e-4)
        
        # Target网络
        self.critic1_target = V6Critic(state_dim, action_dim, hidden_dims, dropout=dropout).to(self.device)
        self.critic2_target = V6Critic(state_dim, action_dim, hidden_dims, dropout=dropout).to(self.device)
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
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha, eps=1e-4)
        else:
            self.alpha = torch.tensor(0.2, device=self.device)
        
        # 大回放缓冲区
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, capacity=5000000)
        
        self.training_step = 0
        self._print_network_info()
    
    def _print_network_info(self):
        actor_params = sum(p.numel() for p in self.actor.parameters())
        critic_params = sum(p.numel() for p in self.critic1.parameters())
        print(f"V6 SAC (大网络专用版):")
        print(f"  Actor parameters: {actor_params:,}")
        print(f"  Critic parameters: {critic_params:,}")
        print(f"  Total parameters: {actor_params + critic_params * 2:,}")
        print(f"  Gamma: {self.gamma}, Tau: {self.tau}, GradClip: {self.grad_clip}")
        print(f"  LR: actor={self.lr_actor_init:.2e}, critic={self.lr_critic_init:.2e}")
    
    def adjust_lr(self, factor=0.8):
        """调整学习率"""
        self.lr_actor_init *= factor
        self.lr_critic_init *= factor
        self.lr_alpha_init *= factor
        
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = self.lr_actor_init
        for param_group in self.critic1_optimizer.param_groups:
            param_group['lr'] = self.lr_critic_init
        for param_group in self.critic2_optimizer.param_groups:
            param_group['lr'] = self.lr_critic_init
        for param_group in self.alpha_optimizer.param_groups:
            param_group['lr'] = self.lr_alpha_init
    
    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _ = self.actor.sample(state_tensor, deterministic)
            return action.cpu().numpy()[0]
    
    def update(self, batch_size=2048, n_updates=4):  # 每次多更新几次
        """更新网络 - 大网络需要更多更新"""
        if len(self.replay_buffer) < batch_size:
            return None
        
        results = []
        for _ in range(n_updates):
            # 采样
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
                alpha_val = self.alpha.item() if isinstance(self.alpha, torch.Tensor) else self.alpha
                q_next = torch.min(q1_next, q2_next) - alpha_val * next_log_probs
                q_target = rewards + (1 - dones) * self.gamma * q_next
            
            # 更新Critic
            q1 = self.critic1(states, actions)
            q2 = self.critic2(states, actions)
            
            critic1_loss = F.mse_loss(q1, q_target)
            critic2_loss = F.mse_loss(q2, q_target)
            
            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.grad_clip)
            self.critic1_optimizer.step()
            
            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.grad_clip)
            self.critic2_optimizer.step()
            
            # 更新Actor
            new_actions, log_probs = self.actor.sample(states)
            q1_new = self.critic1(states, new_actions)
            q2_new = self.critic2(states, new_actions)
            q_new = torch.min(q1_new, q2_new)
            
            alpha_val = self.alpha.item() if isinstance(self.alpha, torch.Tensor) else self.alpha
            actor_loss = (alpha_val * log_probs - q_new).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            self.actor_optimizer.step()
            
            # 更新Alpha
            if self.auto_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp()
            
            # 软更新
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)
            
            self.training_step += 1
            
            results.append({
                'critic1_loss': critic1_loss.item(),
                'critic2_loss': critic2_loss.item(),
                'actor_loss': actor_loss.item(),
                'alpha': self.alpha.item()
            })
        
        return results[-1]  # 返回最后一次的结果
    
    def _soft_update(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def save(self, filepath):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'alpha': self.alpha.item() if isinstance(self.alpha, torch.Tensor) else self.alpha,
            'training_step': self.training_step,
            'lr_actor': self.lr_actor_init,
            'lr_critic': self.lr_critic_init,
            'lr_alpha': self.lr_alpha_init
        }, filepath)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        if 'training_step' in checkpoint:
            self.training_step = checkpoint['training_step']
        if 'lr_critic' in checkpoint:
            self.lr_critic_init = checkpoint['lr_critic']
            self.lr_actor_init = checkpoint['lr_actor']
            self.lr_alpha_init = checkpoint['lr_alpha']


if __name__ == "__main__":
    print("Testing V6 SAC (大网络专用)...")
    sac = V6SAC(state_dim=9, action_dim=3)
    
    state = np.random.randn(9)
    action = sac.select_action(state)
    print(f"State: {state[:3]}")
    print(f"Action: {action}")
    
    # 填充缓冲区
    for i in range(5000):
        s = np.random.randn(9)
        a = np.random.randn(3)
        r = np.random.randn()
        s_next = np.random.randn(9)
        done = 0.0
        sac.replay_buffer.add(s, a, r, s_next, done)
    
    result = sac.update(batch_size=2048, n_updates=4)
    print(f"Update result: {result}")
    
    print("\n✓ V6 SAC测试通过！准备训练大网络...")
