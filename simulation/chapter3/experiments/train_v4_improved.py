"""
V4 Improved 训练脚本
使用ISE惩罚+改进终止代替ISE累积误差
"""

import sys
import os
import importlib.util
import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from typing import Tuple, Dict

BASE_DIR = '/home/haydn/Documents/AERAOFMINE/mythesis/simulation/chapter3'
sys.path.insert(0, os.path.join(BASE_DIR, 'common'))

# 加载滑动窗口环境
env_spec = importlib.util.spec_from_file_location('rl_env_v4_improved', 
    os.path.join(BASE_DIR, 'env', 'rl_env_v4_improved.py'))
env_module = importlib.util.module_from_spec(env_spec)
env_spec.loader.exec_module(env_module)
PlatformRLEnvV4Improved = env_module.PlatformRLEnvV4Improved


# ==================== V4SAC网络架构 ====================

class V4Actor(nn.Module):
    def __init__(self, state_dim, action_dim, 
                 hidden_dims=[512, 512, 256, 256],
                 log_std_min=-20, log_std_max=2,
                 dropout=0.1):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        layers = []
        prev_dim = state_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(dropout) if dropout > 0 else nn.Identity())
            prev_dim = hidden_dim
        
        self.layers = nn.ModuleList(layers)
        
        self.residual_proj = nn.ModuleList()
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            if prev_dim != hidden_dim:
                self.residual_proj.append(nn.Linear(prev_dim, hidden_dim))
            else:
                self.residual_proj.append(nn.Identity())
            prev_dim = hidden_dim
        
        self.mean = nn.Linear(prev_dim, action_dim)
        self.log_std = nn.Linear(prev_dim, action_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, state):
        x = state
        for i in range(0, len(self.layers), 3):
            linear = self.layers[i]
            norm = self.layers[i+1]
            dropout = self.layers[i+2]
            residual = self.residual_proj[i//3](x)
            x = linear(x)
            x = norm(x)
            x = F.relu(x)
            x = dropout(x)
            x = x + residual
        
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


class V4Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[512, 512, 256, 256], dropout=0.1):
        super().__init__()
        
        layers1 = []
        prev_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            layers1.append(nn.Linear(prev_dim, hidden_dim))
            layers1.append(nn.LayerNorm(hidden_dim))
            layers1.append(nn.Dropout(dropout) if dropout > 0 else nn.Identity())
            prev_dim = hidden_dim
        layers1.append(nn.Linear(prev_dim, 1))
        self.Q1 = nn.Sequential(*layers1)
        
        layers2 = []
        prev_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            layers2.append(nn.Linear(prev_dim, hidden_dim))
            layers2.append(nn.LayerNorm(hidden_dim))
            layers2.append(nn.Dropout(dropout) if dropout > 0 else nn.Identity())
            prev_dim = hidden_dim
        layers2.append(nn.Linear(prev_dim, 1))
        self.Q2 = nn.Sequential(*layers2)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.Q1(sa), self.Q2(sa)


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int = 2000000):
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
    
    def sample(self, batch_size: int) -> Tuple:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.states[idxs]),
            torch.FloatTensor(self.actions[idxs]),
            torch.FloatTensor(self.rewards[idxs]),
            torch.FloatTensor(self.next_states[idxs]),
            torch.FloatTensor(self.dones[idxs])
        )


class V4SACAgent:
    def __init__(self, state_dim, action_dim, device='cuda'):
        self.device = torch.device(device)
        self.gamma = 0.995  # 与V4 Optimized一致
        self.tau = 0.002    # 与V4 Optimized一致
        
        self.actor = V4Actor(state_dim, action_dim, 
                           hidden_dims=[512, 512, 256, 256],
                           dropout=0.1).to(self.device)
        self.critic = V4Critic(state_dim, action_dim,
                              hidden_dims=[512, 512, 256, 256],
                              dropout=0.1).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
    
    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _ = self.actor.sample(state_t, deterministic)
            return action.cpu().numpy()[0]
    
    def update(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.log_alpha.exp() * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        q1, q2 = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q1, target_q) + nn.MSELoss()(q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.log_alpha.exp() * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        alpha_loss = -(self.log_alpha.exp() * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, path)
    
    def load(self, path, device='cuda'):
        checkpoint = torch.load(path, map_location=device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])


def evaluate_policy(env, agent, n_episodes=3, deterministic=True):
    episode_rewards = []
    episode_steps = []
    
    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done:
            action = agent.select_action(state, deterministic=deterministic)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
            step += 1
            if step >= env.max_episode_steps:
                break
        
        episode_rewards.append(episode_reward)
        episode_steps.append(step)
    
    return np.mean(episode_rewards), np.mean(episode_steps)


class V4SlidingConfig:
    def __init__(self):
        # Stage 1: 20秒
        self.stage1 = {
            'name': 'Stage1',
            'episode_length': 20,
            'max_episodes': 200,
            'Hs': 1.0,
            'T1': 8.0,
            'q_des_type': 'constant',
            'save_dir': 'models_v4_improved/stage1',
            'actor_lr': 1e-4,
            'critic_lr': 3e-4,
            'patience': 30,
            'sliding_window': 50
        }
        
        # Stage 2: 30秒
        self.stage2 = {
            'name': 'Stage2',
            'episode_length': 30,
            'max_episodes': 250,
            'Hs': 2.0,
            'T1': 8.0,
            'q_des_type': 'sinusoidal',
            'save_dir': 'models_v4_improved/stage2',
            'actor_lr': 1e-4,
            'critic_lr': 3e-4,
            'patience': 40,
            'sliding_window': 50
        }
        
        # Stage 3: 40秒
        self.stage3 = {
            'name': 'Stage3',
            'episode_length': 40,
            'max_episodes': 300,
            'Hs': 2.0,
            'T1': 8.0,
            'q_des_type': 'sinusoidal',
            'save_dir': 'models_v4_improved/stage3',
            'actor_lr': 5e-5,
            'critic_lr': 2e-4,
            'patience': 50,
            'sliding_window': 50
        }


def train_stage(agent, env_config, stage_name, load_model=None, device='cuda'):
    save_dir = Path(env_config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"训练 {stage_name} (滑动窗口)")
    print(f"{'='*70}")
    print(f"回合长度: {env_config['episode_length']}秒")
    print(f"最大回合: {env_config['max_episodes']}")
    print(f"海浪Hs: {env_config['Hs']}m")
    print(f"轨迹类型: {env_config['q_des_type']}")
        
    max_steps = int(env_config['episode_length'] / 0.01)
    env = PlatformRLEnvV4Improved(
        use_model_compensation=True,
        max_episode_steps=max_steps,
        Hs=env_config['Hs'],
        T1=env_config['T1'],
        q_des_type=env_config['q_des_type'],
            )
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    if load_model:
        print(f"加载预训练模型: {load_model}")
        agent.load(load_model, device)
        for param_group in agent.actor_optimizer.param_groups:
            param_group['lr'] = env_config['actor_lr']
        for param_group in agent.critic_optimizer.param_groups:
            param_group['lr'] = env_config['critic_lr']
    
    replay_buffer = ReplayBuffer(state_dim, action_dim, capacity=2000000)
    
    episode_rewards = []
    eval_rewards = []
    best_eval_reward = -float('inf')
    best_episode = 0
    no_improve_count = 0
    
    batch_size = 512
    warmup_steps = 5000  # 与V4 Optimized一致
    
    start_time = time.time()
    
    for episode in range(env_config['max_episodes']):
        
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done:
            if replay_buffer.size < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, deterministic=False)  # SAC自带探索
            
            next_state, reward, done, info = env.step(action)
            replay_buffer.add(state, action, reward, next_state, float(done))
            
            if replay_buffer.size >= warmup_steps:
                batch = replay_buffer.sample(batch_size)
                agent.update(batch)
            
            episode_reward += reward
            state = next_state
            step += 1
            
            if step >= env.max_episode_steps:
                break
        
        episode_rewards.append(episode_reward)
        
        eval_interval = 25
        if (episode + 1) % eval_interval == 0:
            eval_reward, eval_steps = evaluate_policy(env, agent, n_episodes=3)
            eval_rewards.append(eval_reward)
            
            elapsed = time.time() - start_time
            avg_train = np.mean(episode_rewards[-eval_interval:])
            
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                best_episode = episode + 1
                no_improve_count = 0
                best_path = save_dir / f'{stage_name}_best.pt'
                agent.save(str(best_path))
            else:
                no_improve_count += eval_interval
            
            print(f"Ep {episode+1:3d} | Train={episode_reward:8.2f} | "
                  f"Avg={avg_train:8.2f} | Eval={eval_reward:8.2f} | "
                  f"Steps={eval_steps:4.0f} | Best={best_eval_reward:8.2f} | "
                  f"Time={elapsed/60:.1f}m")
            
            if no_improve_count >= env_config['patience']:
                print(f"早停: {env_config['patience']}回合无改善")
                break
    
    final_path = save_dir / f'{stage_name}_final.pt'
    agent.save(str(final_path))
    
    total_time = time.time() - start_time
    
    print(f"\n{stage_name} 完成!")
    print(f"  总回合: {len(episode_rewards)}")
    print(f"  最佳评估: {best_eval_reward:.2f} (回合 {best_episode})")
    print(f"  总时间: {total_time/60:.1f}分钟")
    
    return agent, {
        'episode_rewards': episode_rewards,
        'eval_rewards': eval_rewards,
        'best_reward': best_eval_reward,
        'best_episode': best_episode,
        'total_time': total_time
    }


def compute_ise(env, agent, n_episodes=3):
    ise_list = []
    steps_list = []
    
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        step = 0
        
        while not done:
            action = agent.select_action(state, deterministic=True)
            next_state, reward, done, info = env.step(action)
            state = next_state
            step += 1
            if step >= env.max_episode_steps:
                break
        
        history = env.get_history()
        errors = np.array(history['q']) - np.array(history['q_des'])
        ise = np.sum(errors**2, axis=0) * env.dt
        ise_list.append(ise)
        steps_list.append(step)
    
    avg_ise = np.mean(ise_list, axis=0)
    avg_steps = np.mean(steps_list)
    avg_ise_per_step = avg_ise / max(avg_steps, 1)
    
    return avg_ise, avg_ise_per_step, avg_steps


def main():
    print("=" * 70)
    print("V4 Improved 训练")
    print("使用ISE惩罚+改进终止代替ISE累积误差")
    print("=" * 70)
    
    config = V4SlidingConfig()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n设备: {device}")
    
    env_init = PlatformRLEnvV4Improved(
        use_model_compensation=True,
        max_episode_steps=2000,
        Hs=2.0,
        T1=8.0,
        
    )
    state_dim = env_init.observation_space.shape[0]
    action_dim = env_init.action_space.shape[0]
    
    agent = V4SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device
    )
    
    total_start = time.time()
    all_results = []
    
    # Stage 1
    agent, history1 = train_stage(
        agent, config.stage1, 'Stage1', 
        load_model=None, device=device
    )
    all_results.append(('Stage1', history1['best_reward']))
    
    # Stage 2
    agent, history2 = train_stage(
        agent, config.stage2, 'Stage2',
        load_model=f"{config.stage1['save_dir']}/Stage1_best.pt",
        device=device
    )
    all_results.append(('Stage2', history2['best_reward']))
    
    # Stage 3
    agent, history3 = train_stage(
        agent, config.stage3, 'Stage3',
        load_model=f"{config.stage2['save_dir']}/Stage2_best.pt",
        device=device
    )
    all_results.append(('Stage3', history3['best_reward']))
    
    total_time = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)
    print(f"总时间: {total_time/60:.1f}分钟")
    
    print("\n各阶段最佳评估奖励:")
    for stage_name, best_reward in all_results:
        print(f"  {stage_name}: {best_reward:.2f}")
    
    print("\n" + "=" * 70)
    print("最终模型评估 (ISE/step)")
    print("=" * 70)
    
    final_model_path = Path(config.stage3['save_dir']) / 'Stage3_best.pt'
    agent.load(str(final_model_path), device)
    
    env_final = PlatformRLEnvV4Improved(
        use_model_compensation=True,
        max_episode_steps=4000,
        Hs=2.0,
        T1=8.0,
        q_des_type='sinusoidal',
        
    )
    
    ise, ise_per_step, avg_steps = compute_ise(env_final, agent, n_episodes=5)
    
    print(f"平均回合步数: {avg_steps:.0f}")
    print(f"ISE/step:")
    print(f"  z:   {ise_per_step[0]:.6f}")
    print(f"  alpha: {ise_per_step[1]:.6f}")
    print(f"  beta:  {ise_per_step[2]:.6f}")
    print(f"  总:  {np.sum(ise_per_step):.6f}")
    
    print("\n对比:")
    print(f"  V4 Optimized ISE/step: 0.000745")
    print(f"  V4 Fixed v2 ISE/step:  0.000885")
    print(f"  V4 Sliding ISE/step:   {np.sum(ise_per_step):.6f}")


if __name__ == "__main__":
    main()
