"""
Chapter 4 训练脚本  —— 课程学习版
V5 + STESO + 硬切换 DRARL

课程分三阶段：
  Stage 1 (ep 0~299):   小振幅轨迹，弱扰动 Hs=1.0，让智能体先学会基本跟踪
  Stage 2 (ep 300~799): 正常振幅，正常扰动 Hs=2.0
  Stage 3 (ep 800~1499): 正常振幅，随机强扰动 Hs∈[1.5, 3.5]，提升鲁棒性
"""

import sys
import os

PROJECT_ROOT = '/home/haydn/Documents/AERAOFMINE/mythesis/simulation'
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'common'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'disturbance'))
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


# ── 课程阶段定义 ──────────────────────────────────────────────────────────────
CURRICULUM = [
    # stage, ep_start, ep_end, Hs_range, T1, q_des_type, diverge_thr
    (1,    0,  299, (1.0, 1.0),    8.0, 'sinusoidal_small', 0.6),
    (2,  300,  799, (2.0, 2.0),    8.0, 'sinusoidal',       0.5),
    (3,  800, 1499, (1.5, 3.5),    8.0, 'sinusoidal',       0.5),
]


def get_stage(episode: int):
    for stage, ep_start, ep_end, Hs_range, T1, q_des_type, div_thr in CURRICULUM:
        if ep_start <= episode <= ep_end:
            return stage, Hs_range, T1, q_des_type, div_thr
    # 超出范围默认最后阶段
    s = CURRICULUM[-1]
    return s[0], s[3], s[4], s[5], s[6]


def make_env(Hs: float, T1: float, q_des_type: str, diverge_thr: float, dt: float = 0.01,
             episode_length: float = 30.0) -> PlatformRLEnvChapter4:
    return PlatformRLEnvChapter4(
        use_model_compensation=True,
        use_eso=True,
        use_steso=True,
        use_hard_switch=True,
        dt=dt,
        max_episode_steps=int(episode_length / dt),
        Hs=Hs,
        T1=T1,
        q_des_type=q_des_type,
        diverge_threshold=diverge_thr,
        steso_lambda1=4.0,
        steso_beta1=12.0,
        steso_beta2=30.0,
        switch_threshold=0.6,
        switch_beta=0.5,
    )


class Chapter4Trainer:

    def __init__(self, save_dir: str, resume: bool = False):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.resume = resume

        self.dt = 0.01
        self.episode_length = 30.0
        self.max_episodes = 1500
        self.warmup_steps = 8000
        self.batch_size = 256
        self.updates_per_step = 2        # 每步更新次数，加快学习

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.training_history = {
            'variant_name': 'Chapter4_STESO_HSC_Curriculum',
            'start_time': None,
            'end_time': None,
            'episodes': [],
        }

    # ── 智能体构建 ─────────────────────────────────────────────────────────────
    def _build_agent(self, state_dim: int, action_dim: int) -> V4SAC:
        return V4SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[256, 256],          # 与已有 best_model.pt 结构一致
            lr_actor=1e-4,
            lr_critic=3e-4,
            gamma=0.995,
            tau=0.002,
            device=self.device,
        )

    # ── 训练主循环 ─────────────────────────────────────────────────────────────
    def train(self):
        print("\n" + "=" * 70)
        print("Chapter 4 课程学习训练  (STESO + HSC + SAC)")
        print("=" * 70)
        print(f"  总回合: {self.max_episodes}  |  设备: {self.device}")
        print(f"  warm-up: {self.warmup_steps} steps  |  batch: {self.batch_size}"
              f"  |  updates/step: {self.updates_per_step}")

        # 用 Stage 1 环境初始化智能体维度
        init_env = make_env(1.0, 8.0, 'sinusoidal_small', 0.6,
                            self.dt, self.episode_length)
        state_dim  = init_env.observation_space.shape[0]
        action_dim = init_env.action_space.shape[0]
        init_env.close() if hasattr(init_env, 'close') else None

        agent = self._build_agent(state_dim, action_dim)

        # 续训：加载已有最佳模型
        resume_path = self.save_dir / 'best_model.pt'
        if self.resume and resume_path.exists():
            agent.load(str(resume_path))
            print(f"  已加载续训模型: {resume_path}")

        actor_p  = sum(p.numel() for p in agent.actor.parameters())
        critic_p = sum(p.numel() for p in agent.critic1.parameters()) + \
                   sum(p.numel() for p in agent.critic2.parameters())
        print(f"  网络参数量: actor={actor_p:,}  critic={critic_p:,}")

        # ── 训练 ──
        episode_rewards  = []
        best_reward      = -float('inf')
        best_episode     = 0
        step_count       = 0
        prev_stage       = -1
        start_time       = time.time()
        self.training_history['start_time'] = datetime.now().isoformat()

        for episode in range(self.max_episodes):
            stage, Hs_range, T1, q_des_type, div_thr = get_stage(episode)

            # 随机在 Hs_range 内采样
            Hs = float(np.random.uniform(*Hs_range))

            env = make_env(Hs, T1, q_des_type, div_thr, self.dt, self.episode_length)

            if stage != prev_stage:
                print(f"\n{'─'*60}")
                print(f"  ▶ 进入 Stage {stage}  "
                      f"(Hs∈{Hs_range}, {q_des_type})")
                print(f"{'─'*60}")
                prev_stage = stage

            state = env.reset()
            episode_reward = 0.0
            episode_ise    = 0.0
            max_steps = int(self.episode_length / self.dt)

            for step in range(max_steps):
                # 动作选择
                if step_count < self.warmup_steps:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(state, deterministic=False)

                next_state, reward, done, info = env.step(action)
                agent.replay_buffer.add(state, action, reward, next_state, float(done))

                episode_reward += reward
                episode_ise    += info.get('ise', 0.0)

                # 更新
                if (step_count >= self.warmup_steps and
                        len(agent.replay_buffer) >= self.batch_size):
                    for _ in range(self.updates_per_step):
                        agent.update(self.batch_size)

                state = next_state
                step_count += 1
                if done:
                    break

            episode_rewards.append(episode_reward)

            # 保存最佳
            if episode_reward > best_reward:
                best_reward   = episode_reward
                best_episode  = episode
                agent.save(self.save_dir / 'best_model.pt')
                print(f"  ep{episode:5d} [S{stage}] Hs={Hs:.1f} | "
                      f"Reward={episode_reward:9.3f}  ISE={episode_ise*self.dt:.5f}"
                      f"  ← 新最佳")
            elif episode % 50 == 0:
                elapsed = (time.time() - start_time) / 60
                print(f"  ep{episode:5d} [S{stage}] Hs={Hs:.1f} | "
                      f"Reward={episode_reward:9.3f}  ISE={episode_ise*self.dt:.5f}"
                      f"  best={best_reward:.3f}  ({elapsed:.1f}min)")

            # 每 200 回合保存 checkpoint
            if episode > 0 and episode % 200 == 0:
                agent.save(self.save_dir / f'checkpoint_ep{episode}.pt')

            self.training_history['episodes'].append({
                'episode': episode,
                'stage': stage,
                'Hs': Hs,
                'reward': episode_reward,
                'ise': episode_ise * self.dt,
            })

        # 保存最终模型
        agent.save(self.save_dir / 'final_model.pt')

        elapsed = time.time() - start_time
        self.training_history.update({
            'end_time': datetime.now().isoformat(),
            'best_reward': best_reward,
            'best_episode': best_episode,
            'total_time': elapsed,
        })
        with open(self.save_dir / 'training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)

        print(f"\n{'='*70}")
        print(f"训练完成!  最佳 Reward={best_reward:.3f} @ ep{best_episode}")
        print(f"总时长: {elapsed/60:.1f} 分钟")
        print(f"模型保存: {self.save_dir}")
        return agent, self.training_history


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--resume', action='store_true',
                   help='从 best_model.pt 续训')
    p.add_argument('--outdir', default=None,
                   help='模型输出目录，默认 chapter4_models_v2')
    args = p.parse_args()

    out = Path(__file__).parent / (args.outdir or 'chapter4_models_v2')
    trainer = Chapter4Trainer(save_dir=str(out), resume=args.resume)
    trainer.train()

