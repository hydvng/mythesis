"""
第3章 V5 / V5.1 Uniform 控制效果评估与可视化

默认行为：
1. 加载 `models_v5_uniform_v2/stage3/Stage3_V51Uniform_best.pt`
2. 在 30s sinusoidal 跟踪任务上运行 deterministic evaluation
3. 生成控制效果图和相平面图

用法示例：
    python evaluate_v5_uniform.py
    python evaluate_v5_uniform.py --variant v5_uniform
    python evaluate_v5_uniform.py --variant v5_uniform_v2
    python evaluate_v5_uniform.py --model /abs/path/to/model.pt
"""

import sys
import os
import argparse

THIS_DIR = os.path.dirname(__file__)
CH3_DIR = os.path.abspath(os.path.join(THIS_DIR, '..'))
SIM_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))

for p in [
    os.path.join(CH3_DIR, 'env'),
    os.path.join(CH3_DIR, 'agents'),
    os.path.join(SIM_DIR, 'common'),
    os.path.join(SIM_DIR, 'disturbance'),
    SIM_DIR,
]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import warnings

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from advanced_sac import AdvancedSAC
from rl_env_v5_uniform import PlatformRLEnvV5Uniform
from rl_env_v5_uniform_v2 import PlatformRLEnvV5UniformV2


VARIANT_CONFIG = {
    'v5_uniform': {
        'env_cls': PlatformRLEnvV5Uniform,
        'model_path': os.path.join(THIS_DIR, 'models_v5_uniform_quick', 'stage3', 'Stage3_V5Uniform_Quick_best.pt'),
        'episode_length': 30,
        'dt': 0.01,
        'Hs': 2.0,
        'T1': 8.0,
        'q_des_type': 'sinusoidal',
        'enable_burst_step': False,
        'title': 'V5 Uniform',
        'file_tag': 'v5_uniform',
    },
    'v5_uniform_v2': {
        'env_cls': PlatformRLEnvV5UniformV2,
        'model_path': os.path.join(THIS_DIR, 'models_v5_uniform_v2', 'stage3', 'Stage3_V51Uniform_best.pt'),
        'episode_length': 30,
        'dt': 0.01,
        'Hs': 2.0,
        'T1': 8.0,
        'q_des_type': 'sinusoidal',
        'enable_burst_step': False,
        'title': 'V5.1 Uniform',
        'file_tag': 'v5_uniform_v2',
    },
}

FIGURE_DIR = os.path.join(CH3_DIR, 'figures')
os.makedirs(FIGURE_DIR, exist_ok=True)


def build_agent(device):
    return AdvancedSAC(
        state_dim=9,
        action_dim=3,
        hidden_dims=[512, 512, 256, 256],
        lr_actor=3e-4,
        lr_critic=1e-3,
        lr_alpha=3e-4,
        gamma=0.99,
        tau=0.005,
        device=device,
    )


def build_env(cfg):
    max_steps = int(cfg['episode_length'] / cfg['dt'])
    env_cls = cfg['env_cls']
    kwargs = dict(
        use_model_compensation=True,
        dt=cfg['dt'],
        max_episode_steps=max_steps,
        Hs=cfg['Hs'],
        T1=cfg['T1'],
        q_des_type=cfg['q_des_type'],
        diverge_threshold=0.5,
        enable_burst_step=cfg.get('enable_burst_step', False),
    )

    if env_cls is PlatformRLEnvV5UniformV2:
        kwargs.update(dict(
            vel_error_weight=0.10,
            constraint_penalty_weight=2.0,
        ))

    return env_cls(**kwargs)


def run_evaluation(cfg, model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"评估设备: {device}")

    agent = build_agent(device)
    assert os.path.exists(model_path), f"找不到模型: {model_path}"
    agent.load(model_path)
    print(f"已加载模型: {model_path}")

    env = build_env(cfg)
    max_steps = int(cfg['episode_length'] / cfg['dt'])

    state = env.reset()
    total_reward = 0.0
    for _ in range(max_steps):
        action = agent.select_action(state, deterministic=True)
        state, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break

    history = env.get_history()
    print(f"回合完成: steps={len(history['time'])}, total_reward={total_reward:.4f}")
    return history, total_reward


def plot_control_performance(history, total_reward, cfg):
    t = np.array(history['time'])
    q = np.array(history['q'])
    qd = np.array(history['qd'])
    q_des = np.array(history['q_des'])
    u = np.array(history['u'])
    v_rl = np.array(history['v_RL'])
    tau_model = np.array(history['tau_model'])
    rewards = np.array(history['reward'])

    labels_q = ['z (m)', r'$\alpha$ (rad)', r'$\beta$ (rad)']
    labels_u = [r'$F_z$ (N)', r'$\tau_\alpha$ (N·m)', r'$\tau_\beta$ (N·m)']

    fig, axes = plt.subplots(4, 3, figsize=(18, 16), constrained_layout=True)
    fig.suptitle(
        f"{cfg['title']} Control Performance  "
        f"(Hs={cfg['Hs']}m, T1={cfg['T1']}s, {cfg['q_des_type']})\n"
        f"Total Reward = {total_reward:.4f}",
        fontsize=14, fontweight='bold',
    )

    for i in range(3):
        ax = axes[0, i]
        ax.plot(t, q[:, i], 'b-', linewidth=1.2, label='Actual')
        ax.plot(t, q_des[:, i], 'r--', linewidth=1.0, label='Desired')
        ax.set_ylabel(labels_q[i])
        ax.set_title(f'Pose Tracking – {labels_q[i]}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    error = q - q_des
    for i in range(3):
        ax = axes[1, i]
        ax.plot(t, error[:, i], 'k-', linewidth=1.0)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_ylabel(f'Error – {labels_q[i]}')
        ax.set_title(f'Tracking Error – {labels_q[i]}')
        ax.grid(True, alpha=0.3)

    for i in range(3):
        ax = axes[2, i]
        ax.plot(t, u[:, i], 'b-', linewidth=1.0, label='Total u')
        ax.plot(t, tau_model[:, i], 'g--', linewidth=0.8, label='Model comp.')
        ax.plot(t, v_rl[:, i], 'r:', linewidth=0.8, label='RL output')
        ax.set_ylabel(labels_u[i])
        ax.set_title(f'Control Input – {labels_u[i]}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    ax = axes[3, 0]
    ax.plot(t, rewards, 'm-', linewidth=0.8)
    ax.set_ylabel('Reward')
    ax.set_title('Instantaneous Reward')
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)

    ax = axes[3, 1]
    error_norm = np.linalg.norm(error, axis=1)
    ax.plot(t, error_norm, 'k-', linewidth=1.0)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_ylabel('‖e‖')
    ax.set_title('Tracking Error Norm')
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)

    ax = axes[3, 2]
    for i, (lbl, c) in enumerate(zip([r'$\dot{z}$', r'$\dot{\alpha}$', r'$\dot{\beta}$'], ['tab:blue', 'tab:orange', 'tab:green'])):
        ax.plot(t, qd[:, i], color=c, linewidth=0.8, label=lbl)
    ax.set_ylabel('Velocity')
    ax.set_title('Generalized Velocities')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig_path = os.path.join(FIGURE_DIR, f"{cfg['file_tag']}_control_performance.png")
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"控制效果图已保存: {fig_path}")
    plt.close(fig)

    print("\n--- 跟踪误差统计 ---")
    for i, name in enumerate(['z', 'alpha', 'beta']):
        e_i = error[:, i]
        print(f"  {name:>5s}: mean={np.mean(np.abs(e_i)):.6f}, max={np.max(np.abs(e_i)):.6f}, rms={np.sqrt(np.mean(e_i**2)):.6f}")
    print(f"  Overall RMS error norm: {np.sqrt(np.mean(error_norm ** 2)):.6f}")

    return fig_path


def plot_phase_portrait(history, cfg):
    t = np.array(history['time'])
    q = np.array(history['q'])
    qd = np.array(history['qd'])
    q_des = np.array(history['q_des'])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    fig.suptitle(f"{cfg['title']} Phase Portraits", fontsize=13, fontweight='bold')

    labels = [('z', r'$\dot{z}$'), (r'$\alpha$', r'$\dot{\alpha}$'), (r'$\beta$', r'$\dot{\beta}$')]
    for i, (xlabel, ylabel) in enumerate(labels):
        ax = axes[i]
        sc = ax.scatter(q[:, i], qd[:, i], c=t, cmap='viridis', s=2, alpha=0.7)
        ax.plot(q[0, i], qd[0, i], 'go', markersize=8, label='Start')
        ax.plot(q[-1, i], qd[-1, i], 'rs', markersize=8, label='End')
        ax.plot(q_des[-1, i], 0, 'k+', markersize=12, markeredgewidth=2, label='Desired')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, label='Time (s)')

    fig_path = os.path.join(FIGURE_DIR, f"{cfg['file_tag']}_phase_portrait.png")
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"相平面图已保存: {fig_path}")
    plt.close(fig)
    return fig_path


def main():
    parser = argparse.ArgumentParser(description='Evaluate V5/V5.1 Uniform SAC model')
    parser.add_argument('--variant', choices=list(VARIANT_CONFIG.keys()), default='v5_uniform_v2')
    parser.add_argument('--model', type=str, default=None, help='override model path')
    args = parser.parse_args()

    cfg = VARIANT_CONFIG[args.variant].copy()
    model_path = args.model or cfg['model_path']
    history, total_reward = run_evaluation(cfg, model_path)
    plot_control_performance(history, total_reward, cfg)
    plot_phase_portrait(history, cfg)
    print('\n评估完成 ✓')


if __name__ == '__main__':
    main()