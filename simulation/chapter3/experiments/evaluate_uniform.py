"""
第3章 UniformRod SAC 控制效果评估与可视化

用法:
    conda activate thesis-rl
    cd simulation/chapter3/experiments
    python evaluate_uniform.py

功能:
    1. 加载 Stage3 最佳模型
    2. 在 sinusoidal 跟踪任务上运行 30s
    3. 绘制控制效果图:
       - 位姿跟踪 (z, alpha, beta) 及期望轨迹
       - 跟踪误差
       - 控制力 (模型补偿 + RL 输出)
       - 累积奖励
"""

import sys
import os

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

from rl_env_uniform import PlatformRLEnvUniform
from advanced_sac import AdvancedSAC

# ---------- 配置 ----------
MODEL_PATH = os.path.join(THIS_DIR, 'models_v3_uniform', 'stage3', 'Stage3_Uniform_best.pt')
EPISODE_LENGTH = 30       # 秒
DT = 0.01
Hs = 2.0
T1 = 8.0
Q_DES_TYPE = 'sinusoidal'
FIGURE_DIR = os.path.join(CH3_DIR, 'figures')
os.makedirs(FIGURE_DIR, exist_ok=True)


def run_evaluation():
    """加载模型, 运行 episode, 返回 history."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"评估设备: {device}")

    # 构建 agent (架构需与训练一致)
    agent = AdvancedSAC(
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

    assert os.path.exists(MODEL_PATH), f"找不到模型: {MODEL_PATH}"
    agent.load(MODEL_PATH)
    print(f"已加载模型: {MODEL_PATH}")

    # 构建环境
    max_steps = int(EPISODE_LENGTH / DT)
    env = PlatformRLEnvUniform(
        use_model_compensation=True,
        dt=DT,
        max_episode_steps=max_steps,
        Hs=Hs,
        T1=T1,
        q_des_type=Q_DES_TYPE,
    )

    # 运行 episode (deterministic)
    state = env.reset()
    total_reward = 0.0
    for step in range(max_steps):
        action = agent.select_action(state, deterministic=True)
        state, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break

    history = env.get_history()
    print(f"回合完成: steps={len(history['time'])}, total_reward={total_reward:.2f}")
    return history, total_reward


def plot_control_performance(history, total_reward):
    """绘制 4 组控制效果子图并保存."""

    t = np.array(history['time'])
    q = np.array(history['q'])          # (N, 3)
    qd = np.array(history['qd'])        # (N, 3)
    q_des = np.array(history['q_des'])  # (N, 3)
    u = np.array(history['u'])          # (N, 3)
    v_rl = np.array(history['v_RL'])    # (N, 3)
    tau_model = np.array(history['tau_model'])  # (N, 3)
    rewards = np.array(history['reward'])

    labels_q = ['z (m)', r'$\alpha$ (rad)', r'$\beta$ (rad)']
    labels_u = [r'$F_z$ (N)', r'$\tau_\alpha$ (N·m)', r'$\tau_\beta$ (N·m)']

    fig, axes = plt.subplots(4, 3, figsize=(18, 16), constrained_layout=True)
    fig.suptitle(
        f'Chapter 3 – UniformRod SAC Control Performance  '
        f'(Hs={Hs}m, T1={T1}s, {Q_DES_TYPE})\n'
        f'Total Reward = {total_reward:.2f}',
        fontsize=14, fontweight='bold',
    )

    # ---------- Row 1: 位姿跟踪 ----------
    for i in range(3):
        ax = axes[0, i]
        ax.plot(t, q[:, i], 'b-', linewidth=1.2, label='Actual')
        ax.plot(t, q_des[:, i], 'r--', linewidth=1.0, label='Desired')
        ax.set_ylabel(labels_q[i])
        ax.set_title(f'Pose Tracking – {labels_q[i]}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # ---------- Row 2: 跟踪误差 ----------
    error = q - q_des
    for i in range(3):
        ax = axes[1, i]
        ax.plot(t, error[:, i], 'k-', linewidth=1.0)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_ylabel(f'Error – {labels_q[i]}')
        ax.set_title(f'Tracking Error – {labels_q[i]}')
        ax.grid(True, alpha=0.3)

    # ---------- Row 3: 控制力分解 ----------
    for i in range(3):
        ax = axes[2, i]
        ax.plot(t, u[:, i], 'b-', linewidth=1.0, label='Total u')
        ax.plot(t, tau_model[:, i], 'g--', linewidth=0.8, label='Model comp.')
        ax.plot(t, v_rl[:, i], 'r:', linewidth=0.8, label='RL output')
        ax.set_ylabel(labels_u[i])
        ax.set_title(f'Control Input – {labels_u[i]}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # ---------- Row 4: 奖励 & 误差范数 & 速度 ----------
    # 4a: 瞬时奖励
    ax = axes[3, 0]
    ax.plot(t, rewards, 'm-', linewidth=0.8)
    ax.set_ylabel('Reward')
    ax.set_title('Instantaneous Reward')
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)

    # 4b: 误差范数
    ax = axes[3, 1]
    error_norm = np.linalg.norm(error, axis=1)
    ax.plot(t, error_norm, 'k-', linewidth=1.0)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_ylabel('‖e‖')
    ax.set_title('Tracking Error Norm')
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)

    # 4c: 速度
    ax = axes[3, 2]
    for i, (lbl, c) in enumerate(zip(
            [r'$\dot{z}$', r'$\dot{\alpha}$', r'$\dot{\beta}$'],
            ['tab:blue', 'tab:orange', 'tab:green'])):
        ax.plot(t, qd[:, i], color=c, linewidth=0.8, label=lbl)
    ax.set_ylabel('Velocity')
    ax.set_title('Generalized Velocities')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 保存
    fig_path = os.path.join(FIGURE_DIR, 'uniform_sac_control_performance.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"控制效果图已保存: {fig_path}")
    plt.close(fig)

    # ---------- 额外: 误差统计 ----------
    print("\n--- 跟踪误差统计 ---")
    for i, name in enumerate(['z', 'alpha', 'beta']):
        e_i = error[:, i]
        print(f"  {name:>5s}: mean={np.mean(np.abs(e_i)):.6f}, "
              f"max={np.max(np.abs(e_i)):.6f}, "
              f"rms={np.sqrt(np.mean(e_i**2)):.6f}")

    overall_rms = np.sqrt(np.mean(error_norm ** 2))
    print(f"  Overall RMS error norm: {overall_rms:.6f}")


def plot_phase_portrait(history):
    """绘制 z vs z_dot 等相平面图（可选的更深入可视化）."""

    t = np.array(history['time'])
    q = np.array(history['q'])
    qd = np.array(history['qd'])
    q_des = np.array(history['q_des'])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    fig.suptitle('Phase Portraits', fontsize=13, fontweight='bold')

    labels = [
        ('z', r'$\dot{z}$'),
        (r'$\alpha$', r'$\dot{\alpha}$'),
        (r'$\beta$', r'$\dot{\beta}$'),
    ]

    for i, (xlabel, ylabel) in enumerate(labels):
        ax = axes[i]
        # 彩色线: 颜色表示时间
        sc = ax.scatter(q[:, i], qd[:, i], c=t, cmap='viridis', s=2, alpha=0.7)
        # 标记起点和终点
        ax.plot(q[0, i], qd[0, i], 'go', markersize=8, label='Start')
        ax.plot(q[-1, i], qd[-1, i], 'rs', markersize=8, label='End')
        # 期望稳态
        ax.plot(q_des[-1, i], 0, 'k+', markersize=12, markeredgewidth=2, label='Desired')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, label='Time (s)')

    fig_path = os.path.join(FIGURE_DIR, 'uniform_sac_phase_portrait.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"相平面图已保存: {fig_path}")
    plt.close(fig)


# ==================== main ====================
if __name__ == '__main__':
    history, total_reward = run_evaluation()
    plot_control_performance(history, total_reward)
    plot_phase_portrait(history)
    print("\n评估完成 ✓")
