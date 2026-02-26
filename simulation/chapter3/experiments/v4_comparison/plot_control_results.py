"""
绘制V4变体控制结果对比图
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'common'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'env'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'agents'))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from rl_env_v4 import PlatformRLEnvV4
from rl_env_v4_improved import PlatformRLEnvV4Improved
from v4_sac import V4SAC

plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False


def run_episode(env, agent, max_steps=5000):
    """运行一个回合并收集数据"""
    state = env.reset()
    
    data = {
        'time': [],
        'q': [], 'qd': [], 'q_des': [], 'qd_des': [],
        'u': [], 'u_model': [], 'u_rl': [],
        'reward': [], 'error': [],
    }
    
    for step in range(max_steps):
        action = agent.select_action(state, deterministic=True)
        next_state, reward, done, info = env.step(action)
        
        t = step * env.dt
        data['time'].append(t)
        data['q'].append(env.q.copy())
        data['qd'].append(env.qd.copy())
        data['q_des'].append(env.q_des.copy())
        data['qd_des'].append(env.qd_des.copy())
        data['u'].append(info['v_RL'].copy())
        data['u_model'].append(info['tau_model'].copy())
        data['reward'].append(reward)
        data['error'].append(env.q - env.q_des)
        
        state = next_state
        if done:
            break
    
    for key in data:
        data[key] = np.array(data[key])
    
    return data


def plot_tracking_comparison(results_dict, save_dir):
    """绘制跟踪效果对比图"""
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Trajectory Tracking Comparison', fontsize=14, fontweight='bold')
    
    labels = ['Heave z (mm)', 'Roll α (deg)', 'Pitch β (deg)']
    scales = [1000, 180/np.pi, 180/np.pi]
    
    colors = {'Baseline': 'blue', 'Variant2': 'red'}
    
    for variant_name, data in results_dict.items():
        col = 0 if variant_name == 'Baseline' else 1
        
        for i in range(3):
            ax = axes[i, col]
            scale = scales[i]
            
            ax.plot(data['time'], data['q'][:, i] * scale, 
                   color=colors[variant_name], linewidth=1.5, label='Actual')
            ax.plot(data['time'], data['q_des'][:, i] * scale, 
                   'k--', linewidth=1, alpha=0.7, label='Desired')
            
            # 计算误差
            error = (data['q'][:, i] - data['q_des'][:, i]) * scale
            rms = np.sqrt(np.mean(error**2))
            ax.text(0.02, 0.95, f'RMS: {rms:.2f}', transform=ax.transAxes,
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_ylabel(labels[i])
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.set_title(f'{variant_name} Tracking')
    
    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'tracking_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ 保存: {save_dir / 'tracking_comparison.png'}")
    plt.close()


def plot_control_inputs(results_dict, save_dir):
    """绘制控制输入对比图"""
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle('Control Inputs Comparison', fontsize=14, fontweight='bold')
    
    leg_labels = ['Leg 1 (N)', 'Leg 2 (N)', 'Leg 3 (N)']
    colors = {'Baseline': 'blue', 'Variant2': 'red'}
    
    for variant_name, data in results_dict.items():
        linestyle = '-' if variant_name == 'Baseline' else '--'
        
        for i in range(3):
            ax = axes[i]
            ax.plot(data['time'], data['u_model'][:, i], 
                   color=colors[variant_name], linestyle=linestyle,
                   linewidth=1.5, label=f'{variant_name} Model')
            ax.plot(data['time'], data['u'][:, i], 
                   color=colors[variant_name], linestyle=':',
                   linewidth=1, alpha=0.7, label=f'{variant_name} RL')
            
            ax.set_ylabel(leg_labels[i])
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'control_inputs.png', dpi=150, bbox_inches='tight')
    print(f"✓ 保存: {save_dir / 'control_inputs.png'}")
    plt.close()


def plot_error_comparison(results_dict, save_dir):
    """绘制误差对比图"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Tracking Error Analysis', fontsize=14, fontweight='bold')
    
    colors = {'Baseline': 'blue', 'Variant2': 'red'}
    scales = [1000, 180/np.pi, 180/np.pi]
    labels = ['z (mm)', 'α (deg)', 'β (deg)']
    
    # 时域误差
    ax = axes[0, 0]
    for variant_name, data in results_dict.items():
        for i in range(3):
            error = (data['q'][:, i] - data['q_des'][:, i]) * scales[i]
            ax.plot(data['time'], error, 
                   color=colors[variant_name], 
                   linewidth=1, alpha=0.7, 
                   label=f'{variant_name} - {labels[i]}')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error')
    ax.set_title('Tracking Error Over Time')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # 误差分布
    ax = axes[0, 1]
    for variant_name, data in results_dict.items():
        total_error = np.linalg.norm(
            (data['q'] - data['q_des']) * np.array([scales[0], scales[1], scales[2]]), 
            axis=1
        )
        ax.hist(total_error, bins=30, alpha=0.5, 
               label=variant_name, color=colors[variant_name])
    ax.set_xlabel('Total Error')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 统计表格
    ax = axes[1, 0]
    ax.axis('off')
    
    stats = []
    for variant_name, data in results_dict.items():
        for i in range(3):
            error = (data['q'][:, i] - data['q_des'][:, i]) * scales[i]
            rms = np.sqrt(np.mean(error**2))
            max_err = np.max(np.abs(error))
            stats.append([variant_name, labels[i], f'{rms:.3f}', f'{max_err:.3f}'])
    
    table_data = [['Variant', 'DOF', 'RMS Error', 'Max Error']] + stats
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.2, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax.set_title('Error Statistics')
    
    # ISE对比
    ax = axes[1, 1]
    variants = list(results_dict.keys())
    ise_values = []
    
    for variant_name, data in results_dict.items():
        errors = data['q'] - data['q_des']
        ise = np.sum(errors**2, axis=0) * data['time'][-1] / len(data['time'])
        ise_total = np.sum(ise) * 1000  # 转换为更小的单位
        ise_values.append(ise_total)
    
    bars = ax.bar(variants, ise_values, color=['blue', 'red'], alpha=0.7, edgecolor='black')
    for bar, val in zip(bars, ise_values):
        ax.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, val),
                   xytext=(0, 3), textcoords='offset points', ha='center')
    
    ax.set_ylabel('ISE (×10⁻³)')
    ax.set_title('Integrated Squared Error')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'error_analysis.png', dpi=150, bbox_inches='tight')
    print(f"✓ 保存: {save_dir / 'error_analysis.png'}")
    plt.close()


def plot_single_variant(data, variant_name, save_dir):
    """绘制单个变体的详细控制结果"""
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'{variant_name} - Control Results', fontsize=14, fontweight='bold')
    
    labels = ['Heave z (mm)', 'Roll α (deg)', 'Pitch β (deg)']
    scales = [1000, 180/np.pi, 180/np.pi]
    
    # 1. 轨迹跟踪
    for i in range(3):
        ax = fig.add_subplot(4, 3, i+1)
        scale = scales[i]
        
        ax.plot(data['time'], data['q'][:, i] * scale, 'b-', linewidth=1.5, label='Actual')
        ax.plot(data['time'], data['q_des'][:, i] * scale, 'r--', linewidth=1, label='Desired')
        ax.fill_between(data['time'], 
                      data['q'][:, i] * scale, 
                      data['q_des'][:, i] * scale,
                      alpha=0.3, color='blue')
        
        error = (data['q'][:, i] - data['q_des'][:, i]) * scale
        rms = np.sqrt(np.mean(error**2))
        ax.text(0.02, 0.95, f'RMS: {rms:.2f}', transform=ax.transAxes,
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_ylabel(labels[i])
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title('Position Tracking')
    
    # 2. 跟踪误差
    for i in range(3):
        ax = fig.add_subplot(4, 3, i+4)
        scale = scales[i]
        
        error = (data['q'][:, i] - data['q_des'][:, i]) * scale
        ax.plot(data['time'], error, 'b-', linewidth=1)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        ax.set_ylabel(f'Error {labels[i]}')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title('Tracking Error')
    
    # 3. 控制量
    for i in range(3):
        ax = fig.add_subplot(4, 3, i+7)
        
        total_u = data['u_model'][:, i] + data['u'][:, i]
        ax.plot(data['time'], total_u/1000, 'k-', linewidth=1.5, label='Total')
        ax.plot(data['time'], data['u_model'][:, i]/1000, 'g--', linewidth=1, label='Model')
        ax.plot(data['time'], data['u'][:, i]/1000, 'r:', linewidth=1, label='RL')
        
        ax.set_ylabel(f'Leg {i+1} (kN)')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title('Control Force')
    
    # 4. 奖励
    ax = fig.add_subplot(4, 3, 10)
    ax.plot(data['time'], data['reward'], 'b-', linewidth=1)
    ax.axhline(y=np.mean(data['reward']), color='r', linestyle='--', 
              label=f'Mean: {np.mean(data["reward"]):.1f}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Reward')
    ax.set_title('Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. 统计信息
    ax = fig.add_subplot(4, 3, 11)
    ax.axis('off')
    
    stats_text = f"""
    Performance Statistics
    
    Episode Length: {data['time'][-1]:.1f}s
    Total Steps: {len(data['time'])}
    
    RMS Errors:
    • z:     {np.sqrt(np.mean((data['q'][:,0]-data['q_des'][:,0])**2))*1000:.2f} mm
    • α:     {np.sqrt(np.mean((data['q'][:,1]-data['q_des'][:,1])**2))*180/np.pi:.2f} deg
    • β:     {np.sqrt(np.mean((data['q'][:,2]-data['q_des'][:,2])**2))*180/np.pi:.2f} deg
    
    Mean Reward: {np.mean(data['reward']):.2f}
    Total Reward: {np.sum(data['reward']):.2f}
    """
    ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
           verticalalignment='center',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(save_dir / f'{variant_name}_control.png', dpi=150, bbox_inches='tight')
    print(f"✓ 保存: {save_dir / f'{variant_name}_control.png'}")
    plt.close()


def main():
    print("="*70)
    print("绘制V4变体控制结果图")
    print("="*70)
    
    base_dir = Path(__file__).parent
    save_dir = base_dir / 'control_plots'
    save_dir.mkdir(exist_ok=True)
    
    # 配置
    episode_length = 30
    max_steps = int(episode_length / 0.01)
    
    # 定义变体
    variants = {
        'Baseline': {
            'env_class': PlatformRLEnvV4,
            'model_path': base_dir.parent / 'models_v4_optimized/stage3/Stage3_best.pt',
            'extra_params': {}
        },
        'Variant2': {
            'env_class': PlatformRLEnvV4Improved,
            'model_path': base_dir / 'trained_variants/Variant2_No_Warning_Penalty/Stage3_best.pt',
            'extra_params': {'diverge_threshold': 0.5, 'warning_threshold': 0.4, 'warning_penalty': 0.0}
        },
    }
    
    results = {}
    
    # 运行每个变体
    for variant_name, config in variants.items():
        print(f"\n运行: {variant_name}")
        
        if not config['model_path'].exists():
            print(f"  ⚠️ 模型不存在: {config['model_path']}")
            continue
        
        # 加载agent
        agent = V4SAC(
            state_dim=9,
            action_dim=3,
            hidden_dims=[512, 512, 256, 256],
            device=None
        )
        agent.load(str(config['model_path']))
        
        # 创建环境
        env = config['env_class'](
            use_model_compensation=True,
            max_episode_steps=max_steps,
            Hs=2.0,
            T1=8.0,
            q_des_type='sinusoidal',
            **config['extra_params']
        )
        
        # 运行回合
        print(f"  运行回合...")
        data = run_episode(env, agent, max_steps)
        results[variant_name] = data
        
        print(f"  回合完成: {len(data['time'])}步")
        
        # 绘制单个变体图
        plot_single_variant(data, variant_name, save_dir)
    
    # 绘制对比图
    if len(results) >= 2:
        print("\n绘制对比图...")
        plot_tracking_comparison(results, save_dir)
        plot_control_inputs(results, save_dir)
        plot_error_comparison(results, save_dir)
    
    print("\n" + "="*70)
    print("所有控制结果图已保存!")
    print("="*70)
    print(f"\n保存目录: {save_dir}")
    print("\n生成的文件:")
    print("  - [variant]_control.png (单个变体详细结果)")
    print("  - tracking_comparison.png (跟踪对比)")
    print("  - control_inputs.png (控制输入对比)")
    print("  - error_analysis.png (误差分析)")


if __name__ == "__main__":
    main()
