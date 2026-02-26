"""
绘制V4变体对比结果图
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'common'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'env'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'agents'))

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_training_histories():
    """加载训练历史"""
    base_dir = Path(__file__).parent / 'trained_variants'
    
    histories = {}
    for variant_name in ['Original_V4_Improved', 'Variant1_Relaxed_Threshold', 'Variant2_No_Warning_Penalty']:
        history_file = base_dir / variant_name / 'training_history.json'
        if history_file.exists():
            with open(history_file, 'r') as f:
                histories[variant_name] = json.load(f)
    
    return histories


def plot_training_curves(histories):
    """绘制训练曲线"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('Training Curves Comparison', fontsize=14, fontweight='bold')
    
    stage_colors = {'Stage1': 'blue', 'Stage2': 'green', 'Stage3': 'red'}
    
    for idx, (variant_name, history) in enumerate(histories.items()):
        ax = axes[idx]
        
        episode_offset = 0
        for stage in history['stages']:
            stage_name = stage['stage_name']
            rewards = stage['episode_rewards']
            
            episodes = np.arange(episode_offset, episode_offset + len(rewards))
            ax.plot(episodes, rewards, color=stage_colors[stage_name], 
                   label=f'{stage_name} ({len(rewards)} eps)', alpha=0.8)
            
            # 标记最佳点
            best_ep = stage['best_episode']
            best_reward = stage['best_reward']
            ax.scatter([episode_offset + best_ep], [best_reward], 
                      color='gold', s=100, zorder=5, marker='*', 
                      edgecolors='black')
            
            episode_offset += len(rewards)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title(f'{variant_name}\nBest: Stage1={history["stages"][0]["best_reward"]:.1f}, '
                    f'Stage2={history["stages"][1]["best_reward"]:.1f}, '
                    f'Stage3={history["stages"][2]["best_reward"]:.1f}')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print("✓ 保存: training_curves.png")
    plt.close()


def plot_ise_comparison():
    """绘制ISE对比图（基于评估结果）"""
    
    # 评估结果数据（来自之前的运行）
    results = {
        'Easy_20s': {
            'Baseline_V4_Optimized': {'sinusoidal': 0.000925, 'sinusoidal_small': 0.000435},
            'Original_V4_Improved': {'sinusoidal': 0.000233, 'sinusoidal_small': 0.000197},
            'Variant1_Relaxed_Threshold': {'sinusoidal': 0.000454, 'sinusoidal_small': 0.000541},
            'Variant2_No_Warning_Penalty': {'sinusoidal': 0.000107, 'sinusoidal_small': 0.000053},
        },
        'Medium_30s': {
            'Baseline_V4_Optimized': {'sinusoidal': 0.000918, 'sinusoidal_small': 0.000449},
            'Original_V4_Improved': {'sinusoidal': 0.000228, 'sinusoidal_small': 0.000181},
            'Variant1_Relaxed_Threshold': {'sinusoidal': 0.000462, 'sinusoidal_small': 0.000532},
            'Variant2_No_Warning_Penalty': {'sinusoidal': 0.000108, 'sinusoidal_small': 0.000057},
        }
    }
    
    scenarios = list(results.keys())
    variants = list(results[scenarios[0]].keys())
    trajectories = ['sinusoidal', 'sinusoidal_small']
    
    # 图1: 所有场景的ISE对比
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('ISE/step Comparison Across Variants', fontsize=14, fontweight='bold')
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    bar_width = 0.2
    
    for row, scenario in enumerate(scenarios):
        for col, traj in enumerate(trajectories):
            ax = axes[row, col]
            
            x = np.arange(len(variants))
            values = [results[scenario][v][traj] for v in variants]
            
            bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor='black')
            
            # 添加数值标签
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{val:.6f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8, rotation=45)
            
            ax.set_xlabel('Variant')
            ax.set_ylabel('ISE/step')
            ax.set_title(f'{scenario} - {traj}')
            ax.set_xticks(x)
            ax.set_xticklabels([v.replace('_', '\n') for v in variants], fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
            
            # 标记最佳
            best_idx = np.argmin(values)
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig('ise_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ 保存: ise_comparison.png")
    plt.close()


def plot_performance_summary():
    """绘制性能汇总图"""
    
    # 最终性能数据
    variants = ['Baseline\nV4_Optimized', 'Original\nV4_Improved', 'Variant1\nRelaxed', 'Variant2\nNo_Warning']
    ise_sinusoidal = [0.000918, 0.000228, 0.000462, 0.000108]
    ise_small = [0.000449, 0.000181, 0.000532, 0.000057]
    
    x = np.arange(len(variants))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars1 = ax.bar(x - width/2, ise_sinusoidal, width, label='Sinusoidal', 
                   color='#2E86AB', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, ise_small, width, label='Sinusoidal_small', 
                  color='#F18F01', alpha=0.8, edgecolor='black')
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.6f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.6f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Variant', fontsize=12)
    ax.set_ylabel('ISE/step (lower is better)', fontsize=12)
    ax.set_title('V4 Variants Performance Comparison (Medium_30s Scenario)\nISE/step - Lower is Better', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加最佳标记
    best_idx_sin = np.argmin(ise_sinusoidal)
    best_idx_small = np.argmin(ise_small)
    bars1[best_idx_sin].set_edgecolor('gold')
    bars1[best_idx_sin].set_linewidth(3)
    bars2[best_idx_small].set_edgecolor('gold')
    bars2[best_idx_small].set_linewidth(3)
    
    # 添加说明文字
    improvement_sin = ise_sinusoidal[0] / ise_sinusoidal[3]
    improvement_small = ise_small[0] / ise_small[3]
    
    textstr = f'Variant2 improvements:\n'
    textstr += f'vs Baseline: {improvement_sin:.1f}x (sin), {improvement_small:.1f}x (small)\n'
    textstr += f'vs Original: {ise_sinusoidal[1]/ise_sinusoidal[3]:.1f}x (sin), {ise_small[1]/ise_small[3]:.1f}x (small)'
    
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig('performance_summary.png', dpi=150, bbox_inches='tight')
    print("✓ 保存: performance_summary.png")
    plt.close()


def plot_training_rewards_summary():
    """绘制训练最佳奖励汇总"""
    
    # 训练最佳奖励数据
    variants = ['Original\nV4_Improved', 'Variant1\nRelaxed_Threshold', 'Variant2\nNo_Warning_Penalty']
    stage1 = [-357.52, -481.97, -278.51]
    stage2 = [-811.24, -732.94, 1070.88]
    stage3 = [-1527.80, -1166.25, -257.39]
    
    x = np.arange(len(variants))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars1 = ax.bar(x - width, stage1, width, label='Stage1 (20s)', 
                   color='#2E86AB', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x, stage2, width, label='Stage2 (30s)', 
                  color='#F18F01', alpha=0.8, edgecolor='black')
    bars3 = ax.bar(x + width, stage3, width, label='Stage3 (40s)', 
                  color='#C73E1D', alpha=0.8, edgecolor='black')
    
    # 添加数值标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Variant', fontsize=12)
    ax.set_ylabel('Best Reward (higher is better)', fontsize=12)
    ax.set_title('Training Best Rewards Comparison\nHigher is Better', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 突出显示正值
    bars2[2].set_edgecolor('gold')
    bars2[2].set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig('training_rewards_summary.png', dpi=150, bbox_inches='tight')
    print("✓ 保存: training_rewards_summary.png")
    plt.close()


def main():
    print("="*70)
    print("绘制V4变体对比结果图")
    print("="*70)
    
    # 1. 绘制训练曲线
    print("\n1. 绘制训练曲线...")
    histories = load_training_histories()
    if histories:
        plot_training_curves(histories)
    
    # 2. 绘制ISE对比
    print("\n2. 绘制ISE对比...")
    plot_ise_comparison()
    
    # 3. 绘制性能汇总
    print("\n3. 绘制性能汇总...")
    plot_performance_summary()
    
    # 4. 绘制训练奖励汇总
    print("\n4. 绘制训练奖励汇总...")
    plot_training_rewards_summary()
    
    print("\n" + "="*70)
    print("所有图表已保存!")
    print("="*70)
    print("\n生成的文件:")
    print("  - training_curves.png")
    print("  - ise_comparison.png")
    print("  - performance_summary.png")
    print("  - training_rewards_summary.png")


if __name__ == "__main__":
    main()
