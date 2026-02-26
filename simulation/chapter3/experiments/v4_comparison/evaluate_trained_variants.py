"""
评估所有训练好的V4变体模型
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'common'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'env'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'agents'))

import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from rl_env_v4 import PlatformRLEnvV4
from rl_env_v4_improved import PlatformRLEnvV4Improved
from v4_sac import V4SAC


def compute_ise_metrics(env, agent, n_episodes=5, seed=None):
    """计算ISE指标"""
    if seed is not None:
        np.random.seed(seed)
    
    ise_list = []
    steps_list = []
    rewards_list = []
    
    for ep in range(n_episodes):
        state = env.reset()
        done = False
        step = 0
        episode_reward = 0
        
        while not done:
            action = agent.select_action(state, deterministic=True)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
            step += 1
            
            if step >= env.max_episode_steps:
                break
        
        history = env.get_history()
        errors = np.array(history['q']) - np.array(history['q_des'])
        ise = np.sum(errors**2, axis=0) * env.dt
        
        ise_list.append(ise)
        steps_list.append(step)
        rewards_list.append(episode_reward)
    
    avg_ise = np.mean(ise_list, axis=0)
    std_ise = np.std(ise_list, axis=0)
    avg_steps = np.mean(steps_list)
    avg_reward = np.mean(rewards_list)
    ise_per_step = avg_ise / max(avg_steps, 1)
    
    return {
        'ise_per_step': np.sum(ise_per_step),
        'ise_per_step_z': ise_per_step[0],
        'ise_per_step_alpha': ise_per_step[1],
        'ise_per_step_beta': ise_per_step[2],
        'ise_total': np.sum(avg_ise),
        'avg_steps': avg_steps,
        'avg_reward': avg_reward,
    }


def main():
    print("="*70)
    print("V4变体ISE评估")
    print("="*70)
    
    base_dir = Path(__file__).parent
    results_dir = base_dir / 'evaluation_results'
    results_dir.mkdir(exist_ok=True)
    
    # 测试配置
    trajectory_types = ['constant', 'sinusoidal', 'sinusoidal_small']
    test_scenarios = [
        {'episode_length': 20, 'Hs': 1.0, 'T1': 8.0, 'name': 'Easy_20s'},
        {'episode_length': 30, 'Hs': 2.0, 'T1': 8.0, 'name': 'Medium_30s'},
        {'episode_length': 40, 'Hs': 2.0, 'T1': 8.0, 'name': 'Hard_40s'},
    ]
    seeds = [42, 123, 456]
    
    # 定义所有变体（包括原始V4 Optimized基线）
    variants = {
        'Baseline_V4_Optimized': {
            'env_class': PlatformRLEnvV4,
            'model_path': base_dir.parent / 'models_v4_optimized/stage3/Stage3_best.pt',
            'extra_params': {}
        },
        'Original_V4_Improved': {
            'env_class': PlatformRLEnvV4Improved,
            'model_path': base_dir / 'trained_variants/Original_V4_Improved/Stage3_best.pt',
            'extra_params': {'diverge_threshold': 0.5, 'warning_threshold': 0.4, 'warning_penalty': 5.0}
        },
        'Variant1_Relaxed_Threshold': {
            'env_class': PlatformRLEnvV4Improved,
            'model_path': base_dir / 'trained_variants/Variant1_Relaxed_Threshold/Stage3_best.pt',
            'extra_params': {'diverge_threshold': 1.0, 'warning_threshold': 0.4, 'warning_penalty': 5.0}
        },
        'Variant2_No_Warning_Penalty': {
            'env_class': PlatformRLEnvV4Improved,
            'model_path': base_dir / 'trained_variants/Variant2_No_Warning_Penalty/Stage3_best.pt',
            'extra_params': {'diverge_threshold': 0.5, 'warning_threshold': 0.4, 'warning_penalty': 0.0}
        },
    }
    
    all_results = {}
    
    # 对每个场景测试
    for scenario in test_scenarios:
        scenario_name = scenario['name']
        print(f"\n{'='*70}")
        print(f"场景: {scenario_name}")
        print(f"{'='*70}")
        
        scenario_results = {}
        
        # 对每个变体测试
        for variant_name, variant_config in variants.items():
            print(f"\n评估: {variant_name}")
            
            # 检查模型是否存在
            if not variant_config['model_path'].exists():
                print(f"  ⚠️ 模型不存在: {variant_config['model_path']}")
                continue
            
            # 加载agent
            agent = V4SAC(
                state_dim=9,
                action_dim=3,
                hidden_dims=[512, 512, 256, 256],
                device=None
            )
            agent.load(str(variant_config['model_path']))
            
            variant_results = {}
            
            # 对每种轨迹类型测试
            for traj_type in trajectory_types:
                # 创建环境
                max_steps = int(scenario['episode_length'] / 0.01)
                env = variant_config['env_class'](
                    use_model_compensation=True,
                    max_episode_steps=max_steps,
                    Hs=scenario['Hs'],
                    T1=scenario['T1'],
                    q_des_type=traj_type,
                    **variant_config['extra_params']
                )
                
                # 对每个种子测试
                seed_results = []
                for seed in seeds:
                    metrics = compute_ise_metrics(env, agent, n_episodes=3, seed=seed)
                    seed_results.append(metrics)
                
                # 汇总结果
                avg_ise = np.mean([r['ise_per_step'] for r in seed_results])
                std_ise = np.std([r['ise_per_step'] for r in seed_results])
                avg_reward = np.mean([r['avg_reward'] for r in seed_results])
                
                variant_results[traj_type] = {
                    'ise_per_step': avg_ise,
                    'ise_per_step_std': std_ise,
                    'avg_reward': avg_reward,
                }
                
                print(f"  {traj_type:20s}: ISE/step = {avg_ise:.6f} ± {std_ise:.6f}, Reward = {avg_reward:.2f}")
            
            scenario_results[variant_name] = variant_results
        
        all_results[scenario_name] = scenario_results
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = results_dir / f'evaluation_{timestamp}.json'
    with open(result_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ 结果已保存: {result_file}")
    
    # 生成报告
    report_file = results_dir / f'report_{timestamp}.txt'
    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("V4变体ISE评估报告\n")
        f.write("="*70 + "\n\n")
        
        for scenario_name, scenario_results in all_results.items():
            f.write(f"\n场景: {scenario_name}\n")
            f.write("-"*70 + "\n")
            f.write(f"{'变体':<35} {'轨迹':<20} {'ISE/step':<15} {'Reward':<12}\n")
            f.write("-"*70 + "\n")
            
            for variant_name, variant_results in scenario_results.items():
                for traj_type, metrics in variant_results.items():
                    f.write(f"{variant_name:<35} {traj_type:<20} "
                           f"{metrics['ise_per_step']:<15.6f} "
                           f"{metrics['avg_reward']:<12.2f}\n")
            f.write("\n")
        
        # 找出最佳变体
        f.write("\n" + "="*70 + "\n")
        f.write("最佳性能分析\n")
        f.write("="*70 + "\n")
        
        for scenario_name, scenario_results in all_results.items():
            f.write(f"\n{scenario_name}:\n")
            
            for traj_type in trajectory_types:
                best_variant = None
                best_ise = float('inf')
                
                for variant_name, variant_results in scenario_results.items():
                    ise = variant_results[traj_type]['ise_per_step']
                    if ise < best_ise:
                        best_ise = ise
                        best_variant = variant_name
                
                f.write(f"  {traj_type}: {best_variant} (ISE/step: {best_ise:.6f})\n")
    
    print(f"✓ 报告已保存: {report_file}")
    print("\n✓ 所有评估完成！")


if __name__ == "__main__":
    main()
