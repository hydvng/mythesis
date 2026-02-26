"""
V4 Improved变体全面评估脚本
在多种期望轨迹下测试不同版本的性能
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
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 导入环境
from rl_env_v4 import PlatformRLEnvV4
from rl_env_v4_improved import PlatformRLEnvV4Improved

# 导入agent
from v4_sac import V4SAC


class V4Comparator:
    """V4变体对比评估器"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / 'results'
        self.results_dir.mkdir(exist_ok=True)
        
        # 定义测试轨迹类型
        self.trajectory_types = [
            'constant',           # 恒定轨迹
            'sinusoidal',         # 正弦轨迹
            'sinusoidal_small',   # 小幅正弦
        ]
        
        # 定义测试场景
        self.test_scenarios = [
            {'episode_length': 20, 'Hs': 1.0, 'T1': 8.0, 'name': 'Easy_20s'},
            {'episode_length': 30, 'Hs': 2.0, 'T1': 8.0, 'name': 'Medium_30s'},
            {'episode_length': 40, 'Hs': 2.0, 'T1': 8.0, 'name': 'Hard_40s'},
        ]
        
        # 随机种子
        self.seeds = [42, 123, 456, 789, 1024]
    
    def compute_ise_metrics(self, env, agent, n_episodes: int = 5, seed: int = None) -> Dict:
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
            
            # 计算ISE
            history = env.get_history()
            errors = np.array(history['q']) - np.array(history['q_des'])
            ise = np.sum(errors**2, axis=0) * env.dt
            
            ise_list.append(ise)
            steps_list.append(step)
            rewards_list.append(episode_reward)
        
        # 计算统计
        avg_ise = np.mean(ise_list, axis=0)
        std_ise = np.std(ise_list, axis=0)
        avg_steps = np.mean(steps_list)
        avg_reward = np.mean(rewards_list)
        
        # 计算ISE/step
        ise_per_step = avg_ise / max(avg_steps, 1)
        
        return {
            'ise_z': avg_ise[0],
            'ise_alpha': avg_ise[1],
            'ise_beta': avg_ise[2],
            'ise_total': np.sum(avg_ise),
            'ise_std': np.sum(std_ise),
            'ise_per_step': np.sum(ise_per_step),
            'ise_per_step_z': ise_per_step[0],
            'ise_per_step_alpha': ise_per_step[1],
            'ise_per_step_beta': ise_per_step[2],
            'avg_steps': avg_steps,
            'avg_reward': avg_reward,
        }
    
    def evaluate_variant(self, 
                        env_class,
                        env_params: Dict,
                        agent: V4SAC,
                        variant_name: str) -> Dict:
        """评估单个变体"""
        print(f"\n评估变体: {variant_name}")
        print(f"环境参数: {env_params}")
        
        # 创建环境
        max_steps = int(env_params['episode_length'] / 0.01)
        env = env_class(
            use_model_compensation=True,
            max_episode_steps=max_steps,
            Hs=env_params['Hs'],
            T1=env_params['T1'],
            q_des_type=env_params.get('q_des_type', 'sinusoidal'),
            **env_params.get('extra_params', {})
        )
        
        # 对每种轨迹类型进行测试
        results = {}
        for traj_type in self.trajectory_types:
            env.q_des_type = traj_type
            print(f"  轨迹类型: {traj_type}")
            
            # 对每个种子进行测试
            seed_results = []
            for seed in self.seeds:
                metrics = self.compute_ise_metrics(env, agent, n_episodes=3, seed=seed)
                seed_results.append(metrics)
            
            # 汇总结果
            avg_results = {
                'ise_per_step': np.mean([r['ise_per_step'] for r in seed_results]),
                'ise_per_step_std': np.std([r['ise_per_step'] for r in seed_results]),
                'ise_per_step_z': np.mean([r['ise_per_step_z'] for r in seed_results]),
                'ise_per_step_alpha': np.mean([r['ise_per_step_alpha'] for r in seed_results]),
                'ise_per_step_beta': np.mean([r['ise_per_step_beta'] for r in seed_results]),
                'avg_reward': np.mean([r['avg_reward'] for r in seed_results]),
                'avg_steps': np.mean([r['avg_steps'] for r in seed_results]),
            }
            
            results[traj_type] = avg_results
            print(f"    ISE/step: {avg_results['ise_per_step']:.6f} ± {avg_results['ise_per_step_std']:.6f}")
        
        return results
    
    def run_full_comparison(self, model_path: str):
        """运行完整对比测试"""
        print("="*70)
        print("V4 Improved变体全面对比测试")
        print("="*70)
        print(f"模型路径: {model_path}")
        print(f"测试种子: {self.seeds}")
        print(f"轨迹类型: {self.trajectory_types}")
        print(f"测试场景: {[s['name'] for s in self.test_scenarios]}")
        
        # 加载agent
        agent = V4SAC(
            state_dim=9,
            action_dim=3,
            hidden_dims=[512, 512, 256, 256],
            device=None
        )
        
        if Path(model_path).exists():
            agent.load(model_path)
            print(f"✓ 模型加载成功")
        else:
            print(f"✗ 模型未找到: {model_path}")
            return None
        
        all_results = {}
        
        # 定义所有变体
        variants = {
            'Baseline_V4_Optimized': {
                'env_class': PlatformRLEnvV4,
                'params': {'extra_params': {}}
            },
            'Original_V4_Improved': {
                'env_class': PlatformRLEnvV4Improved,
                'params': {'extra_params': {
                    'diverge_threshold': 0.5,
                    'warning_threshold': 0.4
                }}
            },
            'Variant1_Relaxed_Threshold': {
                'env_class': PlatformRLEnvV4Improved,
                'params': {'extra_params': {
                    'diverge_threshold': 1.0,
                    'warning_threshold': 0.4
                }}
            },
            'Variant2_No_Warning': {
                'env_class': PlatformRLEnvV4Improved,
                'params': {'extra_params': {
                    'diverge_threshold': 0.5,
                    'warning_threshold': 0.4,
                    'warning_penalty': 0.0  # 需要在环境中支持
                }}
            },
        }
        
        # 对每个测试场景
        for scenario in self.test_scenarios:
            scenario_name = scenario['name']
            print(f"\n{'='*70}")
            print(f"测试场景: {scenario_name}")
            print(f"{'='*70}")
            
            scenario_results = {}
            
            # 对每个变体进行测试
            for variant_name, variant_config in variants.items():
                env_params = {
                    'episode_length': scenario['episode_length'],
                    'Hs': scenario['Hs'],
                    'T1': scenario['T1'],
                    **variant_config['params']
                }
                
                results = self.evaluate_variant(
                    variant_config['env_class'],
                    env_params,
                    agent,
                    variant_name
                )
                
                scenario_results[variant_name] = results
            
            all_results[scenario_name] = scenario_results
        
        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = self.results_dir / f'comparison_{timestamp}.json'
        
        with open(result_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n✓ 结果已保存: {result_file}")
        
        # 生成汇总报告
        self.generate_summary_report(all_results, timestamp)
        
        return all_results
    
    def generate_summary_report(self, results: Dict, timestamp: str):
        """生成汇总报告"""
        report_file = self.results_dir / f'summary_{timestamp}.txt'
        
        with open(report_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("V4 Improved变体对比汇总报告\n")
            f.write("="*70 + "\n\n")
            
            for scenario_name, scenario_results in results.items():
                f.write(f"\n场景: {scenario_name}\n")
                f.write("-"*70 + "\n")
                
                # 创建表格
                f.write(f"{'变体':<30} {'轨迹':<15} {'ISE/step':<15} {'Std':<10}\n")
                f.write("-"*70 + "\n")
                
                for variant_name, variant_results in scenario_results.items():
                    for traj_type, metrics in variant_results.items():
                        f.write(f"{variant_name:<30} {traj_type:<15} "
                               f"{metrics['ise_per_step']:<15.6f} "
                               f"{metrics['ise_per_step_std']:<10.6f}\n")
                
                f.write("\n")
            
            # 找出最佳变体
            f.write("\n" + "="*70 + "\n")
            f.write("最佳性能分析\n")
            f.write("="*70 + "\n")
            
            for scenario_name, scenario_results in results.items():
                f.write(f"\n{scenario_name}:\n")
                
                for traj_type in self.trajectory_types:
                    best_variant = None
                    best_ise = float('inf')
                    
                    for variant_name, variant_results in scenario_results.items():
                        ise = variant_results[traj_type]['ise_per_step']
                        if ise < best_ise:
                            best_ise = ise
                            best_variant = variant_name
                    
                    f.write(f"  {traj_type}: {best_variant} (ISE/step: {best_ise:.6f})\n")
        
        print(f"✓ 汇总报告已保存: {report_file}")


def main():
    """主函数"""
    base_dir = Path(__file__).parent
    
    # 指定要测试的模型路径
    model_path = base_dir.parent / 'models_v4_optimized/stage3/Stage3_best.pt'
    
    # 创建对比器
    comparator = V4Comparator(str(base_dir))
    
    # 运行完整对比
    results = comparator.run_full_comparison(str(model_path))
    
    if results:
        print("\n✓ 所有测试完成！")
        print(f"结果保存在: {comparator.results_dir}")
    else:
        print("\n✗ 测试失败")


if __name__ == "__main__":
    main()
