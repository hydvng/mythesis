"""
Plot V5 Simplified control results - 3 subplots with unit conversion
- z: m -> mm
- alpha, beta: rad -> deg
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'common'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'env'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'agents'))

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from rl_env_v5_simplified import PlatformRLEnvV5Simplified
from v4_sac import V4SAC


def evaluate_model(model_path, q_des_type, Hs=2.0, T1=8.0, episode_length=30):
    """Evaluate model and return control results"""
    
    env = PlatformRLEnvV5Simplified(
        use_model_compensation=True,
        max_episode_steps=int(episode_length / 0.01),
        Hs=Hs,
        T1=T1,
        q_des_type=q_des_type,
        diverge_threshold=0.5,
        warning_penalty=0.0
    )
    
    # Create agent
    agent = V4SAC(
        state_dim=9,
        action_dim=3,
        hidden_dims=[512, 512, 256, 256],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Load model
    agent.load(model_path)
    
    # Run evaluation
    state = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state, deterministic=True)
        state, reward, done, info = env.step(action)
    
    history = env.get_history()
    
    return history, env


def plot_control_results(history, title, save_path):
    """Plot control results - 3 separate figures with unit conversion"""
    
    time = history['time']
    q = np.array(history['q'])
    q_des = np.array(history['q_des'])
    u = np.array(history['u'])
    ise = history['ise']
    
    # Unit conversion
    # z: m -> mm, alpha/beta: rad -> deg
    q_mm = q.copy()
    q_mm[:, 0] = q[:, 0] * 1000  # z: m -> mm
    q_des_mm = q_des.copy()
    q_des_mm[:, 0] = q_des[:, 0] * 1000  # z: m -> mm
    
    q_deg = q.copy()
    q_deg[:, 1:] = np.degrees(q[:, 1:])  # alpha, beta: rad -> deg
    q_des_deg = q_des.copy()
    q_des_deg[:, 1:] = np.degrees(q_des[:, 1:])  # alpha, beta: rad -> deg
    
    # ===== Figure 1: Position Tracking =====
    fig1, axes1 = plt.subplots(3, 1, figsize=(12, 8))
    
    # z position (mm)
    ax = axes1[0]
    ax.plot(time, q_mm[:, 0], 'b-', label='z (actual)', linewidth=1.5)
    ax.plot(time, q_des_mm[:, 0], 'b--', label='z (target)', linewidth=1, alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('z (mm)')
    ax.set_title(f'{title} - Heave (z) Position Tracking')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # alpha angle (deg)
    ax = axes1[1]
    ax.plot(time, q_deg[:, 1], 'r-', label='alpha (actual)', linewidth=1.5)
    ax.plot(time, q_des_deg[:, 1], 'r--', label='alpha (target)', linewidth=1, alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('alpha (deg)')
    ax.set_title(f'{title} - Roll (alpha) Angle Tracking')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # beta angle (deg)
    ax = axes1[2]
    ax.plot(time, q_deg[:, 2], 'g-', label='beta (actual)', linewidth=1.5)
    ax.plot(time, q_des_deg[:, 2], 'g--', label='beta (target)', linewidth=1, alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('beta (deg)')
    ax.set_title(f'{title} - Pitch (beta) Angle Tracking')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    pos_path = str(save_path).replace('.png', '_position.png')
    fig1.savefig(pos_path, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"Saved: {pos_path}")
    
    # ===== Figure 2: Tracking Error =====
    error_mm = q_mm - q_des_mm
    error_deg = q_deg - q_des_deg
    
    fig2, axes2 = plt.subplots(3, 1, figsize=(12, 8))
    
    # z error (mm)
    ax = axes2[0]
    ax.plot(time, error_mm[:, 0], 'b-', label='z error', linewidth=1.5)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('z error (mm)')
    ax.set_title(f'{title} - Heave Error (mm)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # alpha error (deg)
    ax = axes2[1]
    ax.plot(time, error_deg[:, 1], 'r-', label='alpha error', linewidth=1.5)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('alpha error (deg)')
    ax.set_title(f'{title} - Roll Error (deg)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # beta error (deg)
    ax = axes2[2]
    ax.plot(time, error_deg[:, 2], 'g-', label='beta error', linewidth=1.5)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('beta error (deg)')
    ax.set_title(f'{title} - Pitch Error (deg)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    err_path = str(save_path).replace('.png', '_error.png')
    fig2.savefig(err_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved: {err_path}")
    
    # ===== Figure 3: Control Input =====
    fig3, axes3 = plt.subplots(3, 1, figsize=(12, 8))
    
    ax = axes3[0]
    ax.plot(time, u[:, 0], 'b-', label='u1', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('u1 (N)')
    ax.set_title(f'{title} - Control Input u1')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    ax = axes3[1]
    ax.plot(time, u[:, 1], 'r-', label='u2', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('u2 (N)')
    ax.set_title(f'{title} - Control Input u2')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    ax = axes3[2]
    ax.plot(time, u[:, 2], 'g-', label='u3', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('u3 (N)')
    ax.set_title(f'{title} - Control Input u3')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    ctrl_path = str(save_path).replace('.png', '_control.png')
    fig3.savefig(ctrl_path, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"Saved: {ctrl_path}")
    
    # Print statistics
    final_error_mm = error_mm[-1]
    final_error_deg = error_deg[-1]
    cumulative_ise = np.cumsum(ise)
    
    print(f"Final Error: z={final_error_mm[0]:.3f}mm, alpha={final_error_deg[1]:.3f}deg, beta={final_error_deg[2]:.3f}deg")
    print(f"Final Cumulative ISE: {cumulative_ise[-1]:.6f}")
    print(f"Avg ISE/step: {np.mean(ise):.6f}")
    
    return cumulative_ise[-1], np.mean(ise)


if __name__ == '__main__':
    # Model path
    model_path = Path(__file__).parent / 'v5_simplified_training' / 'best_model.pt'
    
    # Output directory
    output_dir = Path(__file__).parent / 'v5_control_plots'
    output_dir.mkdir(exist_ok=True)
    
    # Test scenarios
    scenarios = [
        {'name': 'Sinusoidal', 'q_des_type': 'sinusoidal', 'Hs': 2.0},
        {'name': 'Sinusoidal_Small', 'q_des_type': 'sinusoidal_small', 'Hs': 2.0},
        {'name': 'Constant', 'q_des_type': 'constant', 'Hs': 2.0},
    ]
    
    results = {}
    
    for scenario in scenarios:
        print(f"\n{'='*50}")
        print(f"Evaluating: {scenario['name']}")
        print('='*50)
        
        history, env = evaluate_model(
            model_path=model_path,
            q_des_type=scenario['q_des_type'],
            Hs=scenario['Hs'],
            episode_length=30
        )
        
        save_path = output_dir / f"V5_{scenario['name']}.png"
        total_ise, ise_per_step = plot_control_results(
            history, 
            f"V5 Simplified - {scenario['name']}",
            save_path
        )
        
        results[scenario['name']] = {
            'total_ise': total_ise,
            'ise_per_step': ise_per_step
        }
    
    # Comparison plot
    print(f"\n{'='*50}")
    print("ISE Comparison Summary")
    print('='*50)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(results.keys())
    ise_values = [results[n]['ise_per_step'] * 1000 for n in names]
    
    bars = ax.bar(names, ise_values, color=['blue', 'green', 'orange'])
    
    ax.set_ylabel('ISE/step (x10^-3)')
    ax.set_title('V5 Simplified - ISE/step Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, ise_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'V5_ISE_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nComparison plot saved: {output_dir / 'V5_ISE_comparison.png'}")
    
    print("\nFinal Results:")
    for name in names:
        print(f"  {name}: ISE/step = {results[name]['ise_per_step']:.6f}")
