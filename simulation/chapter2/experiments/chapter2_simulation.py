"""
第2章仿真脚本：系统建模与约束分析验证
生成论文所需的图2-1到图2-5
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'common'))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from platform_dynamics import ParallelPlatform3DOF
from wave_disturbance import WaveDisturbance

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Chapter2Simulation:
    """第2章仿真实验"""
    
    def __init__(self):
        self.platform = ParallelPlatform3DOF()
        self.figures_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
        os.makedirs(self.figures_dir, exist_ok=True)
        self.figures = []
        
    def plot_figure_2_1_platform_structure(self):
        """图2-1: 平台结构示意图"""
        fig = plt.figure(figsize=(14, 6))
        
        ax1 = fig.add_subplot(121, projection='3d')
        
        base_joints = self.platform.base_joints
        base_triangle = np.vstack([base_joints, base_joints[0]])
        ax1.plot(base_triangle[:, 0], base_triangle[:, 1], base_triangle[:, 2], 
                'b-', linewidth=2, label='Base Platform')
        ax1.scatter(base_joints[:, 0], base_joints[:, 1], base_joints[:, 2], 
                   c='blue', s=100, marker='o')
        
        q0 = np.array([0.529, 0, 0])
        platform_joints = self.platform.platform_joints_global(q0)
        platform_triangle = np.vstack([platform_joints, platform_joints[0]])
        ax1.plot(platform_triangle[:, 0], platform_triangle[:, 1], platform_triangle[:, 2], 
                'r-', linewidth=2, label='Moving Platform')
        ax1.scatter(platform_joints[:, 0], platform_joints[:, 1], platform_joints[:, 2], 
                   c='red', s=100, marker='s')
        
        for i in range(3):
            ax1.plot([base_joints[i, 0], platform_joints[i, 0]],
                    [base_joints[i, 1], platform_joints[i, 1]],
                    [base_joints[i, 2], platform_joints[i, 2]],
                    'g--', linewidth=1.5, alpha=0.7)
        
        ax1.plot([0, 0], [0, 0], [0, q0[0]], 'm-', linewidth=3, label='Passive P Joint')
        
        ax1.quiver(0, 0, 0, 0.2, 0, 0, color='r', arrow_length_ratio=0.3, linewidth=2)
        ax1.quiver(0, 0, 0, 0, 0.2, 0, color='g', arrow_length_ratio=0.3, linewidth=2)
        ax1.quiver(0, 0, 0, 0, 0, 0.2, color='b', arrow_length_ratio=0.3, linewidth=2)
        ax1.text(0.22, 0, 0, 'X', fontsize=12, color='red')
        ax1.text(0, 0.22, 0, 'Y', fontsize=12, color='green')
        ax1.text(0, 0, 0.22, 'Z', fontsize=12, color='blue')
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3-UPS/PU Parallel Platform Structure')
        ax1.legend()
        
        ax2 = fig.add_subplot(122)
        theta = np.linspace(0, 2*np.pi, 100)
        
        r_base = self.platform.r_base
        ax2.plot(r_base * np.cos(theta), r_base * np.sin(theta), 'b-', linewidth=2, label='Base')
        
        r_platform = self.platform.r_platform
        ax2.plot(r_platform * np.cos(theta), r_platform * np.sin(theta), 'r-', linewidth=2, label='Platform')
        
        base_joints_2d = base_joints[:, :2]
        platform_joints_2d = self.platform.platform_joints_local[:, :2]
        
        ax2.scatter(base_joints_2d[:, 0], base_joints_2d[:, 1], c='blue', s=150, marker='o', zorder=5)
        ax2.scatter(platform_joints_2d[:, 0], platform_joints_2d[:, 1], c='red', s=150, marker='s', zorder=5)
        
        ax2.annotate('', xy=(r_base, 0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
        ax2.text(r_base/2, 0.05, f'R_b={r_base}m', fontsize=10, color='blue')
        
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Top View with Dimensions')
        ax2.axis('equal')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'fig2_1_platform_structure.png'), 
                   dpi=300, bbox_inches='tight')
        self.figures.append('fig2_1_platform_structure.png')
        print("Generated: fig2_1_platform_structure.png")
        plt.close()
        
    def plot_figure_2_2_kinematics_validation(self):
        """图2-2: 运动学验证"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        t = np.linspace(0, 5, 500)
        z_des = 0.529 + 0.05 * np.sin(2*np.pi*0.5*t)
        alpha_des = 0.1 * np.sin(2*np.pi*0.3*t)
        beta_des = 0.08 * np.sin(2*np.pi*0.4*t + np.pi/4)
        
        leg_lengths = np.zeros((len(t), 3))
        for i in range(len(t)):
            q = np.array([z_des[i], alpha_des[i], beta_des[i]])
            leg_lengths[i] = self.platform.compute_leg_lengths(q)
        
        ax = axes[0, 0]
        ax.plot(t, z_des, 'b-', label='z (heave)', linewidth=2)
        ax.plot(t, np.degrees(alpha_des), 'r-', label='alpha (roll)', linewidth=2)
        ax.plot(t, np.degrees(beta_des), 'g-', label='beta (pitch)', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Displacement (m) / Angle (deg)')
        ax.set_title('Desired Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        ax.plot(t, leg_lengths[:, 0], 'b-', label='Leg 1', linewidth=2)
        ax.plot(t, leg_lengths[:, 1], 'r-', label='Leg 2', linewidth=2)
        ax.plot(t, leg_lengths[:, 2], 'g-', label='Leg 3', linewidth=2)
        ax.axhline(y=self.platform.l_leg_min, color='k', linestyle='--', alpha=0.5)
        ax.axhline(y=self.platform.l_leg_max, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Leg Length (m)')
        ax.set_title('Inverse Kinematics: Leg Lengths')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 0]
        cond_numbers = np.zeros(len(t))
        for i in range(len(t)):
            q = np.array([z_des[i], alpha_des[i], beta_des[i]])
            J = self.platform.jacobian(q)
            cond_numbers[i] = np.linalg.cond(J)
        ax.plot(t, cond_numbers, 'b-', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Condition Number')
        ax.set_title('Jacobian Condition Number')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        z_range = np.linspace(self.platform.z_min, self.platform.z_max, 30)
        alpha_range = np.linspace(-self.platform.alpha_max, self.platform.alpha_max, 30)
        Z, Alpha = np.meshgrid(z_range, alpha_range)
        workspace = np.zeros_like(Z, dtype=bool)
        
        for i in range(len(z_range)):
            for j in range(len(alpha_range)):
                q = np.array([Z[j, i], Alpha[j, i], 0])
                lengths = self.platform.compute_leg_lengths(q)
                workspace[j, i] = np.all((lengths >= self.platform.l_leg_min) & 
                                        (lengths <= self.platform.l_leg_max))
        
        ax.contourf(Z, np.degrees(Alpha), workspace.astype(int), levels=[0.5, 1.5], 
                   colors=['lightblue'], alpha=0.5)
        ax.contour(Z, np.degrees(Alpha), workspace.astype(int), levels=[0.5], 
                  colors='blue', linewidths=2)
        ax.set_xlabel('z (m)')
        ax.set_ylabel('alpha (deg)')
        ax.set_title('Workspace (beta=0)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'fig2_2_kinematics_validation.png'), 
                   dpi=300, bbox_inches='tight')
        self.figures.append('fig2_2_kinematics_validation.png')
        print("Generated: fig2_2_kinematics_validation.png")
        plt.close()
        
    def plot_figure_2_3_dynamics_validation(self):
        """图2-3: 动力学验证"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        dt = 0.001
        t = np.arange(0, 3, dt)
        
        ax = axes[0, 0]
        q0 = np.array([0.6, 0, 0])
        qd0 = np.array([0, 0, 0])
        tau = np.array([0, 0, 0])
        
        q_traj = np.zeros((len(t), 3))
        qd_traj = np.zeros((len(t), 3))
        q_traj[0] = q0
        qd_traj[0] = qd0
        
        for i in range(len(t)-1):
            qdd = self.platform.forward_dynamics(q_traj[i], qd_traj[i], tau)
            qd_traj[i+1] = qd_traj[i] + qdd * dt
            q_traj[i+1] = q_traj[i] + qd_traj[i] * dt
        
        z_theory = q0[0] - 0.5 * 9.81 * t**2
        ax.plot(t, q_traj[:, 0], 'b-', label='Simulation', linewidth=2)
        ax.plot(t, z_theory, 'r--', label='Theory (g=9.81)', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('z (m)')
        ax.set_title('Free Fall Test')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 0.7])
        
        ax = axes[0, 1]
        q0 = np.array([0.529, 0, 0])
        qd0 = np.array([0, 0, 0])
        F_amp = 100
        F_freq = 1.0
        q_traj = np.zeros((len(t), 3))
        qd_traj = np.zeros((len(t), 3))
        q_traj[0] = q0
        qd_traj[0] = qd0
        
        for i in range(len(t)-1):
            tau = np.array([F_amp * np.sin(2*np.pi*F_freq*t[i])] * 3)
            qdd = self.platform.forward_dynamics(q_traj[i], qd_traj[i], tau)
            qd_traj[i+1] = qd_traj[i] + qdd * dt
            q_traj[i+1] = q_traj[i] + qd_traj[i] * dt
        
        ax.plot(t, q_traj[:, 0] - q0[0], 'b-', label='z displacement', linewidth=2)
        ax.plot(t, np.degrees(q_traj[:, 1]), 'r-', label='alpha (roll)', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Displacement (m) / Angle (deg)')
        ax.set_title(f'Sinusoidal Response (F={F_amp}N)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 0]
        q0 = np.array([0.6, 0.1, 0])
        qd0 = np.array([0, 0.5, 0])
        tau = np.array([0, 0, 0])
        
        kinetic = np.zeros(len(t))
        potential = np.zeros(len(t))
        total = np.zeros(len(t))
        
        q_traj = np.zeros((len(t), 3))
        qd_traj = np.zeros((len(t), 3))
        q_traj[0] = q0
        qd_traj[0] = qd0
        
        ke, pe, te = self.platform.energy(q0, qd0)
        kinetic[0] = ke
        potential[0] = pe
        total[0] = te
        
        for i in range(len(t)-1):
            qdd = self.platform.forward_dynamics(q_traj[i], qd_traj[i], tau)
            qd_traj[i+1] = qd_traj[i] + qdd * dt
            q_traj[i+1] = q_traj[i] + qd_traj[i] * dt
            ke, pe, te = self.platform.energy(q_traj[i+1], qd_traj[i+1])
            kinetic[i+1] = ke
            potential[i+1] = pe
            total[i+1] = te
        
        ax.plot(t, kinetic, 'r-', label='Kinetic', linewidth=2)
        ax.plot(t, potential, 'b-', label='Potential', linewidth=2)
        ax.plot(t, total, 'g--', label='Total', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Energy (J)')
        ax.set_title('Energy Conservation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        q_test = np.array([0.529, 0.05, 0.03])
        qd_test = np.array([0.1, 0.2, 0.15])
        n_test = 50
        errors = []
        
        for _ in range(n_test):
            qdd_des = np.random.uniform(-1, 1, 3)
            tau = self.platform.inverse_dynamics(q_test, qd_test, qdd_des)
            qdd_computed = self.platform.forward_dynamics(q_test, qd_test, tau)
            error = np.linalg.norm(qdd_des - qdd_computed)
            errors.append(error)
        
        ax.bar(range(n_test), errors, color='blue', alpha=0.7)
        ax.axhline(y=np.mean(errors), color='r', linestyle='--', 
                  label=f'Mean Error: {np.mean(errors):.2e}')
        ax.set_xlabel('Test Case')
        ax.set_ylabel('Error Norm')
        ax.set_title('Forward/Inverse Dynamics Consistency')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'fig2_3_dynamics_validation.png'), 
                   dpi=300, bbox_inches='tight')
        self.figures.append('fig2_3_dynamics_validation.png')
        print("Generated: fig2_3_dynamics_validation.png")
        plt.close()
        
    def plot_figure_2_4_constraints_analysis(self):
        """图2-4: 约束分析"""
        fig = plt.figure(figsize=(14, 10))
        
        ax1 = fig.add_subplot(221, projection='3d')
        
        n_points = 1000
        z_samples = np.random.uniform(self.platform.z_min, self.platform.z_max, n_points)
        alpha_samples = np.random.uniform(-self.platform.alpha_max, self.platform.alpha_max, n_points)
        beta_samples = np.random.uniform(-self.platform.beta_max, self.platform.beta_max, n_points)
        
        feasible_points = []
        for i in range(n_points):
            q = np.array([z_samples[i], alpha_samples[i], beta_samples[i]])
            lengths = self.platform.compute_leg_lengths(q)
            if np.all((lengths >= self.platform.l_leg_min) & (lengths <= self.platform.l_leg_max)):
                feasible_points.append([z_samples[i], alpha_samples[i], beta_samples[i]])
        
        feasible_points = np.array(feasible_points)
        
        if len(feasible_points) > 0:
            ax1.scatter(feasible_points[:, 0], 
                       np.degrees(feasible_points[:, 1]), 
                       np.degrees(feasible_points[:, 2]),
                       c='blue', alpha=0.5, s=20)
        
        ax1.set_xlabel('z (m)')
        ax1.set_ylabel('alpha (deg)')
        ax1.set_zlabel('beta (deg)')
        ax1.set_title('3D Workspace')
        
        ax2 = fig.add_subplot(222)
        tau_range = np.linspace(self.platform.u_min, self.platform.u_max, 100)
        ax2.fill_between(tau_range/1000, 0, 1, alpha=0.3, color='green')
        ax2.axvline(x=self.platform.u_max/1000, color='r', linestyle='--', linewidth=2)
        ax2.axvline(x=self.platform.u_min/1000, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Force (kN)')
        ax2.set_ylabel('Normalized')
        ax2.set_title('Input Constraint')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([-6, 6])
        
        ax3 = fig.add_subplot(223)
        alpha_range = np.linspace(-np.pi/4, np.pi/4, 100)
        beta_range = np.linspace(-np.pi/4, np.pi/4, 100)
        Alpha, Beta = np.meshgrid(alpha_range, beta_range)
        
        constraint_mask = (np.abs(Alpha) <= self.platform.alpha_max) & \
                         (np.abs(Beta) <= self.platform.beta_max)
        
        ax3.contourf(np.degrees(Alpha), np.degrees(Beta), constraint_mask.astype(int), 
                    levels=[0.5, 1.5], colors=['lightgreen'], alpha=0.5)
        ax3.contour(np.degrees(Alpha), np.degrees(Beta), constraint_mask.astype(int), 
                   levels=[0.5], colors='green', linewidths=2)
        ax3.axvline(x=np.degrees(self.platform.alpha_max), color='r', linestyle='--', linewidth=2)
        ax3.axvline(x=-np.degrees(self.platform.alpha_max), color='r', linestyle='--', linewidth=2)
        ax3.axhline(y=np.degrees(self.platform.beta_max), color='r', linestyle='--', linewidth=2)
        ax3.axhline(y=-np.degrees(self.platform.beta_max), color='r', linestyle='--', linewidth=2)
        ax3.set_xlabel('alpha (deg)')
        ax3.set_ylabel('beta (deg)')
        ax3.set_title('Attitude Constraints')
        ax3.grid(True, alpha=0.3)
        
        ax4 = fig.add_subplot(224)
        z_range = np.linspace(0.3, 0.8, 100)
        feasible_z = ((z_range >= self.platform.z_min) & (z_range <= self.platform.z_max)).astype(float)
        
        ax4.fill_between(z_range, 0, feasible_z, alpha=0.3, color='blue')
        ax4.axvline(x=self.platform.z_min, color='r', linestyle='--', linewidth=2)
        ax4.axvline(x=self.platform.z_max, color='r', linestyle='--', linewidth=2)
        ax4.set_xlabel('z (m)')
        ax4.set_ylabel('Feasible')
        ax4.set_title('Heave Constraint')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'fig2_4_constraints_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        self.figures.append('fig2_4_constraints_analysis.png')
        print("Generated: fig2_4_constraints_analysis.png")
        plt.close()
        
    def plot_figure_2_5_wave_disturbance(self):
        """图2-5: 海浪扰动"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        wave_conditions = [
            {'Hs': 1.0, 'T1': 6.0, 'label': 'Calm (H=1m)'},
            {'Hs': 2.0, 'T1': 8.0, 'label': 'Moderate (H=2m)'},
            {'Hs': 4.0, 'T1': 10.0, 'label': 'Rough (H=4m)'}
        ]
        
        t = np.linspace(0, 60, 6000)
        
        ax = axes[0, 0]
        for condition in wave_conditions:
            wave = WaveDisturbance(Hs=condition['Hs'], 
                                  T1=condition['T1'],
                                  n_components=15,
                                  random_seed=42)
            disturbance = wave.generate_disturbance(t)
            ax.plot(t, disturbance[:, 0]/1000, label=condition['label'], linewidth=1.5)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Heave Force (kN)')
        ax.set_title('Wave Disturbance Time Domain')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 60])
        
        ax = axes[0, 1]
        omega = np.linspace(0.3, 3.0, 100)
        
        for condition in wave_conditions:
            wave = WaveDisturbance(Hs=condition['Hs'], 
                                  T1=condition['T1'],
                                  n_components=15,
                                  random_seed=42)
            S = wave._ittc_spectrum(omega)
            ax.plot(omega, S, label=condition['label'], linewidth=2)
        
        ax.set_xlabel('Angular Frequency (rad/s)')
        ax.set_ylabel('Spectral Density (m²·s)')
        ax.set_title('ITTC Wave Spectrum')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 0]
        wave = WaveDisturbance(Hs=2.0, T1=8.0, n_components=20, random_seed=42)
        disturbance = wave.generate_disturbance(t)
        
        ax.hist(disturbance[:, 0]/1000, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(x=np.mean(disturbance[:, 0]/1000), color='r', linestyle='--', 
                  linewidth=2, label=f'Mean: {np.mean(disturbance[:, 0]/1000):.2f} kN')
        ax.set_xlabel('Heave Force (kN)')
        ax.set_ylabel('Frequency')
        ax.set_title('Disturbance Distribution (H=2m)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        ax = axes[1, 1]
        wave = WaveDisturbance(Hs=2.0, T1=8.0, n_components=15, random_seed=42)
        disturbance = wave.generate_disturbance(t[:2000])
        
        ax.plot(t[:2000], disturbance[:, 0]/1000, 'b-', label='Heave', linewidth=1.5)
        ax.plot(t[:2000], disturbance[:, 1]/1000, 'r-', label='Roll', linewidth=1.5)
        ax.plot(t[:2000], disturbance[:, 2]/1000, 'g-', label='Pitch', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Force/Moment (kN / kNm)')
        ax.set_title('3-DOF Wave Disturbance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'fig2_5_wave_disturbance.png'), 
                   dpi=300, bbox_inches='tight')
        self.figures.append('fig2_5_wave_disturbance.png')
        print("Generated: fig2_5_wave_disturbance.png")
        plt.close()
        
    def plot_figure_2_6_coordinate_frames(self):
        """图2-6: 坐标系定义示意图"""
        fig = plt.figure(figsize=(14, 6))
        
        # 左侧：基座坐标系
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_xlim([-0.5, 0.5])
        ax1.set_ylim([-0.5, 0.5])
        ax1.set_zlim([0, 0.6])
        
        # 绘制基座平面
        xx, yy = np.meshgrid(np.linspace(-0.4, 0.4, 10), np.linspace(-0.4, 0.4, 10))
        zz = np.zeros_like(xx)
        ax1.plot_surface(xx, yy, zz, alpha=0.2, color='blue')
        
        # 基座坐标系 {B}
        ax1.quiver(0, 0, 0, 0.3, 0, 0, color='r', arrow_length_ratio=0.2, linewidth=3)
        ax1.quiver(0, 0, 0, 0, 0.3, 0, color='g', arrow_length_ratio=0.2, linewidth=3)
        ax1.quiver(0, 0, 0, 0, 0, 0.3, color='b', arrow_length_ratio=0.2, linewidth=3)
        ax1.text(0.32, 0, 0, 'Xb', fontsize=14, color='red', fontweight='bold')
        ax1.text(0, 0.32, 0, 'Yb', fontsize=14, color='green', fontweight='bold')
        ax1.text(0, 0, 0.32, 'Zb', fontsize=14, color='blue', fontweight='bold')
        
        # 标注基座中心
        ax1.scatter([0], [0], [0], color='black', s=100, marker='o')
        ax1.text(0.05, 0.05, 0.05, 'Ob (Base Origin)', fontsize=12)
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('Base Coordinate Frame {B}', fontsize=14, fontweight='bold')
        
        # 右侧：动平台坐标系
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.set_xlim([-0.5, 0.5])
        ax2.set_ylim([-0.5, 0.5])
        ax2.set_zlim([0.4, 1.0])
        
        q = np.array([0.529, 0.1, 0.05])
        R = self.platform.rotation_matrix(q[1], q[2])
        
        # 动平台中心
        center = np.array([0, 0, q[0]])
        
        # 动平台坐标系 {P}
        scale = 0.25
        x_axis = center + scale * R[:, 0]
        y_axis = center + scale * R[:, 1]
        z_axis = center + scale * R[:, 2]
        
        ax2.quiver(center[0], center[1], center[2],
                  x_axis[0]-center[0], x_axis[1]-center[1], x_axis[2]-center[2],
                  color='r', arrow_length_ratio=0.2, linewidth=3)
        ax2.quiver(center[0], center[1], center[2],
                  y_axis[0]-center[0], y_axis[1]-center[1], y_axis[2]-center[2],
                  color='g', arrow_length_ratio=0.2, linewidth=3)
        ax2.quiver(center[0], center[1], center[2],
                  z_axis[0]-center[0], z_axis[1]-center[1], z_axis[2]-center[2],
                  color='b', arrow_length_ratio=0.2, linewidth=3)
        
        ax2.text(x_axis[0]+0.02, x_axis[1], x_axis[2], 'Xp', fontsize=14, color='red', fontweight='bold')
        ax2.text(y_axis[0], y_axis[1]+0.02, y_axis[2], 'Yp', fontsize=14, color='green', fontweight='bold')
        ax2.text(z_axis[0], z_axis[1], z_axis[2]+0.02, 'Zp', fontsize=14, color='blue', fontweight='bold')
        
        # 标注动平台中心
        ax2.scatter([center[0]], [center[1]], [center[2]], color='black', s=100, marker='s')
        ax2.text(center[0]+0.05, center[1]+0.05, center[2]+0.05, 
                'Op (Platform Origin)', fontsize=12)
        
        # 绘制动平台
        platform_joints = self.platform.platform_joints_global(q)
        platform_triangle = np.vstack([platform_joints, platform_joints[0]])
        ax2.plot(platform_triangle[:, 0], platform_triangle[:, 1], platform_triangle[:, 2], 
                'r-', linewidth=2, alpha=0.5)
        
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_zlabel('Z (m)')
        ax2.set_title('Platform Coordinate Frame {P}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'fig2_6_coordinate_frames.png'), 
                   dpi=300, bbox_inches='tight')
        self.figures.append('fig2_6_coordinate_frames.png')
        print("Generated: fig2_6_coordinate_frames.png")
        plt.close()
        
    def plot_figure_2_7_actuator_model(self):
        """图2-7: 电动缸驱动模型"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 子图1: 电动缸示意图
        ax1 = axes[0, 0]
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.axis('off')
        ax1.set_title('Electric Cylinder Structure', fontsize=14, fontweight='bold')
        
        # 简化的电动缸示意图
        # 缸体
        rect1 = plt.Rectangle((0.2, 0.45), 0.4, 0.1, linewidth=2, 
                             edgecolor='blue', facecolor='lightblue', label='Cylinder Body')
        ax1.add_patch(rect1)
        # 活塞杆
        rect2 = plt.Rectangle((0.6, 0.47), 0.25, 0.06, linewidth=2,
                             edgecolor='red', facecolor='pink', label='Piston Rod')
        ax1.add_patch(rect2)
        # 电机
        circle = plt.Circle((0.15, 0.5), 0.08, linewidth=2,
                           edgecolor='green', facecolor='lightgreen', label='Motor')
        ax1.add_patch(circle)
        
        ax1.annotate('', xy=(0.85, 0.5), xytext=(0.6, 0.5),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax1.text(0.7, 0.55, 'F (Force)', fontsize=12, color='red')
        ax1.text(0.1, 0.35, 'Motor', fontsize=10, color='green')
        ax1.text(0.35, 0.35, 'Screw', fontsize=10, color='blue')
        ax1.text(0.7, 0.35, 'Rod', fontsize=10, color='red')
        
        ax1.legend(loc='upper right')
        
        # 子图2: 力-速度特性曲线
        ax2 = axes[0, 1]
        v = np.linspace(-0.5, 0.5, 100)  # 速度 m/s
        F_max = 5000  # 最大推力 N
        
        # 简化模型：力随速度线性减小
        F_available = F_max * (1 - np.abs(v) / 0.6)
        F_available = np.clip(F_available, 0, F_max)
        
        ax2.plot(v, F_available/1000, 'b-', linewidth=2, label='Available Force')
        ax2.axhline(y=F_max/1000, color='r', linestyle='--', label='Max Force', alpha=0.5)
        ax2.fill_between(v, 0, F_available/1000, alpha=0.3, color='blue')
        ax2.set_xlabel('Velocity (m/s)')
        ax2.set_ylabel('Force (kN)')
        ax2.set_title('Force-Velocity Characteristics')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([-0.5, 0.5])
        
        # 子图3: 执行器饱和特性
        ax3 = axes[1, 0]
        u_command = np.linspace(-8000, 8000, 200)  # 指令力
        u_max = 5000
        u_actual = np.clip(u_command, -u_max, u_max)  # 实际输出（饱和）
        
        ax3.plot(u_command/1000, u_actual/1000, 'b-', linewidth=2)
        ax3.plot(u_command/1000, u_command/1000, 'r--', linewidth=1, alpha=0.5, label='Linear (no saturation)')
        ax3.axvline(x=u_max/1000, color='g', linestyle=':', alpha=0.5)
        ax3.axvline(x=-u_max/1000, color='g', linestyle=':', alpha=0.5)
        ax3.fill_between(u_command/1000, u_actual/1000, u_command/1000, 
                        where=(np.abs(u_command) > u_max), alpha=0.3, color='red', label='Saturation')
        ax3.set_xlabel('Command Force (kN)')
        ax3.set_ylabel('Actual Force (kN)')
        ax3.set_title('Actuator Saturation')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 子图4: 支链长度与力关系
        ax4 = axes[1, 1]
        l_range = np.linspace(0.4, 0.7, 100)
        l_nominal = 0.529
        
        # 假设推力随长度变化（电机转矩恒定，丝杠传动比变化）
        F_effective = 5000 * (l_nominal / l_range)
        F_effective = np.clip(F_effective, 3000, 7000)
        
        ax4.plot(l_range*1000, F_effective/1000, 'b-', linewidth=2)
        ax4.axvline(x=self.platform.l_leg_min*1000, color='r', linestyle='--', alpha=0.5, label='Min/Max Length')
        ax4.axvline(x=self.platform.l_leg_max*1000, color='r', linestyle='--', alpha=0.5)
        ax4.axvline(x=l_nominal*1000, color='g', linestyle=':', alpha=0.5, label='Nominal')
        ax4.set_xlabel('Leg Length (mm)')
        ax4.set_ylabel('Effective Force (kN)')
        ax4.set_title('Force vs Leg Length')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'fig2_7_actuator_model.png'), 
                   dpi=300, bbox_inches='tight')
        self.figures.append('fig2_7_actuator_model.png')
        print("Generated: fig2_7_actuator_model.png")
        plt.close()
        
    def plot_figure_2_8_workspace_3d(self):
        """图2-8: 3D工作空间可视化"""
        fig = plt.figure(figsize=(14, 10))
        
        # 生成工作空间点
        n_points = 5000
        z_samples = np.random.uniform(self.platform.z_min, self.platform.z_max, n_points)
        alpha_samples = np.random.uniform(-self.platform.alpha_max, self.platform.alpha_max, n_points)
        beta_samples = np.random.uniform(-self.platform.beta_max, self.platform.beta_max, n_points)
        
        feasible_points = []
        for i in range(n_points):
            q = np.array([z_samples[i], alpha_samples[i], beta_samples[i]])
            lengths = self.platform.compute_leg_lengths(q)
            if np.all((lengths >= self.platform.l_leg_min) & (lengths <= self.platform.l_leg_max)):
                feasible_points.append([z_samples[i], alpha_samples[i], beta_samples[i]])
        
        feasible_points = np.array(feasible_points)
        
        # 主图：3D工作空间
        ax1 = fig.add_subplot(121, projection='3d')
        if len(feasible_points) > 0:
            scatter = ax1.scatter(feasible_points[:, 0], 
                                 np.degrees(feasible_points[:, 1]), 
                                 np.degrees(feasible_points[:, 2]),
                                 c=feasible_points[:, 0], cmap='viridis', 
                                 alpha=0.6, s=20)
            plt.colorbar(scatter, ax=ax1, label='z (m)')
        
        ax1.set_xlabel('z (m)')
        ax1.set_ylabel('α (deg)')
        ax1.set_zlabel('β (deg)')
        ax1.set_title('3D Workspace', fontsize=14, fontweight='bold')
        
        # 右侧：三个2D投影
        ax2 = fig.add_subplot(222)
        if len(feasible_points) > 0:
            ax2.scatter(feasible_points[:, 0], np.degrees(feasible_points[:, 1]), 
                       c='blue', alpha=0.5, s=10)
        ax2.set_xlabel('z (m)')
        ax2.set_ylabel('α (deg)')
        ax2.set_title('z-α Projection')
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(223)
        if len(feasible_points) > 0:
            ax3.scatter(feasible_points[:, 0], np.degrees(feasible_points[:, 2]), 
                       c='red', alpha=0.5, s=10)
        ax3.set_xlabel('z (m)')
        ax3.set_ylabel('β (deg)')
        ax3.set_title('z-β Projection')
        ax3.grid(True, alpha=0.3)
        
        ax4 = fig.add_subplot(224)
        if len(feasible_points) > 0:
            ax4.scatter(np.degrees(feasible_points[:, 1]), 
                       np.degrees(feasible_points[:, 2]), 
                       c='green', alpha=0.5, s=10)
        ax4.set_xlabel('α (deg)')
        ax4.set_ylabel('β (deg)')
        ax4.set_title('α-β Projection')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'fig2_8_workspace_3d.png'), 
                   dpi=300, bbox_inches='tight')
        self.figures.append('fig2_8_workspace_3d.png')
        print("Generated: fig2_8_workspace_3d.png")
        plt.close()
        
    def plot_figure_2_9_gravity_friction(self):
        """图2-9: 重力和摩擦力随姿态变化"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 参数范围
        alpha_range = np.linspace(-np.pi/6, np.pi/6, 100)
        beta_range = np.linspace(-np.pi/6, np.pi/6, 100)
        z_fixed = 0.529
        
        # 子图1: 重力随横摇变化
        ax1 = axes[0, 0]
        G_z = []
        for alpha in alpha_range:
            q = np.array([z_fixed, alpha, 0])
            G = self.platform.gravity_vector(q)
            G_z.append(G[0])
        ax1.plot(np.degrees(alpha_range), G_z, 'b-', linewidth=2)
        ax1.set_xlabel('Roll Angle α (deg)')
        ax1.set_ylabel('Gravity Force (N)')
        ax1.set_title('Gravity vs Roll Angle')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=self.platform.m_platform*9.81, color='r', 
                   linestyle='--', alpha=0.5, label='Nominal')
        ax1.legend()
        
        # 子图2: 重力随纵摇变化
        ax2 = axes[0, 1]
        G_z = []
        for beta in beta_range:
            q = np.array([z_fixed, 0, beta])
            G = self.platform.gravity_vector(q)
            G_z.append(G[0])
        ax2.plot(np.degrees(beta_range), G_z, 'b-', linewidth=2)
        ax2.set_xlabel('Pitch Angle β (deg)')
        ax2.set_ylabel('Gravity Force (N)')
        ax2.set_title('Gravity vs Pitch Angle')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=self.platform.m_platform*9.81, color='r', 
                   linestyle='--', alpha=0.5, label='Nominal')
        ax2.legend()
        
        # 子图3: 摩擦力随速度变化
        ax3 = axes[1, 0]
        qd_range = np.linspace(-0.5, 0.5, 100)
        F_coulomb = self.platform.params['F_coulomb']
        F_viscous = self.platform.params['F_viscous']
        
        F_friction = F_coulomb * np.sign(qd_range) + F_viscous * qd_range
        ax3.plot(qd_range, F_friction, 'b-', linewidth=2)
        ax3.set_xlabel('Velocity (m/s)')
        ax3.set_ylabel('Friction Force (N)')
        ax3.set_title('Friction Model')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=F_coulomb, color='r', linestyle='--', alpha=0.5, label='Coulomb')
        ax3.axhline(y=-F_coulomb, color='r', linestyle='--', alpha=0.5)
        ax3.legend()
        
        # 子图4: 惯量随姿态变化
        ax4 = axes[1, 1]
        I_eff = []
        for alpha in alpha_range:
            q = np.array([z_fixed, alpha, 0])
            M = self.platform.mass_matrix(q)
            I_eff.append(M[2, 2])  # 横摇惯量
        ax4.plot(np.degrees(alpha_range), I_eff, 'b-', linewidth=2, label='I_xx')
        
        I_eff = []
        for beta in beta_range:
            q = np.array([z_fixed, 0, beta])
            M = self.platform.mass_matrix(q)
            I_eff.append(M[2, 2])  # 纵摇惯量（注意质量矩阵定义）
        ax4.plot(np.degrees(beta_range), I_eff, 'r-', linewidth=2, label='I_yy')
        
        ax4.set_xlabel('Angle (deg)')
        ax4.set_ylabel('Effective Inertia (kg·m²)')
        ax4.set_title('Inertia vs Angle')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'fig2_9_gravity_friction.png'), 
                   dpi=300, bbox_inches='tight')
        self.figures.append('fig2_9_gravity_friction.png')
        print("Generated: fig2_9_gravity_friction.png")
        plt.close()
        
    def plot_figure_2_10_platform_parameters(self):
        """图2-10: 平台参数总结表格"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # 参数数据
        parameters = [
            ['Parameter', 'Symbol', 'Value', 'Unit', 'Description'],
            ['', '', '', '', ''],
            ['Mass', 'm', f'{self.platform.m_platform}', 'kg', 'Platform total mass'],
            ['Inertia X', 'I_xx', f'{self.platform.Ixx}', 'kg·m²', 'Roll moment of inertia'],
            ['Inertia Y', 'I_yy', f'{self.platform.Iyy}', 'kg·m²', 'Pitch moment of inertia'],
            ['Inertia Z', 'I_zz', f'{self.platform.Izz}', 'kg·m²', 'Yaw moment of inertia'],
            ['', '', '', '', ''],
            ['Base Radius', 'R_b', f'{self.platform.r_base}', 'm', 'Base joint distribution radius'],
            ['Platform Radius', 'R_t', f'{self.platform.r_platform}', 'm', 'Platform joint distribution radius'],
            ['Leg Min Length', 'l_min', f'{self.platform.l_leg_min}', 'm', 'Minimum leg length'],
            ['Leg Max Length', 'l_max', f'{self.platform.l_leg_max}', 'm', 'Maximum leg length'],
            ['Nominal Length', 'l_0', f'{self.platform.params["l_leg_nominal"]}', 'm', 'Nominal leg length'],
            ['', '', '', '', ''],
            ['Max Force', 'F_max', f'{self.platform.u_max/1000:.1f}', 'kN', 'Maximum actuator force'],
            ['Min Force', 'F_min', f'{self.platform.u_min/1000:.1f}', 'kN', 'Minimum actuator force'],
            ['Max Roll', 'α_max', f'{np.degrees(self.platform.alpha_max):.1f}', 'deg', 'Maximum roll angle'],
            ['Max Pitch', 'β_max', f'{np.degrees(self.platform.beta_max):.1f}', 'deg', 'Maximum pitch angle'],
            ['', '', '', '', ''],
            ['Coulomb Friction', 'F_c', f'{self.platform.params["F_coulomb"]}', 'N', 'Coulomb friction force'],
            ['Viscous Friction', 'F_v', f'{self.platform.params["F_viscous"]}', 'N·s/m', 'Viscous friction coefficient'],
        ]
        
        # 创建表格
        table = ax.table(cellText=parameters, cellLoc='left', loc='center',
                        colWidths=[0.2, 0.15, 0.15, 0.15, 0.35])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 设置表头样式
        for i in range(5):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 设置分隔行样式
        for i in [2, 8, 14]:
            for j in range(5):
                table[(i, j)].set_facecolor('#E8F5E9')
        
        # 设置空行
        for i in [1, 7, 13, 19]:
            for j in range(5):
                table[(i, j)].set_height(0.02)
        
        plt.title('3-UPS/PU Parallel Platform Parameters', 
                 fontsize=16, fontweight='bold', pad=20)
        
        plt.savefig(os.path.join(self.figures_dir, 'fig2_10_platform_parameters.png'), 
                   dpi=300, bbox_inches='tight')
        self.figures.append('fig2_10_platform_parameters.png')
        print("Generated: fig2_10_platform_parameters.png")
        plt.close()
        
    def run_all_simulations(self):
        """运行所有仿真"""
        print("\n" + "="*70)
        print("Chapter 2: System Modeling and Constraint Analysis")
        print("="*70 + "\n")
        
        self.plot_figure_2_1_platform_structure()
        self.plot_figure_2_2_kinematics_validation()
        self.plot_figure_2_3_dynamics_validation()
        self.plot_figure_2_4_constraints_analysis()
        self.plot_figure_2_5_wave_disturbance()
        self.plot_figure_2_6_coordinate_frames()
        self.plot_figure_2_7_actuator_model()
        self.plot_figure_2_8_workspace_3d()
        self.plot_figure_2_9_gravity_friction()
        self.plot_figure_2_10_platform_parameters()
        
        print("\n" + "="*70)
        print(f"All simulations completed! Generated {len(self.figures)} figures")
        print(f"Figures saved to: {self.figures_dir}")
        print("="*70 + "\n")
        
        for fig in self.figures:
            print(f"  - {fig}")


if __name__ == "__main__":
    sim = Chapter2Simulation()
    sim.run_all_simulations()
