"""
海浪扰动模型 - 基于MSS真实RAO数据

物理链路（每一步都清晰）：
波面高度 η(t) 
    → [运动RAO] → 船体运动幅值 X(ω) = RAO_motion(ω) × η(ω)
    → [运动学] → 加速度 a(ω) = ω² × X(ω)
    → [牛顿定律] → 惯性力 F(ω) = m × a(ω)

数据来源: MSS (Marine Systems Simulator) - Fossen教授
使用真实的船模试验/CFD计算得到的运动RAO
"""

import numpy as np
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator
from typing import Dict, Optional, Union
import os


class WaveDisturbance:
    """
    海浪扰动模型 - 使用MSS真实运动RAO数据
    
    核心公式：
        F_disturbance(ω) = m × ω² × RAO_motion(ω) × η(ω)
        
    其中：
        - m: 平台质量 (kg)
        - ω: 波浪频率 (rad/s)
        - RAO_motion(ω): 运动响应幅值算子 (m/m 或 rad/m)
        - η(ω): 波面幅值 (m)
        
    物理意义:
        - 运动RAO将波面幅值转换为船体运动幅值
        - 通过ω²得到加速度
        - 惯性力 = 平台质量 × 加速度
    """
    
    SEA_STATE_TABLE = {
        0: {"Hs": 0.0,  "T1": 1.0,  "description": "Calm"},
        1: {"Hs": 0.3,  "T1": 3.0,  "description": "Light"},
        2: {"Hs": 0.6,  "T1": 4.5,  "description": "Moderate"},
        3: {"Hs": 1.0,  "T1": 6.0,  "description": "Rough"},
        4: {"Hs": 2.0,  "T1": 8.0,  "description": "Sea State 4"},
        5: {"Hs": 3.0,  "T1": 9.5,  "description": "Heavy"},
        6: {"Hs": 4.0,  "T1": 11.0, "description": "Very Heavy"},
        7: {"Hs": 5.5,  "T1": 12.5, "description": "Phenomenal"},
    }
    
    def __init__(self,
                 Hs: float = 2.0,
                 T1: float = 8.0,
                 platform_params: Optional[Dict] = None,
                 vessel_file: str = 'supply.mat',
                 wave_heading: float = 180.0,
                 vessel_speed: int = 0,
                 n_components: int = 50,
                 use_directional_spectrum: bool = False,
                 n_directions: int = 9,
                 spreading_exponent: int = 2,
                 random_seed: Optional[int] = None):
        """
        初始化扰动模型
        
        Args:
            Hs: 有义波高 (m)
            T1: 平均周期 (s)
            platform_params: 平台参数字典
            vessel_file: MSS船型文件名 (.mat格式)
            wave_heading: 浪向角 (度), 0=following, 180=head
            vessel_speed: 航速索引 (0=静止)
            n_components: 频率分量数
            use_directional_spectrum: 是否使用方向谱
            n_directions: 方向分量数 (奇数，如9, 15, 21)
            spreading_exponent: 方向分布指数 s (cos^2s)
            random_seed: 随机种子
        """
        self.Hs = Hs
        self.T1 = T1
        self.n_components = n_components
        self.use_directional_spectrum = use_directional_spectrum
        self.n_directions = n_directions if use_directional_spectrum else 1
        self.spreading_exponent = spreading_exponent
        self.rng = np.random.default_rng(random_seed)
        
        # 平台物理参数 (质量×10，尺寸×2，惯性矩×40)
        if platform_params is None:
            self.platform = {
                'm': 347.54,
                'Ixx': 60.64,
                'Iyy': 115.4,
                'r': 0.58
            }
        else:
            self.platform = platform_params
        
        # 波向角 (转换为弧度)
        self.wave_heading = np.deg2rad(wave_heading)
        self.vessel_speed_idx = vessel_speed
        
        # 加载MSS RAO数据
        self._load_mss_rao(vessel_file)
        
        # 预计算
        self._precompute()
    
    def _load_mss_rao(self, filename: str):
        """加载MSS运动RAO数据"""
        # 查找文件路径
        if os.path.exists(filename):
            filepath = filename
        else:
            # 在data目录中查找
            module_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(module_dir, 'data')
            filepath = os.path.join(data_dir, filename)
            if not os.path.exists(filepath):
                raise FileNotFoundError(
                    f"找不到船型文件: {filename}\n"
                    f"请确保文件存在于: {data_dir}\n"
                    f"或提供完整路径"
                )
        
        # 加载MAT文件
        data = loadmat(filepath)
        vessel = data['vessel']
        
        # 提取运动RAO
        motionRAO = vessel['motionRAO'][0, 0]
        self.rao_freqs = motionRAO['w'][0, 0].flatten()
        self.rao_headings = vessel['headings'][0, 0].flatten()
        self.rao_velocities = vessel['velocities'][0, 0].flatten()
        
        # 提取船体参数
        main = vessel['main'][0, 0]
        self.ship_params = {
            'name': str(main['name'][0, 0][0]),
            'Lpp': float(main['Lpp'][0, 0].flatten()[0]),
            'B': float(main['B'][0, 0].flatten()[0]),
            'T': float(main['T'][0, 0].flatten()[0]),
            'm': float(main['m'][0, 0].flatten()[0]),
            'GM_T': float(main['GM_T'][0, 0].flatten()[0]),
            'GM_L': float(main['GM_L'][0, 0].flatten()[0]),
        }
        
        # 创建6自由度的插值函数
        # MSS DOF: 0=Surge, 1=Sway, 2=Heave, 3=Roll, 4=Pitch, 5=Yaw
        # 我们只需要: 2=Heave(Fz), 3=Roll(M_alpha), 4=Pitch(M_beta)
        self.rao_interp = {}
        
        amp_data = motionRAO['amp'][0, 0]
        
        for dof in [2, 3, 4]:  # Heave, Roll, Pitch
            # 获取RAO表: (频率, 方向, 速度)
            rao_table = amp_data[0, dof][:, :, self.vessel_speed_idx]
            
            # 创建2D插值函数
            self.rao_interp[dof] = RegularGridInterpolator(
                (self.rao_freqs, self.rao_headings),
                rao_table,
                method='linear',
                bounds_error=False,
                fill_value=0.0
            )
        
        print(f"已加载船型: {self.ship_params['name']}")
        print(f"  Lpp={self.ship_params['Lpp']:.1f}m, "
              f"B={self.ship_params['B']:.1f}m, "
              f"T={self.ship_params['T']:.1f}m")
    
    def _precompute(self):
        """预计算频率和方向相关参数"""
        self.omega_min = max(0.2, self.rao_freqs[0])
        self.omega_max = min(3.0, self.rao_freqs[-1])
        self.omegas = np.linspace(self.omega_min, self.omega_max, self.n_components)
        self.domega = self.omegas[1] - self.omegas[0]
        
        # ITTC频率谱
        S_freq = self._ittc_spectrum(self.omegas)
        
        # 方向谱计算
        if self.use_directional_spectrum:
            # 生成方向角（以主方向为中心，左右对称）
            # 例如：n_directions=9，角度范围 [-pi/2, pi/2]
            half_width = np.pi / 2  # 90度范围
            self.mu_angles = np.linspace(-half_width, half_width, self.n_directions)
            
            # 计算方向分布函数 D(mu) = cos^2s(mu)
            s = self.spreading_exponent
            D = np.cos(self.mu_angles) ** (2 * s)
            
            # 归一化（确保积分=1）
            D = D / np.sum(D)
            
            # 方向谱: S(Omega, mu) = S(Omega) * D(mu)
            self.S = np.outer(S_freq, D)  # 形状: (n_components, n_directions)
            
            # 各方向绝对角度
            self.wave_directions = self.wave_heading + self.mu_angles
            
            # 为每个频率-方向组合生成随机相位
            self.phases = self.rng.uniform(0, 2*np.pi, (self.n_components, self.n_directions))
            
            print(f"使用方向谱: {self.n_directions} 个方向分量")
            print(f"方向分布: cos^{2*s}(mu), 归一化后总和={np.sum(D):.4f}")
        else:
            # 单方向谱（简化模型）
            self.S = S_freq
            self.wave_directions = np.array([self.wave_heading])
            self.phases = self.rng.uniform(0, 2*np.pi, self.n_components)
        
        # 计算RAO
        self.RAO = self._compute_RAO_directional()
    
    def _ittc_spectrum(self, omega):
        """ITTC双参数谱"""
        omega = np.clip(omega, 1e-6, None)
        S = (173.0 * self.Hs**2) / (self.T1**4 * omega**5) * \
            np.exp(-691.0 / (self.T1**4 * omega**4))
        return S
    
    def _compute_RAO_directional(self):
        """计算方向相关的RAO矩阵"""
        # RAO形状: (n_components, n_directions)
        RAO = {
            'Fz': np.zeros((self.n_components, self.n_directions)),
            'M_alpha': np.zeros((self.n_components, self.n_directions)),
            'M_beta': np.zeros((self.n_components, self.n_directions))
        }
        
        for i, w in enumerate(self.omegas):
            for j, direction in enumerate(self.wave_directions):
                # 查询插值函数
                # DOF 2=Heave, DOF 3=Roll, DOF 4=Pitch
                try:
                    heave_rao = self.rao_interp[2]([[w, direction]])[0]
                    roll_rao = self.rao_interp[3]([[w, direction]])[0]
                    pitch_rao = self.rao_interp[4]([[w, direction]])[0]
                except:
                    heave_rao = roll_rao = pitch_rao = 0.0
                
                RAO['Fz'][i, j] = heave_rao
                RAO['M_alpha'][i, j] = roll_rao
                RAO['M_beta'][i, j] = pitch_rao
        
        return RAO
    
    def generate_disturbance(self, t):
        """Generate disturbance forces/moments with optional directional spectrum."""
        n_steps = len(t)
        disturbance = np.zeros((n_steps, 3))
        
        m = self.platform['m']
        Ixx = self.platform['Ixx']
        Iyy = self.platform['Iyy']
        
        if self.use_directional_spectrum:
            # Directional spectrum: integrate over frequencies and directions
            for i, dof in enumerate(['Fz', 'M_alpha', 'M_beta']):
                RAO_matrix = self.RAO[dof]  # Shape: (n_components, n_directions)
                inertia = m if dof == 'Fz' else (Ixx if dof == 'M_alpha' else Iyy)
                F = np.zeros(n_steps)
                
                # Sum over frequencies
                for k in range(self.n_components):
                    omega_k = self.omegas[k]
                    
                    # Sum over directions
                    for d in range(self.n_directions):
                        # Wave amplitude from directional spectrum
                        A_kd = np.sqrt(2.0 * self.S[k, d] * self.domega)
                        
                        # Ship motion
                        X_kd = RAO_matrix[k, d] * A_kd
                        
                        # Acceleration
                        a_kd = omega_k**2 * X_kd
                        
                        # Inertial force
                        F_kd = inertia * a_kd
                        
                        # Time domain synthesis
                        F += F_kd * np.cos(omega_k * t + self.phases[k, d])
                
                disturbance[:, i] = F
        else:
            # Single direction (original simplified model)
            for i, dof in enumerate(['Fz', 'M_alpha', 'M_beta']):
                RAO = self.RAO[dof][:, 0]  # First (and only) direction
                inertia = m if dof == 'Fz' else (Ixx if dof == 'M_alpha' else Iyy)
                F = np.zeros(n_steps)
                
                for k in range(self.n_components):
                    omega_k = self.omegas[k]
                    
                    A_k = np.sqrt(2.0 * self.S[k] * self.domega)
                    X_k = RAO[k] * A_k
                    a_k = omega_k**2 * X_k
                    F_k = inertia * a_k
                    
                    F += F_k * np.cos(omega_k * t + self.phases[k])
                
                disturbance[:, i] = F
        
        return disturbance
    
    def get_rao_curve(self):
        """获取RAO曲线（用于绘图）"""
        return {
            'freq': self.omegas,
            'Heave': self.RAO['Fz'],
            'Roll': self.RAO['M_alpha'],
            'Pitch': self.RAO['M_beta'],
            'ship_name': self.ship_params['name'],
            'heading_deg': np.degrees(self.wave_heading)
        }
    
    def demonstrate_physics(self):
        """演示物理计算过程"""
        print("="*70)
        print("海浪扰动物理过程演示")
        print(f"船型: {self.ship_params['name']}")
        print(f"浪向: {np.degrees(self.wave_heading):.0f}° "
              f"({'迎浪' if abs(self.wave_heading - np.pi) < 0.1 else '其他'})")
        print("="*70)
        
        # 选择一个频率
        k_demo = self.n_components // 2
        omega_k = self.omegas[k_demo]
        
        print(f"\n【频率 ω = {omega_k:.2f} rad/s 处的物理量】\n")
        
        # 波幅
        A_k = np.sqrt(2.0 * self.S[k_demo] * self.domega)
        print(f"1. 波幅 A = {A_k:.3f} m")
        
        for dof in ['Fz', 'M_alpha', 'M_beta']:
            RAO = self.RAO[dof][k_demo]
            inertia = self.platform['m'] if dof == 'Fz' else \
                     (self.platform['Ixx'] if dof == 'M_alpha' else self.platform['Iyy'])
            unit = 'N' if dof == 'Fz' else 'N·m'
            motion_unit = 'm' if dof == 'Fz' else 'rad'
            
            # 各物理量
            X = RAO * A_k
            a = omega_k**2 * X
            F = inertia * a
            
            print(f"\n2. {dof}:")
            print(f"   运动RAO = {RAO:.4f} {motion_unit}/m")
            print(f"   → 船体运动 X = {X:.4f} {motion_unit}")
            print(f"   → 加速度 a = {a:.4f} {motion_unit}/s²")
            print(f"   → 惯性力 F = {F:.2f} {unit}")
        
        # 完整仿真
        print(f"\n" + "="*70)
        t = np.linspace(0, 100, 10000)
        disturbance = self.generate_disturbance(t)
        
        print("完整仿真统计（100秒）:")
        print(f"  Fz:      std = {np.std(disturbance[:,0]):.1f} N")
        print(f"  M_alpha: std = {np.std(disturbance[:,1]):.1f} N·m")
        print(f"  M_beta:  std = {np.std(disturbance[:,2]):.1f} N·m")
        print("="*70)


    def compare_headings(self, headings=[0, 45, 90, 135, 180], t_duration=100):
        """
        对比不同浪向角的扰动特性
        
        Args:
            headings: 浪向角列表（度）
            t_duration: 仿真时长（秒）
            
        Returns:
            dict: 各浪向角的扰动统计结果
        """
        import matplotlib.pyplot as plt
        
        t = np.linspace(0, t_duration, int(t_duration * 100))
        results = {}
        
        print("="*70)
        print(f"不同浪向角扰动对比 ({self.ship_params['name']}, 海况{self.Hs}m)")
        print("="*70)
        print(f"{'浪向角':<12} {'工况':<12} {'Fz(N)':<12} {'M_α(N·m)':<12} {'M_β(N·m)':<12}")
        print("-"*70)
        
        for h_deg in headings:
            # 创建该浪向的扰动模型
            wave_h = WaveDisturbance(
                Hs=self.Hs,
                T1=self.T1,
                platform_params=self.platform,
                vessel_file='supply.mat',
                wave_heading=h_deg,
                random_seed=42
            )
            
            dist = wave_h.generate_disturbance(t)
            
            # 统计
            Fz_std = np.std(dist[:,0])
            Ma_std = np.std(dist[:,1])
            Mb_std = np.std(dist[:,2])
            
            # 工况名称
            if h_deg == 0:
                condition = "随浪"
            elif h_deg == 90:
                condition = "横浪"
            elif h_deg == 180:
                condition = "迎浪"
            else:
                condition = f"斜浪{h_deg}°"
            
            results[h_deg] = {
                'Fz_std': Fz_std,
                'M_alpha_std': Ma_std,
                'M_beta_std': Mb_std,
                'disturbance': dist,
                'condition': condition
            }
            
            print(f"{h_deg:<12.0f} {condition:<12} {Fz_std:<12.2f} {Ma_std:<12.4f} {Mb_std:<12.4f}")
        
        print("="*70)
        
        # 绘制对比图
        self._plot_heading_comparison(results, t)
        
        return results
    
    def _plot_heading_comparison(self, results, t):
        """绘制浪向角对比图"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            
            # 颜色映射
            headings = sorted(results.keys())
            cmap = plt.get_cmap('tab10')
            colors = cmap(np.linspace(0, 1, len(headings)))
            
            # 只绘制前60秒
            mask = t <= 60
            t_plot = t[mask]
            
            for i, h_deg in enumerate(headings):
                res = results[h_deg]
                dist = res['disturbance']
                label = f"{h_deg}° ({res['condition']})"
                
                # Fz扰动
                axes[0, 0].plot(t_plot, dist[mask, 0], label=label, 
                               color=colors[i], linewidth=1.5, alpha=0.8)
                
                # M_alpha扰动
                axes[0, 1].plot(t_plot, dist[mask, 1], label=label,
                               color=colors[i], linewidth=1.5, alpha=0.8)
                
                # M_beta扰动
                axes[0, 2].plot(t_plot, dist[mask, 2], label=label,
                               color=colors[i], linewidth=1.5, alpha=0.8)
            
            axes[0, 0].set_ylabel('Fz (N)')
            axes[0, 0].set_title('Heave Disturbance', fontweight='bold')
            axes[0, 0].legend(loc='upper right', fontsize=8)
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].set_ylabel('M_α (N·m)')
            axes[0, 1].set_title('Roll Disturbance', fontweight='bold')
            axes[0, 1].legend(loc='upper right', fontsize=8)
            axes[0, 1].grid(True, alpha=0.3)
            
            axes[0, 2].set_ylabel('M_β (N·m)')
            axes[0, 2].set_title('Pitch Disturbance', fontweight='bold')
            axes[0, 2].legend(loc='upper right', fontsize=8)
            axes[0, 2].grid(True, alpha=0.3)
            
            # 柱状图对比
            Fz_values = [results[h]['Fz_std'] for h in headings]
            Ma_values = [results[h]['M_alpha_std'] for h in headings]
            Mb_values = [results[h]['M_beta_std'] for h in headings]
            labels = [f"{h}°\n({results[h]['condition']})" for h in headings]
            
            x = np.arange(len(headings))
            width = 0.6
            
            axes[1, 0].bar(x, Fz_values, width, color=colors)
            axes[1, 0].set_ylabel('Std (N)')
            axes[1, 0].set_title('Fz Disturbance (Std)', fontweight='bold')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(labels, fontsize=8)
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            
            axes[1, 1].bar(x, Ma_values, width, color=colors)
            axes[1, 1].set_ylabel('Std (N·m)')
            axes[1, 1].set_title('M_α Disturbance (Std)', fontweight='bold')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(labels, fontsize=8)
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            axes[1, 2].bar(x, Mb_values, width, color=colors)
            axes[1, 2].set_ylabel('Std (N·m)')
            axes[1, 2].set_title('M_β Disturbance (Std)', fontweight='bold')
            axes[1, 2].set_xticks(x)
            axes[1, 2].set_xticklabels(labels, fontsize=8)
            axes[1, 2].grid(True, alpha=0.3, axis='y')
            
            for ax in axes[0, :]:
                ax.set_xlabel('Time (s)')
            
            plt.suptitle(f'Disturbance Comparison Under Different Wave Headings\n'
                        f'({self.ship_params["name"]}, Sea State {self.Hs}m)', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            save_path = '/home/haydn/Documents/AERAOFMINE/mythesis/simulation/common/wave_heading_comparison.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n✅ 浪向对比图已保存: {save_path}")
            
        except Exception as e:
            print(f"绘图失败: {e}")


if __name__ == "__main__":
    # 使用MSS真实RAO数据
    print("\n" + "="*70)
    print("使用 MSS 真实运动RAO数据")
    print("="*70 + "\n")
    
    # 示例1: 单浪向演示
    print("【示例1】单浪向演示（迎浪180°）")
    wave = WaveDisturbance(
        Hs=2.0, 
        T1=8.0,
        vessel_file='supply.mat',
        wave_heading=180.0,
        random_seed=42
    )
    wave.demonstrate_physics()
    
    # 示例2: 多浪向对比
    print("\n" + "="*70)
    print("【示例2】多浪向对比")
    print("="*70)
    results = wave.compare_headings(headings=[0, 45, 90, 135, 180], t_duration=100)
